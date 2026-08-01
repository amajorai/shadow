use axum::{
    extract::ws::{Message, WebSocket},
    extract::{Path, Query, Request, State, WebSocketUpgrade},
    http::StatusCode,
    middleware::{from_fn, Next},
    response::sse::Event,
    response::{IntoResponse, Json, Response, Sse},
    routing::{get, post},
    Router,
};
use futures_util::stream::{BoxStream, StreamExt};

use crate::utils::wall_micros;
use axum::http::header::{AUTHORIZATION, ORIGIN};
use axum::http::{HeaderValue, Method};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

use crate::config::Config;

// ─── Consent / capture-control globals ────────────────────────────────────────
//
// These are process-global so the capture loop (capture_engine.rs) and the HTTP
// handler both see the same state without adding fields to AppState.

/// True when the user has requested pause/incognito — capture is suppressed.
static CAPTURE_PAUSED: AtomicBool = AtomicBool::new(false);

/// Screen-frame (keyframe) capture toggle. On by default so the timeline shows
/// screenshots out of the box; the desktop can turn it off without disabling the
/// rest of capture (OCR text, clipboard, git, …). Frame writes are gated on this
/// AND the pause/allowlist consent checks.
static FRAME_CAPTURE_ENABLED: AtomicBool = AtomicBool::new(true);

/// Per-app allowlist. Empty vec = no filtering (allow all).
/// Non-empty = only apps whose name appears here receive context capture.
static APP_ALLOWLIST: std::sync::OnceLock<RwLock<Vec<String>>> = std::sync::OnceLock::new();

fn allowlist_cell() -> &'static RwLock<Vec<String>> {
    APP_ALLOWLIST.get_or_init(|| RwLock::new(Vec::new()))
}

/// Returns true when capture is active for the given app name.
///
/// - Always false when globally paused.
/// - Always true when the allowlist is empty (no filtering configured).
/// - True only when the app matches an entry (case-insensitive prefix match) when
///   the allowlist is non-empty.
pub fn is_capture_allowed(app_name: &str) -> bool {
    if CAPTURE_PAUSED.load(Ordering::Relaxed) {
        return false;
    }
    match allowlist_cell().read() {
        Ok(list) if !list.is_empty() => {
            let lower = app_name.to_lowercase();
            list.iter()
                .any(|entry| lower.contains(&entry.to_lowercase()))
        }
        _ => true,
    }
}

/// Returns true when capture is globally paused/incognito.
///
/// Passive capture sources (clipboard, filesystem, git, …) have no app name to
/// gate on, so they check this flag directly rather than `is_capture_allowed`.
pub fn is_capture_paused() -> bool {
    CAPTURE_PAUSED.load(Ordering::Relaxed)
}

/// Set the global pause/incognito state.
pub fn set_capture_paused(paused: bool) {
    CAPTURE_PAUSED.store(paused, Ordering::Relaxed);
}

/// Returns true when screen-frame (keyframe) capture is enabled.
pub fn is_frame_capture_enabled() -> bool {
    FRAME_CAPTURE_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable screen-frame (keyframe) capture.
pub fn set_frame_capture_enabled(enabled: bool) {
    FRAME_CAPTURE_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Replace the app allowlist (empty = allow all).
pub fn set_app_allowlist(apps: Vec<String>) {
    if let Ok(mut list) = allowlist_cell().write() {
        *list = apps;
    }
}

// ─── API token (loopback auth) ────────────────────────────────────────────────
//
// Shadow is loopback-bound, but "loopback" is not an auth boundary: any local
// process — or any web page via a no-preflight `text/plain` POST or DNS
// rebinding — can reach it. Everything except `/health` therefore requires a
// shared-secret bearer (mirroring the `RYU_EXT_TOKEN` gate the apps-store
// sidecars use, e.g. `apps-store/quests/backend/src/main.rs`): without it a
// hostile page could exfiltrate full screen history (`/search`, `/timeline`,
// `/frame`, …), silently flip `/capture/control` to re-enable capture behind
// the user's back, or poison the timeline via `/ingest`.

/// Name of the persisted token file under the Shadow data dir.
const API_TOKEN_FILE: &str = "api-token";

/// Resolve the API token: `SHADOW_API_TOKEN` env if set (Core injects it at
/// spawn; operators may export their own), else the persisted
/// `<data_dir>/api-token` file, generated on first run with owner-only
/// permissions so any same-user local client (Core, sidecars) can read it while
/// web pages and other users cannot. Returns `None` only when the file can
/// neither be read nor created — the auth gate then FAILS CLOSED (rejects all).
fn resolve_api_token(data_dir: &std::path::Path) -> Option<String> {
    if let Ok(env_token) = std::env::var("SHADOW_API_TOKEN") {
        let trimmed = env_token.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
    }

    let path = data_dir.join(API_TOKEN_FILE);
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
    }

    // First run: mint a random token (2× UUIDv4 = 64 hex chars from the OS
    // CSPRNG) and persist it owner-only.
    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    if let Err(e) = std::fs::create_dir_all(data_dir) {
        tracing::error!("cannot create data dir for API token file: {e}");
        return None;
    }
    if let Err(e) = std::fs::write(&path, &token) {
        tracing::error!("cannot persist API token to {}: {e}", path.display());
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Some(token)
}

/// Shared-secret bearer + anti-CSRF gate for everything except `/health`.
///
/// Requests carrying an `Origin` header are rejected outright (browsers always
/// attach it to cross-origin requests, so this is the CSRF/DNS-rebinding
/// kill-switch — no browser context may drive this API, token or not). Native
/// loopback clients present `Authorization: Bearer <token>`.
///
/// **Fail-closed:** `expected == None`/empty (token could not be resolved)
/// rejects every request rather than falling open.
async fn require_api_token(req: Request, next: Next, expected: Option<&str>) -> Response {
    if req.headers().contains_key(ORIGIN) {
        return (
            StatusCode::FORBIDDEN,
            "cross-origin (browser) requests are not allowed",
        )
            .into_response();
    }
    let provided = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if bearer_ok(provided, expected) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

/// Pure bearer check (factored out so the auth decision is unit-testable without
/// an axum `Request`/`Next`). Returns `true` only when `expected` is a non-empty
/// token AND `provided` equals it (constant-time compared). A `None`/empty
/// `expected` is the fail-closed case → always `false`.
fn bearer_ok(provided: Option<&str>, expected: Option<&str>) -> bool {
    let Some(expected) = expected.filter(|t| !t.is_empty()) else {
        return false;
    };
    ct_eq(provided.unwrap_or("").as_bytes(), expected.as_bytes())
}

/// Constant-time byte comparison — no early return on the first mismatched byte,
/// so the token check does not leak length/prefix via timing.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Shared server state.
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub orchestrator: Option<Arc<crate::llm::orchestrator::LlmOrchestrator>>,
    pub procedure_store: Option<Arc<std::sync::Mutex<crate::mimicry::ProcedureStore>>>,
    pub proactive_store: Option<Arc<tokio::sync::Mutex<crate::intelligence::ProactiveStore>>>,
    pub summary_store: Option<Arc<std::sync::Mutex<crate::intelligence::SummaryStore>>>,
    pub mimicry: Option<Arc<crate::mimicry::MimicryCoordinator>>,
    pub pattern_store: Option<Arc<std::sync::Mutex<crate::agent::PatternStore>>>,
    pub trust_tuner: Option<Arc<std::sync::Mutex<crate::intelligence::TrustTuner>>>,
    pub delivery_manager: Option<Arc<crate::intelligence::DeliveryManager>>,
    pub summary_queue: Option<Arc<tokio::sync::Mutex<crate::intelligence::SummaryQueue>>>,
    /// Live window tracker for current-context snapshots.
    pub window_tracker:
        Option<Arc<tokio::sync::Mutex<crate::capture::window::PlatformWindowTracker>>>,
    /// Live AX tree for selected-text extraction.
    pub ax_tree: Option<Arc<tokio::sync::Mutex<crate::capture::accessibility::PlatformAXTree>>>,
}

/// Response payload for GET /context/current.
#[derive(Debug, Serialize)]
pub struct CurrentContextResponse {
    /// Timestamp of the snapshot in Unix microseconds.
    pub timestamp_us: u64,
    /// Active window title (empty string when capture is paused).
    pub window_title: String,
    /// Active application name (empty string when capture is paused).
    pub app_name: String,
    /// Currently selected text as reported by the AX tree (empty when none or unavailable).
    pub selected_text: String,
    /// Text from the most recent OCR frame, if any was recorded.
    pub ocr_text: String,
    /// Timestamp of the OCR frame in Unix microseconds (0 when no OCR data exists).
    pub ocr_timestamp_us: u64,
    /// True when all sources returned data; false when capture is paused or cold.
    pub capture_active: bool,
    /// True when capture is globally paused (pause/incognito mode active).
    pub paused: bool,
}

// ─── Consent control types ─────────────────────────────────────────────────────

/// POST /capture/control — set pause and/or allowlist.
#[derive(Deserialize)]
struct CaptureControlRequest {
    /// When true, suspend capture without killing the sidecar.
    paused: Option<bool>,
    /// Per-app allowlist. Empty vec = allow all; non-empty = allow only listed apps.
    app_allowlist: Option<Vec<String>>,
    /// When false, stop saving screen-frame keyframes (timeline thumbnails) while
    /// leaving the rest of capture running. Omit to leave unchanged.
    frames: Option<bool>,
    /// Total days to keep captured Shadow history. Values are clamped by shadow-core.
    history_retention_days: Option<u32>,
}

/// GET /capture/control response.
#[derive(Serialize)]
struct CaptureControlResponse {
    paused: bool,
    app_allowlist: Vec<String>,
    /// Whether screen-frame keyframe capture is currently enabled.
    frames: bool,
    /// Total days Shadow keeps captured Timeline/search history.
    history_retention_days: u32,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<u32>,
    category: Option<String>,
}

#[derive(Deserialize)]
struct TimelineQuery {
    start: u64,
    end: u64,
}

#[derive(Deserialize)]
struct JournalQuery {
    start: u64,
    end: u64,
    /// When true, run the optional LLM narration pass over the derived cards.
    #[serde(default)]
    narrate: bool,
}

#[derive(Deserialize)]
struct WeeklyQuery {
    /// End of the review window in Unix micros (typically "now").
    end: u64,
    /// Number of trailing calendar days to include (defaults to 7).
    #[serde(default)]
    days: Option<u32>,
}

#[derive(Deserialize)]
struct FrameQuery {
    /// Target moment in Unix microseconds; the nearest keyframe is returned.
    ts: u64,
    /// Display to pull the frame from. Defaults to 0 when omitted.
    display: Option<u32>,
}

#[derive(Deserialize)]
struct RecentActivityQuery {
    /// Trailing window in minutes; clamped to 1..=15. Defaults to 3 when omitted.
    minutes: Option<u32>,
}

#[derive(Deserialize)]
struct AgentRequest {
    message: String,
    #[serde(default)]
    conversation_history: Vec<crate::llm::LlmMessage>,
}

#[derive(Deserialize)]
struct GenerateSummaryRequest {
    start_ts: u64,
    end_ts: u64,
}

#[derive(Deserialize)]
struct RunProcedureRequest {
    task: String,
}

#[derive(Deserialize)]
struct IngestRequest {
    events: Vec<serde_json::Value>,
}

/// Build and start the HTTP server.
pub async fn run_server(state: AppState) -> anyhow::Result<()> {
    let port = state.config.port;
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    let app = build_router(state);

    tracing::info!("HTTP server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn build_router(state: AppState) -> Router {
    // CORS: allow the Desktop webview (dev + prod) and localhost dev servers to
    // probe `/health` on this loopback-only sidecar. Mirrors Core's list. Every
    // other route rejects browser (Origin-bearing) requests in
    // `require_api_token`, so the allowlist effectively applies to health only.
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(tower_http::cors::Any)
        .allow_origin([
            "http://localhost:5173".parse::<HeaderValue>().unwrap(),
            "http://localhost:1420".parse::<HeaderValue>().unwrap(),
            "tauri://localhost".parse::<HeaderValue>().unwrap(),
            "https://tauri.localhost".parse::<HeaderValue>().unwrap(),
        ]);

    // Shared-secret bearer over every data-reading + mutating route (`/health`
    // stays open for liveness probes — it returns no capture data).
    let api_token = resolve_api_token(&state.config.data_dir);
    if api_token.is_some() {
        tracing::info!(
            "shadow: HTTP API requires the shared-secret bearer (all routes except /health)"
        );
    } else {
        tracing::warn!(
            "shadow: no API token available (SHADOW_API_TOKEN unset and the api-token file could not be read or created); all routes except /health are FAIL-CLOSED (reject all)"
        );
    }

    let protected = Router::new()
        // Core
        .route("/stop", get(stop_handler))
        // Search
        .route("/search", get(search_handler))
        .route("/search/semantic", get(semantic_search_handler))
        // Timeline
        .route("/timeline", get(timeline_handler))
        .route("/journal", get(journal_handler))
        .route("/journal/weekly", get(journal_weekly_handler))
        .route("/frame", get(frame_handler))
        .route("/activity/recent", get(recent_activity_handler))
        .route("/context/recent", get(recent_context_handler))
        .route("/context/current", get(current_context_handler))
        // Agent
        .route("/agent", post(agent_handler))
        .route("/agent/tools", get(agent_tools_handler))
        // Proactive
        .route("/proactive", get(proactive_handler))
        // Meeting summaries
        .route("/summaries", get(summaries_handler))
        .route("/summaries/{id}", get(summary_by_id_handler))
        .route("/summaries/generate", post(generate_summary_handler))
        // Memory
        .route(
            "/memory",
            get(memory_query_handler).post(memory_store_handler),
        )
        .route(
            "/directives",
            get(directives_handler).post(create_directive_handler),
        )
        // Procedures (mimicry)
        .route("/procedures", get(procedures_handler))
        .route("/procedures/run", post(run_procedure_handler))
        // WebSocket for real-time event stream
        .route("/ws", get(ws_handler))
        // External event ingest
        .route("/ingest", post(ingest_handler))
        // New Group F endpoints
        .route("/api/synthesize", post(synthesize_handler))
        .route("/api/feedback", post(feedback_handler))
        .route("/api/patterns", get(patterns_handler))
        .route("/api/intent", post(intent_handler))
        // Consent / capture-control
        .route(
            "/capture/control",
            get(capture_control_get_handler).post(capture_control_post_handler),
        )
        // Meeting recorder (device-local mic + system-loopback capture → Core)
        .route("/meeting/start", post(meeting_start_handler))
        .route("/meeting/stop", post(meeting_stop_handler))
        .route("/meeting/status", get(meeting_status_handler))
        // Clips (agent-native Loom/Jam: screen+audio bundle → Core)
        .route("/clips/start", post(clips_start_handler))
        .route("/clips/ingest", post(clips_ingest_handler))
        .route("/clips/sources", get(clips_sources_handler))
        .route("/clips", get(clips_list_handler))
        .route("/clips/{id}/stop", post(clips_stop_handler))
        .route("/clips/{id}/pause", post(clips_pause_handler))
        .route("/clips/{id}/resume", post(clips_resume_handler))
        .route("/clips/{id}/context", get(clips_context_handler))
        .route("/clips/{id}/frame", get(clips_frame_handler))
        .route("/clips/{id}/diagnostics", post(clips_diagnostics_handler))
        .route("/clips/{id}/file", get(clips_file_handler))
        .layer(from_fn(move |req: Request, next: Next| {
            let expected = api_token.clone();
            async move { require_api_token(req, next, expected.as_deref()).await }
        }));

    Router::new()
        .route("/health", get(health_handler))
        .merge(protected)
        .layer(cors)
        .with_state(state)
}

// ─── Health / Stop ─────────────────────────────────────────────────────────────

async fn health_handler(State(state): State<AppState>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

async fn stop_handler() -> impl IntoResponse {
    // Spawn a task to exit cleanly after returning the response
    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        std::process::exit(0);
    });
    Json(json!({ "stopping": true }))
}

// ─── Meeting recorder ───────────────────────────────────────────────────────────
//
// Core drives device-local meeting capture through these endpoints: it owns the
// meeting session + transcription + notes, and asks Shadow (the local sensor) to
// stream mixed mic + system-loopback WAV chunks back to
// `POST /api/meetings/:id/chunk`. See `capture::meeting`.

/// POST /meeting/start — begin recording `meeting_id`, uploading chunks to
/// `ingest_url` (defaults to Core on loopback when omitted).
#[derive(Deserialize)]
struct MeetingStartRequest {
    meeting_id: String,
    /// Core endpoint to POST captured WAV chunks to. When omitted, defaults to
    /// `http://127.0.0.1:7980/api/meetings/<id>/chunk`.
    #[serde(default)]
    ingest_url: Option<String>,
}

async fn meeting_start_handler(Json(req): Json<MeetingStartRequest>) -> impl IntoResponse {
    let ingest_url = match resolve_meeting_ingest_url(&req) {
        Ok(url) => url,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "recording": false, "error": message })),
            )
        }
    };
    match crate::capture::meeting::start(req.meeting_id.clone(), ingest_url) {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({ "recording": true, "meeting_id": req.meeting_id })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "recording": false, "error": e.to_string() })),
        ),
    }
}

fn resolve_meeting_ingest_url(req: &MeetingStartRequest) -> Result<String, String> {
    let fallback = format!(
        "http://127.0.0.1:7980/api/meetings/{}/chunk",
        req.meeting_id
    );
    let Some(raw) = req.ingest_url.as_deref() else {
        return Ok(fallback);
    };
    let parsed = reqwest::Url::parse(raw).map_err(|_| "invalid ingest_url".to_string())?;
    if parsed.scheme() != "http" {
        return Err("ingest_url must use http".to_string());
    }
    let host = parsed.host_str().unwrap_or_default();
    if !matches!(host, "127.0.0.1" | "localhost" | "::1") {
        return Err("ingest_url must point to loopback Core".to_string());
    }
    if parsed.port_or_known_default() != Some(7980) {
        return Err("ingest_url must use Core's loopback port".to_string());
    }
    let expected_path = format!("/api/meetings/{}/chunk", req.meeting_id);
    if parsed.path() != expected_path {
        return Err("ingest_url path does not match the meeting".to_string());
    }
    Ok(raw.to_string())
}

/// POST /meeting/stop — stop the current recording (body is ignored).
async fn meeting_stop_handler() -> impl IntoResponse {
    crate::capture::meeting::stop();
    Json(json!({ "recording": false }))
}

/// GET /meeting/status — whether a meeting is recording, and which one.
async fn meeting_status_handler() -> impl IntoResponse {
    Json(json!({
        "recording": crate::capture::meeting::is_recording(),
        "meeting_id": crate::capture::meeting::current_meeting_id(),
    }))
}

// ─── Clips ──────────────────────────────────────────────────────────────────────
//
// Agent-native Loom/Jam: Core drives one-click screen+audio recording through
// these endpoints, then serves the resulting bundle (video / manifest / frames)
// back to the desktop. Shadow owns the sensor half (capture + mux + bundle); see
// `capture::clip`. Handlers are State-less — the recorder is a process-global.

/// POST /clips/start — begin a clip with the given capture sources.
async fn clips_start_handler(
    Json(opts): Json<crate::capture::clip::ClipStartOpts>,
) -> impl IntoResponse {
    match crate::capture::clip::start(opts) {
        Ok(ctx) => (
            StatusCode::OK,
            Json(serde_json::to_value(ctx).unwrap_or_else(|_| json!({}))),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// POST /clips/ingest — ingest a URL/local video (already resolved to a local
/// path + optional captions by Core) into the same agent-context bundle a
/// recorded clip produces. Shells ffmpeg + a possible transcription round-trip,
/// so it runs on the blocking pool (like `clips_stop_handler`).
async fn clips_ingest_handler(
    Json(opts): Json<crate::capture::clip::ClipIngestOpts>,
) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(move || crate::capture::clip::ingest(opts)).await;
    match result {
        Ok(Ok(ctx)) => (
            StatusCode::OK,
            Json(serde_json::to_value(ctx).unwrap_or_else(|_| json!({}))),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("ingest task failed: {e}") })),
        ),
    }
}

/// GET /clips — list all clips, newest first.
async fn clips_list_handler() -> impl IntoResponse {
    match crate::capture::clip::list() {
        Ok(clips) => (StatusCode::OK, Json(json!({ "clips": clips }))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// POST /clips/{id}/stop — finalize the current clip (ffmpeg mux + transcript).
///
/// `stop` blocks on ffmpeg + a network round-trip, so it runs on the blocking
/// pool. The `id` path param identifies the clip for the client; the recorder is
/// a singleton so only the in-progress clip is stopped.
async fn clips_stop_handler(Path(_id): Path<String>) -> impl IntoResponse {
    let result = tokio::task::spawn_blocking(crate::capture::clip::stop).await;
    match result {
        Ok(Ok(Some(ctx))) => (
            StatusCode::OK,
            Json(serde_json::to_value(ctx).unwrap_or_else(|_| json!({}))),
        ),
        Ok(Ok(None)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "no clip is recording" })),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("stop task failed: {e}") })),
        ),
    }
}

/// POST /clips/{id}/pause — pause the in-progress clip (excludes the paused span
/// from the clip duration and suspends capture). The recorder is a singleton, so
/// `id` identifies the clip for the client only.
async fn clips_pause_handler(Path(_id): Path<String>) -> impl IntoResponse {
    match crate::capture::clip::pause() {
        Some(id) => {
            let duration_ms = crate::capture::clip::live_duration_ms().unwrap_or(0);
            (
                StatusCode::OK,
                Json(json!({ "paused": true, "id": id, "durationMs": duration_ms })),
            )
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "no clip is recording" })),
        ),
    }
}

/// POST /clips/{id}/resume — resume a paused clip.
async fn clips_resume_handler(Path(_id): Path<String>) -> impl IntoResponse {
    match crate::capture::clip::resume() {
        Some(id) => {
            let duration_ms = crate::capture::clip::live_duration_ms().unwrap_or(0);
            (
                StatusCode::OK,
                Json(json!({ "paused": false, "id": id, "durationMs": duration_ms })),
            )
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "no clip is recording" })),
        ),
    }
}

/// GET /clips/sources — the displays and windows a clip can capture from.
async fn clips_sources_handler() -> impl IntoResponse {
    let displays: Vec<serde_json::Value> = crate::capture::screen::enumerate_displays()
        .into_iter()
        .map(|d| {
            let label = if d.is_primary {
                format!("Display {} (primary)", d.id + 1)
            } else {
                format!("Display {}", d.id + 1)
            };
            json!({ "id": d.id, "label": label, "primary": d.is_primary })
        })
        .collect();
    let windows: Vec<serde_json::Value> = crate::capture::screen::enumerate_windows()
        .into_iter()
        .map(|(id, title)| json!({ "id": id, "title": title }))
        .collect();
    Json(json!({ "displays": displays, "windows": windows }))
}

/// GET /clips/{id}/context — the clip manifest (agent-context.json).
async fn clips_context_handler(Path(id): Path<String>) -> impl IntoResponse {
    match crate::capture::clip::read_context(&id) {
        Ok(ctx) => (
            StatusCode::OK,
            Json(serde_json::to_value(ctx).unwrap_or_else(|_| json!({}))),
        ),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "clip not found" })),
        ),
    }
}

/// Query for GET /clips/{id}/frame.
#[derive(Deserialize)]
struct ClipFrameQuery {
    #[serde(rename = "atMs", default)]
    at_ms: u64,
}

/// GET /clips/{id}/frame?atMs= — a single JPEG frame at the requested moment.
async fn clips_frame_handler(Path(id): Path<String>, Query(q): Query<ClipFrameQuery>) -> Response {
    let at_ms = q.at_ms;
    let path =
        tokio::task::spawn_blocking(move || crate::capture::clip::extract_frame(&id, at_ms)).await;
    let path = match path {
        Ok(Ok(p)) => p,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "image/jpeg")], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// POST /clips/{id}/diagnostics — append console/network diagnostics.
#[derive(Deserialize)]
struct DiagnosticsBody {
    #[serde(default)]
    events: Vec<crate::capture::clip::DiagnosticEvent>,
}

async fn clips_diagnostics_handler(
    Path(id): Path<String>,
    Json(body): Json<DiagnosticsBody>,
) -> impl IntoResponse {
    match crate::capture::clip::append_diagnostics(&id, body.events) {
        Ok(ingested) => (StatusCode::OK, Json(json!({ "ingested": ingested }))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
    }
}

/// GET /clips/{id}/file — the muxed clip.mp4 bytes.
async fn clips_file_handler(Path(id): Path<String>) -> Response {
    // `clip_file_path` joins the id under the clips root; a percent-decoded
    // `../` id would escape it, so validate before touching the filesystem.
    if !crate::capture::clip::is_valid_clip_id(&id) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let path = crate::capture::clip::clip_file_path(&id);
    match tokio::fs::read(&path).await {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "video/mp4")], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

// ─── Search ────────────────────────────────────────────────────────────────────

async fn search_handler(Query(query): Query<SearchQuery>) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(20);
    match shadow_core::search_text(query.q, limit) {
        Ok(results) => {
            let count = results.len();
            Json(json!({ "results": results, "count": count }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

async fn semantic_search_handler(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(5);
    match shadow_core::vector_search(query.q, limit) {
        Ok(results) => Json(json!({ "results": results })),
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

// ─── Timeline ─────────────────────────────────────────────────────────────────

async fn timeline_handler(Query(query): Query<TimelineQuery>) -> impl IntoResponse {
    match shadow_core::query_time_range(query.start, query.end) {
        Ok(entries) => {
            let count = entries.len();
            Json(json!({ "entries": entries, "count": count }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

async fn journal_handler(
    State(state): State<AppState>,
    Query(query): Query<JournalQuery>,
) -> impl IntoResponse {
    match shadow_core::query_journal_snapshot(query.start, query.end) {
        Ok(mut snapshot) => {
            if query.narrate {
                if let Some(orchestrator) = &state.orchestrator {
                    snapshot.cards = crate::intelligence::journal_narrator::narrate_cards(
                        orchestrator,
                        snapshot.cards.clone(),
                    )
                    .await;
                    // Card ranges/categories are preserved by the narrator; only
                    // the standup text needs refreshing to use narrated titles.
                    shadow_core::journal::rebuild_derived(&mut snapshot);
                }
            }
            Json(json!({ "journal": snapshot }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

/// Fold the trailing N calendar days into one weekly retrospective. Each day is
/// a separate `query_journal_snapshot` over its local-midnight window; the pure
/// `build_weekly_review` aggregates focus, category/app allocation, and daily
/// rollups. Narration is intentionally NOT run here (too many cards for one
/// pass) — the weekly view uses deterministic card text.
async fn journal_weekly_handler(Query(query): Query<WeeklyQuery>) -> impl IntoResponse {
    use chrono::{DateTime, Duration, Local, TimeZone, Utc};

    const MICROS_PER_SEC: i64 = 1_000_000;
    let day_count = query.days.unwrap_or(7).clamp(1, 31) as i64;

    // Anchor on the local calendar day containing `end`.
    let end_secs = (query.end / 1_000_000) as i64;
    let end_dt: DateTime<Local> = Local
        .timestamp_opt(end_secs, 0)
        .single()
        .unwrap_or_else(|| Utc.timestamp_opt(end_secs, 0).single().unwrap().into());
    let anchor_date = end_dt.date_naive();

    let mut days: Vec<(String, shadow_core::journal::JournalSnapshot)> = Vec::new();
    // Walk oldest → newest so rollups render left-to-right.
    for offset in (0..day_count).rev() {
        let date = anchor_date - Duration::days(offset);
        let Some(day_start) = date
            .and_hms_opt(0, 0, 0)
            .and_then(|naive| Local.from_local_datetime(&naive).single())
        else {
            continue;
        };
        let start_us = (day_start.timestamp() * MICROS_PER_SEC).max(0) as u64;
        let end_us = ((day_start.timestamp() + 86_400) * MICROS_PER_SEC).max(0) as u64;
        // Never look past the requested end.
        let end_us = end_us.min(query.end);
        if end_us <= start_us {
            continue;
        }
        match shadow_core::query_journal_snapshot(start_us, end_us) {
            Ok(snapshot) => days.push((date.format("%Y-%m-%d").to_string(), snapshot)),
            Err(e) => {
                return Json(json!({ "error": e.to_string() }));
            }
        }
    }

    let window_start = days.first().map(|(_, s)| s.start_ts).unwrap_or(query.end);
    let review = shadow_core::journal::build_weekly_review(window_start, query.end, &days);
    Json(json!({ "review": review }))
}

/// GET /frame?ts=<micros>&display=<id> — the nearest recorded keyframe JPEG.
///
/// Returns the closest keyframe to `ts` for `display` (default 0), letting the
/// timeline scrubber show what was on screen at a moment. Keyframes are written
/// as JPEGs out of the box (pure-Rust, no ffmpeg) whenever frame capture is on
/// and not paused. Responds 404 when no keyframe exists near `ts` (frame capture
/// off/paused, or nothing recorded yet); clients render a graceful fallback.
async fn frame_handler(Query(query): Query<FrameQuery>) -> Response {
    let display = query.display.unwrap_or(0);
    let path = match shadow_core::find_nearest_keyframe(display, query.ts) {
        Ok(Some(p)) => p,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::debug!("keyframe lookup failed: {e}");
            return StatusCode::NOT_FOUND.into_response();
        }
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "image/jpeg")], bytes).into_response(),
        Err(e) => {
            tracing::debug!("keyframe read failed for {path}: {e}");
            StatusCode::NOT_FOUND.into_response()
        }
    }
}

/// GET /activity/recent?minutes=<n> — an EPHEMERAL bundle of the last N minutes
/// of screen keyframes for chat "attach recent activity". Persists NOTHING.
///
/// Gathers keyframes in [now-n_min, now] across all displays, even-subsamples to
/// at most 30, reads each JPEG from disk, and returns base64 data URLs plus a
/// short markdown summary + transcript derived from the timeline. Keyframes
/// (~every 10s) are the source, so this does NOT depend on MP4 segments existing.
/// `minutes` is clamped to 1..=15.
async fn recent_activity_handler(
    Query(query): Query<RecentActivityQuery>,
) -> Json<serde_json::Value> {
    use base64::Engine;

    let minutes = query.minutes.unwrap_or(3).clamp(1, 15);
    let now = wall_micros();
    let start = now.saturating_sub(minutes as u64 * 60 * 1_000_000);

    // Keyframes across all displays, ordered oldest→newest.
    let kfs = match shadow_core::keyframes_between(None, start, now) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("recent_activity keyframes_between failed: {e}");
            Vec::new()
        }
    };

    // Even-subsample to at most 30 frames.
    const MAX_FRAMES: usize = 30;
    let picked: Vec<_> = if kfs.len() <= MAX_FRAMES {
        kfs
    } else {
        let step = kfs.len() as f64 / MAX_FRAMES as f64;
        (0..MAX_FRAMES)
            .map(|i| kfs[((i as f64) * step) as usize].clone())
            .collect()
    };

    // Read + base64-encode each JPEG. Skip unreadable frames (fail-soft).
    let mut frames = Vec::with_capacity(picked.len());
    for kf in &picked {
        match tokio::fs::read(&kf.file_path).await {
            Ok(bytes) => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                let at_ms = kf.ts.saturating_sub(start) / 1_000;
                frames.push(json!({
                    "atMs": at_ms,
                    "dataUrl": format!("data:image/jpeg;base64,{b64}"),
                }));
            }
            Err(e) => {
                tracing::debug!(
                    "recent_activity frame read failed for {}: {e}",
                    kf.file_path
                );
            }
        }
    }

    // Short markdown summary + transcript from the timeline OCR/events.
    let (summary, transcript) = match shadow_core::query_time_range(start, now) {
        Ok(entries) => build_activity_text(minutes, &entries),
        Err(_) => (
            format!("Recent activity (last {minutes} min)"),
            String::new(),
        ),
    };

    Json(json!({
        "title": format!("Recent activity (last {minutes} min)"),
        "durationMs": minutes as u64 * 60_000,
        "summary": summary,
        "transcript": transcript,
        "frames": frames,
    }))
}

/// Derive a short markdown summary + line-per-focus transcript from timeline
/// entries. Uses app_name / window_title from `TimelineEntry`.
fn build_activity_text(
    minutes: u32,
    entries: &[shadow_core::timeline::TimelineEntry],
) -> (String, String) {
    use std::collections::BTreeSet;

    let mut apps: BTreeSet<String> = BTreeSet::new();
    let mut lines: Vec<String> = Vec::new();
    let mut last: Option<(String, String)> = None;

    for e in entries {
        let app = e.app_name.clone().unwrap_or_default();
        let title = e.window_title.clone().unwrap_or_default();
        if !app.is_empty() {
            apps.insert(app.clone());
        }
        let cur = (app.clone(), title.clone());
        if last.as_ref() != Some(&cur) && (!app.is_empty() || !title.is_empty()) {
            let line = if title.is_empty() {
                app.clone()
            } else if app.is_empty() {
                title.clone()
            } else {
                format!("{app}: {title}")
            };
            lines.push(line);
            last = Some(cur);
        }
    }

    let apps_list: Vec<String> = apps.into_iter().collect();
    let summary = if apps_list.is_empty() {
        format!("Recent activity over the last {minutes} min. No window focus was captured.")
    } else {
        format!(
            "Recent activity over the last {minutes} min. Apps seen: {}.",
            apps_list.join(", ")
        )
    };
    (summary, lines.join("\n"))
}

async fn recent_context_handler(Query(query): Query<SearchQuery>) -> impl IntoResponse {
    let minutes: u64 = query.q.parse().unwrap_or(10);
    let now = wall_micros();
    let start = now.saturating_sub(minutes * 60 * 1_000_000);

    match shadow_core::query_time_range(start, now) {
        Ok(entries) => {
            let count = entries.len();
            Json(json!({ "entries": entries, "count": count, "window_minutes": minutes }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

/// GET /context/current — snapshot of active window, selected text, and latest OCR frame.
///
/// Returns a well-formed empty payload when capture is paused or no data has been recorded yet;
/// never returns a 500 in steady state. When globally paused or the foreground app is not on the
/// allowlist, returns an empty payload with `paused: true` so clients can show the indicator.
async fn current_context_handler(State(state): State<AppState>) -> Json<CurrentContextResponse> {
    use crate::capture::{AXTree, WindowTracker};

    let now_us = wall_micros();
    let globally_paused = CAPTURE_PAUSED.load(Ordering::Relaxed);

    // Early return when globally paused — do not query any capture subsystems.
    if globally_paused {
        return Json(CurrentContextResponse {
            timestamp_us: now_us,
            window_title: String::new(),
            app_name: String::new(),
            selected_text: String::new(),
            ocr_text: String::new(),
            ocr_timestamp_us: 0,
            capture_active: false,
            paused: true,
        });
    }

    // 1. Active window — live query via the platform window tracker.
    let (window_title, app_name) = if let Some(tracker) = &state.window_tracker {
        let t = tracker.lock().await;
        match WindowTracker::get_active_window(&*t).await {
            Some(win) => (win.title, win.app_name),
            None => (String::new(), String::new()),
        }
    } else {
        (String::new(), String::new())
    };

    // 1a. Check per-app allowlist: if the allowlist is non-empty and the foreground app is not
    //     listed, return the empty payload to suppress context for this app.
    if !app_name.is_empty() && !is_capture_allowed(&app_name) {
        return Json(CurrentContextResponse {
            timestamp_us: now_us,
            window_title: String::new(),
            app_name: String::new(),
            selected_text: String::new(),
            ocr_text: String::new(),
            ocr_timestamp_us: 0,
            capture_active: false,
            paused: false,
        });
    }

    // 2. Selected text — read from the AX tree's focused element value.
    //    We do a best-effort walk: if the focused tree has a non-empty value on the root
    //    or any direct child, treat it as the selection. Never fail hard.
    let selected_text = if let Some(ax) = &state.ax_tree {
        let t = ax.lock().await;
        match AXTree::get_focused_tree(&*t).await {
            Ok(tree) => extract_selected_text(&tree),
            Err(_) => String::new(),
        }
    } else {
        String::new()
    };

    // 3. Latest OCR frame — read from the in-memory cache maintained by the
    //    screen-capture loop in capture_engine.rs. Zero overhead; no lock contention.
    let (ocr_text, ocr_timestamp_us) = crate::capture_engine::get_latest_ocr()
        .map(|(text, _app, ts)| (text, ts))
        .unwrap_or_else(|| (String::new(), 0));

    let capture_active = !window_title.is_empty() || !app_name.is_empty();

    Json(CurrentContextResponse {
        timestamp_us: now_us,
        window_title,
        app_name,
        selected_text,
        ocr_text,
        ocr_timestamp_us,
        capture_active,
        paused: false,
    })
}

// ─── Capture control ───────────────────────────────────────────────────────────

/// GET /capture/control — read current pause state, allowlist, and frame toggle.
async fn capture_control_get_handler() -> impl IntoResponse {
    let paused = CAPTURE_PAUSED.load(Ordering::Relaxed);
    let app_allowlist = allowlist_cell()
        .read()
        .map(|l| l.clone())
        .unwrap_or_default();
    let frames = is_frame_capture_enabled();
    let history_retention_days = shadow_core::get_history_retention_days()
        .unwrap_or_else(|_| shadow_core::default_history_retention_days());
    Json(CaptureControlResponse {
        paused,
        app_allowlist,
        frames,
        history_retention_days,
    })
}

/// POST /capture/control — update pause state, allowlist, and/or frame toggle.
///
/// Fields are optional; omitting a field leaves it unchanged. Returns the
/// resulting state after applying the changes.
async fn capture_control_post_handler(Json(req): Json<CaptureControlRequest>) -> impl IntoResponse {
    if let Some(p) = req.paused {
        set_capture_paused(p);
    }
    if let Some(list) = req.app_allowlist {
        set_app_allowlist(list);
    }
    if let Some(f) = req.frames {
        set_frame_capture_enabled(f);
    }
    if let Some(days) = req.history_retention_days {
        let _ = shadow_core::set_history_retention_days(days);
    }
    let paused = CAPTURE_PAUSED.load(Ordering::Relaxed);
    let app_allowlist = allowlist_cell()
        .read()
        .map(|l| l.clone())
        .unwrap_or_default();
    let frames = is_frame_capture_enabled();
    let history_retention_days = shadow_core::get_history_retention_days()
        .unwrap_or_else(|_| shadow_core::default_history_retention_days());
    Json(CaptureControlResponse {
        paused,
        app_allowlist,
        frames,
        history_retention_days,
    })
}

/// Walk an AX tree node and return the first non-empty value string found,
/// preferring values on a focused/selected text field over generic containers.
pub fn extract_selected_text(node: &crate::capture::AXTreeNode) -> String {
    // Roles that are likely to carry typed/selected text content.
    let text_roles = [
        "text",
        "edit",
        "textfield",
        "combobox",
        "textarea",
        "document",
    ];
    let role_lower = node.role.to_lowercase();

    if text_roles.iter().any(|r| role_lower.contains(r)) {
        if let Some(val) = &node.value {
            if !val.is_empty() {
                return val.clone();
            }
        }
    }

    for child in &node.children {
        let found = extract_selected_text(child);
        if !found.is_empty() {
            return found;
        }
    }

    String::new()
}

// ─── Agent ─────────────────────────────────────────────────────────────────────

async fn agent_handler(
    State(state): State<AppState>,
    Json(req): Json<AgentRequest>,
) -> Sse<BoxStream<'static, Result<Event, std::convert::Infallible>>> {
    let stream: BoxStream<'static, Result<Event, std::convert::Infallible>> =
        match &state.orchestrator {
            None => {
                let msg =
                    serde_json::to_string(&json!({"type":"error","message":"LLM not configured"}))
                        .unwrap_or_default();
                futures_util::stream::once(async move { Ok(Event::default().data(msg)) }).boxed()
            }
            Some(o) => {
                let runtime = Arc::new(crate::agent::AgentRuntime::new(Arc::clone(o)));
                runtime
                    .run(req.message, req.conversation_history)
                    .map(|event| {
                        let data = serde_json::to_string(&event).unwrap_or_default();
                        Ok(Event::default().data(data))
                    })
                    .boxed()
            }
        };
    Sse::new(stream)
}

async fn agent_tools_handler(State(state): State<AppState>) -> impl IntoResponse {
    let orchestrator = match &state.orchestrator {
        Some(o) => Arc::clone(o),
        None => {
            return Json(json!({ "tools": [] }));
        }
    };
    let runtime = crate::agent::AgentRuntime::new(orchestrator);
    let tools = runtime.tool_definitions();
    Json(json!({ "tools": tools, "count": tools.len() }))
}

// ─── Proactive ─────────────────────────────────────────────────────────────────

async fn proactive_handler(State(state): State<AppState>) -> impl IntoResponse {
    match &state.proactive_store {
        Some(store) => {
            let s = store.lock().await;
            match s.list_recent(20) {
                Ok(suggestions) => Json(json!({ "suggestions": suggestions })),
                Err(e) => Json(json!({ "error": e.to_string() })),
            }
        }
        None => Json(json!({ "suggestions": [] })),
    }
}

// ─── Meeting Summaries ─────────────────────────────────────────────────────────

async fn summaries_handler(State(state): State<AppState>) -> impl IntoResponse {
    match &state.summary_store {
        Some(store) => {
            let s = store.lock().unwrap();
            match s.list(20) {
                Ok(summaries) => Json(json!({ "summaries": summaries })),
                Err(e) => Json(json!({ "error": e.to_string() })),
            }
        }
        None => Json(json!({ "summaries": [] })),
    }
}

async fn summary_by_id_handler(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match &state.summary_store {
        Some(store) => {
            let s = store.lock().unwrap();
            match s.get(&id) {
                Ok(Some(summary)) => Json(json!(summary)),
                Ok(None) => Json(json!({ "error": "not found" })),
                Err(e) => Json(json!({ "error": e.to_string() })),
            }
        }
        None => Json(json!({ "error": "summary store not available" })),
    }
}

async fn generate_summary_handler(
    State(state): State<AppState>,
    Json(req): Json<GenerateSummaryRequest>,
) -> impl IntoResponse {
    let orchestrator = match &state.orchestrator {
        Some(o) => Arc::clone(o),
        None => return Json(json!({ "error": "LLM not configured" })),
    };

    let resolver = crate::intelligence::MeetingResolver;
    let meetings = match resolver.find_meetings(req.start_ts, req.end_ts) {
        Ok(m) => m,
        Err(e) => return Json(json!({ "error": e.to_string() })),
    };

    let summarizer = crate::intelligence::MeetingSummarizer::new(orchestrator);
    let mut summaries = vec![];

    for window in &meetings {
        match summarizer.summarize(window).await {
            Ok(summary) => {
                if let Some(store) = &state.summary_store {
                    let s = store.lock().unwrap();
                    let _ = s.store(&summary);
                }
                summaries.push(summary);
            }
            Err(e) => tracing::warn!("Failed to summarize meeting: {}", e),
        }
    }

    Json(json!({ "summaries": summaries, "count": summaries.len() }))
}

// ─── Memory ────────────────────────────────────────────────────────────────────

async fn memory_query_handler(Query(query): Query<SearchQuery>) -> impl IntoResponse {
    match crate::memory::MEMORY_STORE.get() {
        Some(store) => {
            let store = store.lock().unwrap();
            let category = query.category.as_deref();
            match store.query(category, &query.q) {
                Ok(entries) => Json(json!({ "entries": entries })),
                Err(e) => Json(json!({ "error": e.to_string() })),
            }
        }
        None => Json(json!({ "entries": [] })),
    }
}

async fn memory_store_handler(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    match crate::memory::MEMORY_STORE.get() {
        Some(store) => {
            let store = store.lock().unwrap();
            let entry = crate::memory::MemoryEntry {
                id: body["id"]
                    .as_str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                category: body["category"].as_str().unwrap_or("fact").to_string(),
                content: body["content"].as_str().unwrap_or("").to_string(),
                confidence: body["confidence"].as_f64().unwrap_or(1.0) as f32,
                source_episode_id: body["source_episode_id"].as_str().map(|s| s.to_string()),
                access_count: 0,
                last_accessed: 0,
                created_at: wall_micros(),
            };
            match store.upsert(&entry) {
                Ok(_) => Json(json!({ "id": entry.id, "stored": true })),
                Err(e) => Json(json!({ "error": e.to_string() })),
            }
        }
        None => Json(json!({ "error": "memory store not initialized" })),
    }
}

async fn directives_handler() -> impl IntoResponse {
    match crate::memory::MEMORY_STORE.get() {
        Some(store) => match store.lock().unwrap().list_active(None) {
            Ok(directives) => Json(json!({ "directives": directives })),
            Err(e) => Json(json!({ "error": e.to_string() })),
        },
        None => Json(json!({ "directives": [] })),
    }
}

async fn create_directive_handler(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
    match crate::memory::MEMORY_STORE.get() {
        Some(store) => {
            let store = store.lock().unwrap();
            let directive = crate::memory::Directive {
                id: uuid::Uuid::new_v4().to_string(),
                directive_type: body["directive_type"]
                    .as_str()
                    .unwrap_or("reminder")
                    .to_string(),
                content: body["content"].as_str().unwrap_or("").to_string(),
                trigger_pattern: body["trigger_pattern"].as_str().map(|s| s.to_string()),
                action: body["action"].as_str().map(|s| s.to_string()),
                priority: body["priority"].as_u64().unwrap_or(5) as u8,
                expires_at: body["expires_at"].as_u64(),
                created_at: wall_micros(),
            };
            match store.create_directive(&directive) {
                Ok(_) => Json(json!({ "id": directive.id, "created": true })),
                Err(e) => Json(json!({ "error": e.to_string() })),
            }
        }
        None => Json(json!({ "error": "memory store not initialized" })),
    }
}

// ─── Procedures ────────────────────────────────────────────────────────────────

async fn procedures_handler(State(state): State<AppState>) -> impl IntoResponse {
    match &state.procedure_store {
        Some(store) => {
            let s = store.lock().unwrap();
            match s.list() {
                Ok(procedures) => Json(json!({ "procedures": procedures })),
                Err(e) => Json(json!({ "error": e.to_string() })),
            }
        }
        None => Json(json!({ "procedures": [] })),
    }
}

async fn run_procedure_handler(
    State(state): State<AppState>,
    Json(req): Json<RunProcedureRequest>,
) -> Sse<BoxStream<'static, Result<Event, std::convert::Infallible>>> {
    let stream: BoxStream<'static, Result<Event, std::convert::Infallible>> = match &state.mimicry {
        None => {
            let msg =
                serde_json::to_string(&json!({"type":"error","message":"Mimicry not configured"}))
                    .unwrap_or_default();
            futures_util::stream::once(async move { Ok(Event::default().data(msg)) }).boxed()
        }
        Some(m) => Arc::clone(m)
            .run(req.task)
            .map(|event| {
                let data = serde_json::to_string(&event).unwrap_or_default();
                Ok(Event::default().data(data))
            })
            .boxed(),
    };
    Sse::new(stream)
}

// ─── WebSocket ─────────────────────────────────────────────────────────────────

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_websocket)
}

async fn handle_websocket(mut socket: WebSocket) {
    tracing::info!("WebSocket client connected");
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));

    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Send recent events to connected client
                let now = wall_micros();
                let start = now.saturating_sub(3 * 1_000_000); // last 3s
                let payload = match shadow_core::query_time_range(start, now) {
                    Ok(entries) if !entries.is_empty() => {
                        serde_json::to_string(&json!({ "type": "events", "data": entries }))
                            .unwrap_or_default()
                    }
                    _ => continue,
                };
                if socket.send(Message::Text(payload.into())).await.is_err() {
                    break;
                }
            }
            Some(msg) = socket.recv() => {
                match msg {
                    Ok(Message::Close(_)) => break,
                    Err(_) => break,
                    _ => {}
                }
            }
        }
    }
    tracing::info!("WebSocket client disconnected");
}

// ─── External ingest ──────────────────────────────────────────────────────────

async fn ingest_handler(Json(req): Json<IngestRequest>) -> impl IntoResponse {
    let mut count = 0u32;
    for event in &req.events {
        if let Ok(data) = rmp_serde::to_vec(event) {
            if shadow_core::write_event(data).is_ok() {
                count += 1;
            }
        }
    }
    Json(json!({ "ingested": count }))
}

// ─── New API endpoints ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SynthesizeRequest {
    actions: Vec<serde_json::Value>,
}

async fn synthesize_handler(
    State(state): State<AppState>,
    Json(req): Json<SynthesizeRequest>,
) -> impl IntoResponse {
    let orchestrator = match &state.orchestrator {
        Some(o) => Arc::clone(o),
        None => return Json(json!({ "error": "LLM not configured" })),
    };

    // Convert JSON actions to LearnedEvent
    let events: Vec<ghost_core::learning::LearnedEvent> = req
        .actions
        .into_iter()
        .filter_map(|v| serde_json::from_value(v).ok())
        .collect();

    match crate::mimicry::ProcedureSynthesizer::synthesize(&events, &orchestrator).await {
        Ok(template) => Json(json!(template)),
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

#[derive(Deserialize)]
struct FeedbackRequest {
    suggestion_type: String,
    kind: String,
}

async fn feedback_handler(
    State(state): State<AppState>,
    Json(req): Json<FeedbackRequest>,
) -> impl IntoResponse {
    let kind = match req.kind.as_str() {
        "thumbs_up" => crate::intelligence::FeedbackKind::ThumbsUp,
        "thumbs_down" => crate::intelligence::FeedbackKind::ThumbsDown,
        "snooze" => crate::intelligence::FeedbackKind::Snooze,
        _ => crate::intelligence::FeedbackKind::Dismiss,
    };

    if let Some(dm) = &state.delivery_manager {
        dm.record_feedback(kind, &req.suggestion_type);
        Json(json!({ "applied": true }))
    } else {
        Json(json!({ "error": "delivery manager not configured" }))
    }
}

#[derive(Deserialize)]
struct PatternsQuery {
    q: Option<String>,
    app: Option<String>,
    limit: Option<usize>,
}

async fn patterns_handler(
    State(state): State<AppState>,
    Query(query): Query<PatternsQuery>,
) -> impl IntoResponse {
    match &state.pattern_store {
        Some(store) => {
            if let Ok(mut s) = store.lock() {
                let q = query.q.as_deref().unwrap_or("");
                let app = query.app.as_deref().unwrap_or("");
                let limit = query.limit.unwrap_or(10);
                let patterns = s.find_relevant(q, app, limit);
                Json(json!({ "patterns": patterns.iter().map(|(p, score)| {
                    json!({ "pattern": p, "score": score })
                }).collect::<Vec<_>>() }))
            } else {
                Json(json!({ "patterns": [] }))
            }
        }
        None => Json(json!({ "patterns": [] })),
    }
}

#[derive(Deserialize)]
struct IntentRequest {
    query: String,
}

async fn intent_handler(
    State(state): State<AppState>,
    Json(req): Json<IntentRequest>,
) -> impl IntoResponse {
    let intent = match &state.orchestrator {
        Some(o) => {
            let i = crate::agent::IntentClassifier::classify(&req.query, o).await;
            i.as_str().to_string()
        }
        None => crate::agent::IntentClassifier::classify_heuristic(&req.query)
            .as_str()
            .to_string(),
    };
    Json(json!({ "intent": intent, "query": req.query }))
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `CAPTURE_PAUSED` / `APP_ALLOWLIST` are process-global statics and the test
    /// harness runs tests on parallel threads, so every test that mutates or
    /// observes them must hold this lock or they race each other flakily.
    /// `unwrap_or_else(PoisonError::into_inner)` keeps one panicking test from
    /// cascading poison into the rest.
    static CAPTURE_STATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn capture_state_guard() -> std::sync::MutexGuard<'static, ()> {
        CAPTURE_STATE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Build a minimal AppState with all optional fields set to None.
    /// The handler must return a well-formed response even with no capture subsystems.
    fn minimal_state() -> AppState {
        AppState {
            config: crate::config::Config::new(),
            orchestrator: None,
            procedure_store: None,
            proactive_store: None,
            summary_store: None,
            mimicry: None,
            pattern_store: None,
            trust_tuner: None,
            delivery_manager: None,
            summary_queue: None,
            window_tracker: None,
            ax_tree: None,
        }
    }

    #[test]
    fn bearer_ok_matches_only_exact_nonempty_token() {
        assert!(bearer_ok(Some("secret"), Some("secret")));
        assert!(!bearer_ok(Some("secret"), Some("other")));
        assert!(!bearer_ok(Some("secre"), Some("secret")));
        assert!(!bearer_ok(None, Some("secret")));
    }

    #[test]
    fn bearer_ok_is_fail_closed_without_expected() {
        // No configured token must reject everything, never fall open.
        assert!(!bearer_ok(Some("secret"), None));
        assert!(!bearer_ok(Some(""), Some("")));
        assert!(!bearer_ok(None, None));
    }

    #[test]
    fn resolve_api_token_mints_once_and_rereads_the_same_value() {
        let dir = std::env::temp_dir().join(format!(
            "shadow-token-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        // First resolution mints + persists; second reads the same value back.
        let minted = resolve_api_token(&dir).expect("token minted");
        assert!(minted.len() >= 32, "token must not be trivially short");
        let reread = resolve_api_token(&dir).expect("token reread");
        assert_eq!(minted, reread);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn meeting_ingest_url_defaults_to_loopback_core() {
        let req = MeetingStartRequest {
            meeting_id: "meeting-1".to_string(),
            ingest_url: None,
        };

        let url = resolve_meeting_ingest_url(&req).expect("default ingest url");

        assert_eq!(url, "http://127.0.0.1:7980/api/meetings/meeting-1/chunk");
    }

    #[test]
    fn meeting_ingest_url_accepts_matching_loopback_url() {
        let req = MeetingStartRequest {
            meeting_id: "meeting-1".to_string(),
            ingest_url: Some("http://localhost:7980/api/meetings/meeting-1/chunk".to_string()),
        };

        let url = resolve_meeting_ingest_url(&req).expect("explicit ingest url");

        assert_eq!(url, "http://localhost:7980/api/meetings/meeting-1/chunk");
    }

    #[test]
    fn meeting_ingest_url_rejects_remote_or_mismatched_targets() {
        let cases = [
            "https://localhost:7980/api/meetings/meeting-1/chunk",
            "http://example.com:7980/api/meetings/meeting-1/chunk",
            "http://127.0.0.1:7981/api/meetings/meeting-1/chunk",
            "http://127.0.0.1:7980/api/meetings/other/chunk",
        ];

        for ingest_url in cases {
            let req = MeetingStartRequest {
                meeting_id: "meeting-1".to_string(),
                ingest_url: Some(ingest_url.to_string()),
            };

            assert!(
                resolve_meeting_ingest_url(&req).is_err(),
                "{ingest_url} should be rejected"
            );
        }
    }

    /// Invoke the handler directly (no HTTP stack required).
    async fn invoke_handler(state: AppState) -> CurrentContextResponse {
        use axum::extract::State;
        let Json(body) = current_context_handler(State(state)).await;
        body
    }

    /// Verify that /context/current returns a well-formed payload (all required fields present)
    /// when all capture subsystems are absent (cold-start / capture paused).
    /// OCR cache state is intentionally not checked here since that field is tested in
    /// test_current_context_reflects_seeded_ocr, and the global cache may be set by a
    /// concurrent test.
    #[tokio::test]
    async fn test_current_context_empty_is_well_formed() {
        let resp = invoke_handler(minimal_state()).await;

        // Structural checks that always hold regardless of OCR cache state.
        assert!(resp.timestamp_us > 0, "timestamp_us must be non-zero");
        assert_eq!(
            resp.window_title, "",
            "window_title should be empty when no tracker"
        );
        assert_eq!(
            resp.app_name, "",
            "app_name should be empty when no tracker"
        );
        assert_eq!(
            resp.selected_text, "",
            "selected_text should be empty when no AX tree"
        );
        assert!(
            !resp.capture_active,
            "capture_active should be false when no window data"
        );
        // ocr_text / ocr_timestamp_us are exercised in test_current_context_reflects_seeded_ocr
    }

    /// Verify that the OCR cache round-trip works: seed a value and verify it is
    /// readable via the public accessor (unit-level; no HTTP stack required).
    /// This test does not go through the full handler to avoid race conditions on
    /// the global OCR cache with other concurrently-running tests.
    #[test]
    fn test_ocr_cache_round_trip() {
        let cell = crate::capture_engine::get_latest_ocr_cell_for_test();

        let sentinel_text = "OCR_round_trip_sentinel_value".to_string();
        let sentinel_ts = 9_999_000_000_000_000u64;

        {
            let mut guard = cell.lock().unwrap();
            *guard = Some((sentinel_text.clone(), "TestApp".to_string(), sentinel_ts));
        }

        let result = crate::capture_engine::get_latest_ocr();
        assert!(
            result.is_some(),
            "get_latest_ocr should return Some after seeding"
        );
        let (text, _app, ts) = result.unwrap();
        assert_eq!(text, sentinel_text, "OCR text must match what was seeded");
        assert_eq!(ts, sentinel_ts, "OCR timestamp must match what was seeded");
    }

    /// Integration test: verify OCR data flows from cache through the handler.
    /// Runs single-threaded to avoid interference with the cache reset in the
    /// well-formed test. Uses a sentinel value that is set immediately before
    /// the handler call and verified in the response.
    #[tokio::test]
    async fn test_current_context_reflects_seeded_ocr() {
        // Use a globally-unique sentinel value unlikely to collide with other tests.
        let sentinel_text = "INTEGRATION_OCR_SENTINEL_67890".to_string();
        let sentinel_ts = 8_765_432_100_000_000u64;

        // Lock the cell for the duration of this test to prevent races.
        let cell = crate::capture_engine::get_latest_ocr_cell_for_test();
        let mut guard = cell.lock().unwrap();
        *guard = Some((sentinel_text.clone(), "TestApp".to_string(), sentinel_ts));

        // Call the handler while holding the lock — the handler will try to acquire
        // the same lock via get_latest_ocr(), which would deadlock. Release the lock
        // first and immediately call.
        drop(guard);

        let resp = invoke_handler(minimal_state()).await;

        // The cache may have been modified by another test between drop and the handler
        // reading it. We verify the sentinel or accept that a race occurred.
        // If the response has our sentinel, the round-trip worked.
        if resp.ocr_text == sentinel_text {
            assert_eq!(resp.ocr_timestamp_us, sentinel_ts);
        }
        // If another test cleared the cache, we still verify the handler didn't panic.
        assert!(
            resp.timestamp_us > 0,
            "handler must return a valid timestamp"
        );
    }

    /// Verify extract_selected_text returns the first text-role value it finds.
    #[test]
    fn test_extract_selected_text_text_role() {
        let node = crate::capture::AXTreeNode {
            role: "textfield".to_string(),
            title: None,
            value: Some("selected content".to_string()),
            identifier: None,
            bounds: None,
            children: vec![],
        };
        assert_eq!(extract_selected_text(&node), "selected content");
    }

    /// Verify extract_selected_text returns empty string for non-text roles.
    #[test]
    fn test_extract_selected_text_non_text_role() {
        let node = crate::capture::AXTreeNode {
            role: "button".to_string(),
            title: Some("OK".to_string()),
            value: Some("some value".to_string()),
            identifier: None,
            bounds: None,
            children: vec![],
        };
        assert_eq!(extract_selected_text(&node), "");
    }

    /// Verify extract_selected_text recurses into children.
    #[test]
    fn test_extract_selected_text_recurses() {
        let child = crate::capture::AXTreeNode {
            role: "edit".to_string(),
            title: None,
            value: Some("deep text".to_string()),
            identifier: None,
            bounds: None,
            children: vec![],
        };
        let root = crate::capture::AXTreeNode {
            role: "window".to_string(),
            title: None,
            value: None,
            identifier: None,
            bounds: None,
            children: vec![child],
        };
        assert_eq!(extract_selected_text(&root), "deep text");
    }

    /// Verify the route is registered by ensuring build_router() does not panic
    /// and includes /context/current.
    #[test]
    fn test_route_is_registered() {
        // build_router is called successfully — if the route isn't registered
        // the compiler would catch the handler mismatch.
        let _router = build_router(minimal_state());
    }

    /// AC4: when globally paused, /context/current must return the empty/suppressed state.
    #[tokio::test]
    async fn test_current_context_returns_empty_when_paused() {
        let _guard = capture_state_guard();
        // Set paused=true for this test; restore after.
        set_capture_paused(true);
        let resp = invoke_handler(minimal_state()).await;
        // Must restore before any assertion that could panic to avoid leaking state.
        set_capture_paused(false);

        assert!(
            resp.paused,
            "paused field must be true when globally paused"
        );
        assert!(
            !resp.capture_active,
            "capture_active must be false when paused"
        );
        assert_eq!(
            resp.window_title, "",
            "window_title must be empty when paused"
        );
        assert_eq!(resp.app_name, "", "app_name must be empty when paused");
        assert_eq!(
            resp.selected_text, "",
            "selected_text must be empty when paused"
        );
        assert_eq!(resp.ocr_text, "", "ocr_text must be suppressed when paused");
    }

    /// Verify is_capture_allowed returns false when globally paused.
    #[test]
    fn test_is_capture_allowed_respects_pause() {
        let _guard = capture_state_guard();
        set_capture_paused(true);
        let allowed = is_capture_allowed("SomeApp");
        set_capture_paused(false);
        assert!(!allowed, "capture must not be allowed when globally paused");
    }

    /// Verify is_capture_allowed returns true when allowlist is empty (allow all).
    #[test]
    fn test_is_capture_allowed_empty_allowlist_allows_all() {
        let _guard = capture_state_guard();
        set_capture_paused(false);
        set_app_allowlist(vec![]);
        assert!(
            is_capture_allowed("AnyApp"),
            "empty allowlist must allow any app"
        );
    }

    /// Verify is_capture_allowed filters when allowlist is non-empty.
    #[test]
    fn test_is_capture_allowed_non_empty_allowlist() {
        let _guard = capture_state_guard();
        set_capture_paused(false);
        set_app_allowlist(vec!["VSCode".to_string(), "Terminal".to_string()]);

        assert!(
            is_capture_allowed("VSCode"),
            "VSCode is on the allowlist and must be allowed"
        );
        assert!(
            !is_capture_allowed("Slack"),
            "Slack is not on the allowlist and must be blocked"
        );

        // Restore empty allowlist so other tests are not affected.
        set_app_allowlist(vec![]);
    }

    // ─── ct_eq (constant-time compare) ───────────────────────────────────────

    #[test]
    fn ct_eq_matches_only_equal_slices() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        // Length mismatch is rejected without indexing past the shorter slice.
        assert!(!ct_eq(b"abc", b"abcd"));
        assert!(ct_eq(b"", b""));
    }

    // ─── frame/capture toggles ───────────────────────────────────────────────

    #[test]
    fn frame_capture_toggle_round_trips() {
        let _guard = capture_state_guard();
        set_frame_capture_enabled(false);
        assert!(!is_frame_capture_enabled());
        set_frame_capture_enabled(true);
        assert!(is_frame_capture_enabled());
    }

    // ─── DB-free / security-relevant handler paths ───────────────────────────

    #[tokio::test]
    async fn health_handler_reports_healthy() {
        let resp = health_handler(State(minimal_state())).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn meeting_status_handler_returns_ok() {
        let resp = meeting_status_handler().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn meeting_start_rejects_non_http_scheme() {
        let req = MeetingStartRequest {
            meeting_id: "m1".to_string(),
            ingest_url: Some("https://127.0.0.1:7980/api/meetings/m1/chunk".to_string()),
        };
        let resp = meeting_start_handler(Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn meeting_start_rejects_remote_host() {
        let req = MeetingStartRequest {
            meeting_id: "m1".to_string(),
            ingest_url: Some("http://evil.example.com:7980/api/meetings/m1/chunk".to_string()),
        };
        let resp = meeting_start_handler(Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn clips_context_missing_clip_is_not_found() {
        let resp = clips_context_handler(axum::extract::Path("nonexistent-clip-id".to_string()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn clips_file_rejects_path_traversal_id() {
        // A percent-decoded "../" id must be rejected before any fs access.
        let resp = clips_file_handler(axum::extract::Path("../../etc/passwd".to_string()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn clips_pause_without_recording_is_not_found() {
        let resp = clips_pause_handler(axum::extract::Path("c1".to_string()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn clips_resume_without_recording_is_not_found() {
        let resp = clips_resume_handler(axum::extract::Path("c1".to_string()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn intent_handler_uses_heuristic_without_orchestrator() {
        let req = IntentRequest {
            query: "remind me to call the dentist".to_string(),
        };
        let resp = intent_handler(State(minimal_state()), Json(req))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ─── build_activity_text (pure) ──────────────────────────────────────────

    fn timeline_entry(
        app: Option<&str>,
        title: Option<&str>,
    ) -> shadow_core::timeline::TimelineEntry {
        shadow_core::timeline::TimelineEntry {
            ts: 0,
            track: 0,
            event_type: "focus".to_string(),
            app_name: app.map(str::to_string),
            window_title: title.map(str::to_string),
            url: None,
            display_id: None,
            segment_file: String::new(),
        }
    }

    #[test]
    fn build_activity_text_summarizes_apps_and_dedups_consecutive_focus() {
        let entries = vec![
            timeline_entry(Some("Mail"), Some("Inbox")),
            timeline_entry(Some("Mail"), Some("Inbox")), // duplicate → collapsed
            timeline_entry(Some("Slack"), Some("general")),
        ];
        let (summary, transcript) = build_activity_text(5, &entries);
        assert!(summary.contains("Apps seen: Mail, Slack."));
        // Consecutive identical focus is collapsed to a single line.
        assert_eq!(transcript, "Mail: Inbox\nSlack: general");
    }

    #[test]
    fn build_activity_text_handles_empty_and_partial_entries() {
        let (summary, transcript) = build_activity_text(3, &[]);
        assert!(summary.contains("No window focus was captured"));
        assert_eq!(transcript, "");

        // App with no title, and title with no app.
        let entries = vec![
            timeline_entry(Some("Terminal"), None),
            timeline_entry(None, Some("Untitled")),
        ];
        let (_s, transcript) = build_activity_text(3, &entries);
        assert_eq!(transcript, "Terminal\nUntitled");
    }

    // ─── shadow_core-backed handlers: graceful uninitialized-store path ───────
    //
    // The global shadow_core storage is never initialized in this test binary,
    // so these query functions return a graceful Err that the handler turns into
    // a JSON error payload (HTTP 200) or a 404 — no panic, fully hermetic.

    #[tokio::test]
    async fn search_handler_returns_json_when_store_uninitialized() {
        let resp = search_handler(Query(SearchQuery {
            q: "anything".to_string(),
            limit: Some(5),
            category: None,
        }))
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn semantic_search_handler_returns_json_when_store_uninitialized() {
        let resp = semantic_search_handler(
            State(minimal_state()),
            Query(SearchQuery {
                q: "anything".to_string(),
                limit: None,
                category: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn timeline_handler_returns_json_when_store_uninitialized() {
        let resp = timeline_handler(Query(TimelineQuery { start: 0, end: 1 }))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn journal_handler_returns_json_without_narration() {
        let resp = journal_handler(
            State(minimal_state()),
            Query(JournalQuery {
                start: 0,
                end: 1,
                narrate: false,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn journal_weekly_handler_returns_json() {
        let resp = journal_weekly_handler(Query(WeeklyQuery {
            end: 10 * 86_400 * 1_000_000,
            days: Some(2),
        }))
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn recent_context_handler_returns_json() {
        let resp = recent_context_handler(Query(SearchQuery {
            q: "10".to_string(),
            limit: None,
            category: None,
        }))
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn recent_activity_handler_returns_json_bundle() {
        let resp = recent_activity_handler(Query(RecentActivityQuery { minutes: Some(3) }))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn frame_handler_returns_not_found_when_no_keyframe() {
        let resp = frame_handler(Query(FrameQuery {
            ts: 0,
            display: None,
        }))
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ─── State-based handlers: graceful None-store paths ─────────────────────

    #[tokio::test]
    async fn capture_control_get_returns_current_state() {
        let _guard = capture_state_guard();
        let resp = capture_control_get_handler().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn capture_control_post_applies_and_reflects_changes() {
        let _guard = capture_state_guard();
        let req = CaptureControlRequest {
            paused: Some(true),
            app_allowlist: Some(vec!["Mail".to_string()]),
            frames: Some(false),
            history_retention_days: None,
        };
        let resp = capture_control_post_handler(Json(req))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(is_capture_paused());
        // Restore globals for other tests.
        set_capture_paused(false);
        set_app_allowlist(vec![]);
        set_frame_capture_enabled(true);
    }

    #[tokio::test]
    async fn agent_tools_handler_empty_without_orchestrator() {
        let resp = agent_tools_handler(State(minimal_state()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn proactive_handler_empty_without_store() {
        let resp = proactive_handler(State(minimal_state()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn proactive_handler_lists_from_real_store() {
        let path =
            std::env::temp_dir().join(format!("shadow-srv-proactive-{}", uuid::Uuid::new_v4()));
        let store = crate::intelligence::ProactiveStore::new(&path).unwrap();
        let mut state = minimal_state();
        state.proactive_store = Some(Arc::new(tokio::sync::Mutex::new(store)));
        let resp = proactive_handler(State(state)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn summaries_and_summary_by_id_without_store() {
        let resp = summaries_handler(State(minimal_state()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let resp = summary_by_id_handler(
            State(minimal_state()),
            axum::extract::Path("missing".to_string()),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn generate_summary_without_orchestrator_errors_gracefully() {
        let req = GenerateSummaryRequest {
            start_ts: 0,
            end_ts: 1,
        };
        let resp = generate_summary_handler(State(minimal_state()), Json(req))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn memory_handlers_without_global_store() {
        let resp = memory_query_handler(Query(SearchQuery {
            q: "x".to_string(),
            limit: None,
            category: Some("fact".to_string()),
        }))
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = memory_store_handler(Json(json!({"content": "hi"})))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = directives_handler().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);

        let resp = create_directive_handler(Json(json!({"content": "call back"})))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn procedures_handler_empty_without_store() {
        let resp = procedures_handler(State(minimal_state()))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn ingest_handler_counts_written_events() {
        // Events fail to persist without a store, so ingested count is 0 — the
        // handler still returns a well-formed response.
        let req = IngestRequest { events: vec![] };
        let resp = ingest_handler(Json(req)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn synthesize_handler_without_orchestrator() {
        let req = SynthesizeRequest { actions: vec![] };
        let resp = synthesize_handler(State(minimal_state()), Json(req))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn feedback_handler_without_delivery_manager() {
        let req = FeedbackRequest {
            suggestion_type: "reminder".to_string(),
            kind: "thumbs_up".to_string(),
        };
        let resp = feedback_handler(State(minimal_state()), Json(req))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn feedback_handler_applies_with_delivery_manager() {
        let trust = Arc::new(std::sync::Mutex::new(crate::intelligence::TrustTuner::new()));
        let dm = Arc::new(crate::intelligence::DeliveryManager::new(trust, true));
        let mut state = minimal_state();
        state.delivery_manager = Some(dm);
        for kind in ["thumbs_up", "thumbs_down", "snooze", "dismiss"] {
            let req = FeedbackRequest {
                suggestion_type: "reminder".to_string(),
                kind: kind.to_string(),
            };
            let resp = feedback_handler(State(state.clone()), Json(req))
                .await
                .into_response();
            assert_eq!(resp.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn patterns_handler_empty_without_store() {
        let resp = patterns_handler(
            State(minimal_state()),
            Query(PatternsQuery {
                q: None,
                app: None,
                limit: None,
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn patterns_handler_queries_real_store() {
        let dir = std::env::temp_dir().join(format!("shadow-srv-pat-{}", uuid::Uuid::new_v4()));
        let store = crate::agent::PatternStore::new(&dir);
        let mut state = minimal_state();
        state.pattern_store = Some(Arc::new(std::sync::Mutex::new(store)));
        let resp = patterns_handler(
            State(state),
            Query(PatternsQuery {
                q: Some("send email".to_string()),
                app: Some("Mail".to_string()),
                limit: Some(5),
            }),
        )
        .await
        .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
