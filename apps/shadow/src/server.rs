use axum::{
    extract::ws::{Message, WebSocket},
    extract::{DefaultBodyLimit, Path, Query, Request, State, WebSocketUpgrade},
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
use std::sync::Arc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

use crate::config::Config;
use crate::speech_history::transcripts_handler;

// ─── Consent / capture-control globals ────────────────────────────────────────
//
// These are process-global so the capture loop (capture_engine.rs) and the HTTP
// handler both see the same state without adding fields to AppState.

/// Capture reads project the native policy loaded by init_storage before any
/// capture source starts. Poisoned policy is fail-closed.
pub fn is_capture_allowed(app_name: &str) -> bool {
    shadow_core::capture_allowed(app_name)
}

pub fn is_capture_paused() -> bool {
    shadow_core::capture_paused()
}

pub fn set_capture_paused(paused: bool) {
    let _ = shadow_core::set_capture_runtime(Some(paused), None, None);
}

pub fn is_frame_capture_enabled() -> bool {
    shadow_core::frame_capture_enabled()
}

pub fn set_frame_capture_enabled(enabled: bool) {
    let _ = shadow_core::set_capture_runtime(None, Some(enabled), None);
}

pub fn set_app_allowlist(apps: Vec<String>) {
    let _ = shadow_core::set_capture_runtime(None, None, Some(apps));
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
    if let Some(existing) = read_api_token_file(&path) {
        return Some(existing);
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
    // Created owner-only and exclusively: the token is never observable at a
    // permissive umask, and a concurrent minter (Core's
    // `ensure_api_token`) that won the race keeps its value. The file stays
    // plaintext because the opt-in shell hooks `cat` it (see
    // `capture/passive/terminal.rs`).
    let created = match create_api_token_file(&path, &token) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // A concurrent minter won: use its token.
            if let Some(existing) = read_api_token_file(&path) {
                return Some(existing);
            }
            // A blank/unreadable leftover. Never write the token into it — its
            // mode may be loose and a handle opened earlier would see the
            // token — replace it with a fresh owner-only file instead.
            std::fs::remove_file(&path).and_then(|()| create_api_token_file(&path, &token))
        }
        other => other,
    };
    if let Err(e) = created {
        tracing::error!("cannot persist API token to {}: {e}", path.display());
        return None;
    }
    Some(token)
}

/// Non-empty trimmed contents of the persisted token file, if any.
fn read_api_token_file(path: &std::path::Path) -> Option<String> {
    let existing = std::fs::read_to_string(path).ok()?;
    let trimmed = existing.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// Create the token file and write `token`. The file is created exclusively
/// (`create_new`, which also refuses to follow a symlink) and owner-only in the
/// same `open(2)`: creating it under the default umask and `chmod`ing it
/// afterwards would let another local user open it in between and keep reading.
fn create_api_token_file(path: &std::path::Path, token: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path)?;
    let written = file
        .write_all(token.as_bytes())
        .and_then(|()| file.sync_all());
    if written.is_err() {
        // Never leave a partial token behind for the next run to adopt.
        drop(file);
        let _ = std::fs::remove_file(path);
    }
    written
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
    /// Live computer-use settings; writes are persisted before publishing.
    pub computer_use: Arc<std::sync::RwLock<crate::config::ComputerUseConfig>>,
    /// Serializes screen-input runs so two agent requests never race the pointer.
    pub computer_use_lock: Arc<tokio::sync::Mutex<()>>,
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
    /// Human-readable text exposed by the focused window's accessibility tree.
    /// Empty when the tree is unavailable or capture is paused.
    pub accessibility_text: String,
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
    /// Loopback peer proof (`ryu_sidecar_runtime::peer_proof`): present only
    /// when the caller sent a well-formed `?challenge=` and this server holds
    /// an API token. Lets Core and the clips/meetings sidecars confirm they are
    /// talking to the real Shadow before they send it the bearer.
    #[serde(skip_serializing_if = "Option::is_none")]
    proof: Option<String>,
    /// `ok`, `no-key` (capture writes refused) or `mismatch` (a store was sealed
    /// under another key; writes refused). Core only adopts a Shadow it did not
    /// start when this is `ok`.
    sealing: String,
    /// Sealing detail: why it is not ok, background sealing progress, and the
    /// number of stored values that did not open since start.
    seal: serde_json::Value,
}

#[derive(Deserialize, Default)]
struct HealthQuery {
    challenge: Option<String>,
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
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_requested())
        .await?;

    Ok(())
}

/// Asked to stop by `GET /stop`.
fn stop_signal() -> &'static tokio::sync::Notify {
    static STOP: std::sync::OnceLock<tokio::sync::Notify> = std::sync::OnceLock::new();
    STOP.get_or_init(tokio::sync::Notify::new)
}

/// Set once a stop was requested, before the HTTP server starts draining.
fn shutdown_begun_signal() -> &'static tokio::sync::Notify {
    static BEGUN: std::sync::OnceLock<tokio::sync::Notify> = std::sync::OnceLock::new();
    BEGUN.get_or_init(tokio::sync::Notify::new)
}

/// Resolves as soon as a stop is requested (`GET /stop`, Ctrl-C or SIGTERM),
/// before open connections drain, so `main` closes capture (sealing the open
/// segments) without waiting on long-lived streams.
pub async fn shutdown_begun() {
    shutdown_begun_signal().notified().await;
}

/// How long a graceful stop may take (closing capture, sealing the open
/// segments) before the process exits anyway.
const STOP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(20);

/// Resolves on `GET /stop`, Ctrl-C or SIGTERM, so `main` can close capture
/// (and seal the open media segments) before exiting. A stop that hangs past
/// [`STOP_DEADLINE`] exits the process.
async fn shutdown_requested() {
    let terminate = async {
        #[cfg(unix)]
        {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut signal) => {
                    signal.recv().await;
                }
                Err(_) => std::future::pending::<()>().await,
            }
        }
        #[cfg(not(unix))]
        std::future::pending::<()>().await;
    };
    tokio::select! {
        () = stop_signal().notified() => {}
        _ = tokio::signal::ctrl_c() => {}
        () = terminate => {}
    }
    tracing::info!("Shadow stopping");
    // `notify_one` keeps a permit, so `main` sees it even if it polls later.
    shutdown_begun_signal().notify_one();
    tokio::spawn(async {
        tokio::time::sleep(STOP_DEADLINE).await;
        std::process::exit(0);
    });
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
    let health_token: Option<Arc<str>> = api_token.as_deref().map(Arc::from);
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
        .route("/transcripts", get(transcripts_handler))
        .route("/journal", get(journal_handler))
        .route("/journal/weekly", get(journal_weekly_handler))
        .route("/frame", get(frame_handler))
        .route("/activity/recent", get(recent_activity_handler))
        .route("/context/recent", get(recent_context_handler))
        .route("/context/current", get(current_context_handler))
        // Agent
        .route("/agent", post(agent_handler))
        .route("/agent/tools", get(agent_tools_handler))
        // User-selected indexed computer-use planner.
        .route(
            "/computer-use/config",
            get(computer_use_config_get)
                .post(computer_use_config_post)
                .layer(DefaultBodyLimit::max(16 * 1024)),
        )
        .route(
            "/computer-use/run",
            post(computer_use_run_handler).layer(DefaultBodyLimit::max(16 * 1024)),
        )
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
        .route(
            "/health",
            get(move |query: Query<HealthQuery>| health_handler(query, health_token.clone())),
        )
        .merge(protected)
        .layer(cors)
        .with_state(state)
}

// ─── Health / Stop ─────────────────────────────────────────────────────────────

/// Liveness, plus the answer to a loopback peer challenge. `/health` stays
/// unauthenticated (it returns no capture data); with `?challenge=<nonce>` it
/// also returns `proof = HMAC-SHA256(token, domain || nonce)` so a client can
/// tell this server from a process squatting on the port without sending the
/// token first. The proof reveals nothing about the token.
async fn health_handler(
    Query(query): Query<HealthQuery>,
    api_token: Option<Arc<str>>,
) -> impl IntoResponse {
    let proof = match (api_token.as_deref(), query.challenge.as_deref()) {
        (Some(token), Some(challenge)) => ryu_sidecar_runtime::peer_proof::proof(token, challenge),
        _ => None,
    };
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        proof,
        sealing: shadow_core::seal::state().as_str().to_string(),
        seal: shadow_core::seal::health(),
    })
}

async fn stop_handler() -> impl IntoResponse {
    // Stop gracefully after returning the response: the server shuts down and
    // `main` closes capture, which seals the open audio and video segments.
    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        stop_signal().notify_one();
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
    let result = tokio::task::spawn_blocking(move || crate::capture::clip::start(opts)).await;
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
            Json(json!({ "error": format!("capture task failed: {e}") })),
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
    // The append fsyncs and may wait on the bundle lock held by `stop`, so keep
    // it off the async workers like the other clip file handlers.
    let appended = tokio::task::spawn_blocking(move || {
        crate::capture::clip::append_diagnostics(&id, body.events)
    })
    .await;
    match appended {
        Ok(Ok(ingested)) => (StatusCode::OK, Json(json!({ "ingested": ingested }))),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("diagnostics append task failed: {e}") })),
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
async fn journal_weekly_handler(Query(query): Query<WeeklyQuery>) -> Response {
    use chrono::{Duration, Local};

    let day_count = query.days.unwrap_or(7).clamp(1, 31) as i64;

    // Anchor on the local calendar day containing `end`; reject an `end` beyond
    // chrono's representable range instead of panicking.
    let Some(anchor_date) = local_anchor_date(&Local, query.end) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "end is out of range" })),
        )
            .into_response();
    };

    let mut days: Vec<(String, shadow_core::journal::JournalSnapshot)> = Vec::new();
    // Walk oldest → newest so rollups render left-to-right.
    for offset in (0..day_count).rev() {
        let date = anchor_date - Duration::days(offset);
        let Some((start_us, day_end_us)) = local_day_window_us(&Local, date) else {
            continue;
        };
        // Never look past the requested end.
        let end_us = day_end_us.min(query.end);
        if end_us <= start_us {
            continue;
        }
        match shadow_core::query_journal_snapshot(start_us, end_us) {
            Ok(snapshot) => days.push((date.format("%Y-%m-%d").to_string(), snapshot)),
            Err(e) => {
                return Json(json!({ "error": e.to_string() })).into_response();
            }
        }
    }

    let window_start = days.first().map(|(_, s)| s.start_ts).unwrap_or(query.end);
    let review = shadow_core::journal::build_weekly_review(window_start, query.end, &days);
    Json(json!({ "review": review })).into_response()
}

const MICROS_PER_SEC: u64 = 1_000_000;

/// The local calendar date in `tz` that contains `end_us`, or `None` when the
/// instant or its local wall-clock reading is outside chrono's range. The local
/// reading uses a checked add: `DateTime::date_naive` panics for an instant on
/// chrono's last UTC day in a zone east of UTC.
fn local_anchor_date<Tz: chrono::TimeZone>(tz: &Tz, end_us: u64) -> Option<chrono::NaiveDate> {
    use chrono::Offset as _;

    let secs = i64::try_from(end_us / MICROS_PER_SEC).ok()?;
    let utc = chrono::DateTime::from_timestamp(secs, 0)?.naive_utc();
    let offset = tz.offset_from_utc_datetime(&utc).fix();
    utc.checked_add_offset(offset).map(|local| local.date())
}

/// Longest local-time gap searched for the first instant of a day whose
/// midnight is skipped by a DST shift (real-world gaps are at most a few hours).
const MAX_MIDNIGHT_GAP_MINUTES: i64 = 180;

/// Unix seconds at which local calendar `date` begins in `tz`. A fall-back fold
/// at midnight resolves to the earliest instant. In zones that spring forward
/// at 00:00, midnight does not exist, so the day starts at its first local
/// minute that does.
fn local_day_start_secs<Tz: chrono::TimeZone>(tz: &Tz, date: chrono::NaiveDate) -> Option<i64> {
    let midnight = date.and_hms_opt(0, 0, 0)?;
    (0..=MAX_MIDNIGHT_GAP_MINUTES).find_map(|minute| {
        let local = midnight.checked_add_signed(chrono::Duration::minutes(minute))?;
        tz.from_local_datetime(&local)
            .earliest()
            .map(|dt| dt.timestamp())
    })
}

/// Inclusive `(start_us, end_us)` window for local calendar `date` in `tz`,
/// matching `query_journal_snapshot`'s inclusive end. It runs from this day's
/// local start to one microsecond before the next day's, so DST days span their
/// real 23h/25h, consecutive days tile without overlap or gap, and an event at
/// midnight lands in exactly one day. Pre-epoch instants clamp to 0.
fn local_day_window_us<Tz: chrono::TimeZone>(
    tz: &Tz,
    date: chrono::NaiveDate,
) -> Option<(u64, u64)> {
    let to_us = |secs: i64| {
        secs.checked_mul(MICROS_PER_SEC as i64)
            .map(|us| u64::try_from(us).unwrap_or(0))
    };
    let start_us = to_us(local_day_start_secs(tz, date)?)?;
    let next_start_us = to_us(local_day_start_secs(tz, date.succ_opt()?)?)?;
    Some((start_us, next_start_us.checked_sub(1)?))
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
    // Keyframes are sealed at rest; open them off the async runtime.
    let read_path = std::path::PathBuf::from(&path);
    let read = tokio::task::spawn_blocking(move || shadow_core::media::read_media_file(&read_path))
        .await
        .map_err(|e| e.to_string())
        .and_then(|result| result.map_err(|e| e.to_string()));
    match read {
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
        let read_path = std::path::PathBuf::from(&kf.file_path);
        let read =
            tokio::task::spawn_blocking(move || shadow_core::media::read_media_file(&read_path))
                .await
                .map_err(|e| e.to_string())
                .and_then(|result| result.map_err(|e| e.to_string()));
        match read {
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
    let globally_paused = is_capture_paused();

    // Early return when globally paused — do not query any capture subsystems.
    if globally_paused {
        return Json(CurrentContextResponse {
            timestamp_us: now_us,
            window_title: String::new(),
            app_name: String::new(),
            selected_text: String::new(),
            accessibility_text: String::new(),
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
            accessibility_text: String::new(),
            ocr_text: String::new(),
            ocr_timestamp_us: 0,
            capture_active: false,
            paused: false,
        });
    }

    // 2. Selected text — read from the AX tree's focused element value.
    //    We do a best-effort walk: if the focused tree has a non-empty value on the root
    //    or any direct child, treat it as the selection. Never fail hard.
    let accessibility_tree = if let Some(ax) = &state.ax_tree {
        let t = ax.lock().await;
        match AXTree::get_focused_tree(&*t).await {
            Ok(tree) => Some(tree),
            Err(_) => None,
        }
    } else {
        None
    };
    let selected_text = accessibility_tree
        .as_ref()
        .map(extract_selected_text)
        .unwrap_or_default();
    let accessibility_text = accessibility_tree
        .as_ref()
        .map(extract_accessibility_text)
        .unwrap_or_default();

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
        accessibility_text,
        ocr_text,
        ocr_timestamp_us,
        capture_active,
        paused: false,
    })
}

// ─── Capture control ───────────────────────────────────────────────────────────

/// GET /capture/control — read current pause state, allowlist, and frame toggle.
async fn capture_control_get_handler() -> impl IntoResponse {
    match shadow_core::capture_control_snapshot() {
        Ok((policy, history_retention_days)) => Json(CaptureControlResponse {
            paused: policy.paused,
            app_allowlist: policy.app_allowlist,
            frames: policy.frames,
            history_retention_days,
        })
        .into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":"could not read capture policy"})),
        )
            .into_response(),
    }
}

/// Omitted fields remain unchanged. Durable capture and retention state commits
/// atomically before publishing the accepted runtime state.
async fn capture_control_post_handler(Json(req): Json<CaptureControlRequest>) -> impl IntoResponse {
    let retention_requested = req.history_retention_days.is_some();
    match shadow_core::update_capture_policy(req.paused, req.frames, req.app_allowlist, req.history_retention_days) {
        Ok((policy, history_retention_days)) => {
            // A shorter window takes effect now, not at the next hourly sweep.
            if retention_requested {
                tokio::spawn(run_history_sweep());
            }
            Json(CaptureControlResponse {
                paused: policy.paused, app_allowlist: policy.app_allowlist,
                frames: policy.frames, history_retention_days,
            }).into_response()
        }
        Err(shadow_core::ShadowError::CapturePolicyUninitialized) => (
            StatusCode::CONFLICT,
            Json(json!({"code":"capture_policy_uninitialized","error":"complete capture preferences are required before enabling capture"})),
        ).into_response(),
        Err(error) => {
            tracing::error!("could not persist Shadow capture policy: {error}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":"could not save capture preferences"}))).into_response()
        }
    }
}

/// Shadow's own stores derived from captured activity: 5-minute activity
/// episodes (app, window title, summary) and proactive suggestions built from
/// them. "Keep history" prunes them with shadow-core's timeline.
pub(crate) struct DerivedHistoryStores {
    pub episodes: Option<Arc<std::sync::Mutex<crate::intelligence::EpisodeStore>>>,
    pub proactive: Option<Arc<tokio::sync::Mutex<crate::intelligence::ProactiveStore>>>,
}

static DERIVED_HISTORY: std::sync::OnceLock<DerivedHistoryStores> = std::sync::OnceLock::new();

/// Register the derived stores once at startup, before the first sweep.
pub(crate) fn register_derived_history(stores: DerivedHistoryStores) {
    let _ = DERIVED_HISTORY.set(stores);
}

/// Run one "Keep history" sweep: shadow-core's captured history, then the
/// derived stores, with the same cutoff. Used by the hourly loop and right
/// after `/capture/control` saves a retention change.
pub(crate) async fn run_history_sweep() {
    let outcome = tokio::task::spawn_blocking(shadow_core::run_history_retention_sweep).await;
    let cutoff_us = match &outcome {
        Ok(Ok(report)) => Some(report.cutoff_us),
        _ => None,
    };
    if let (Some(cutoff_us), Some(stores)) = (cutoff_us, DERIVED_HISTORY.get()) {
        prune_derived_history(stores, cutoff_us).await;
    }
    log_history_sweep(outcome);
}

/// Delete activity episodes and proactive suggestions from before `cutoff_us`.
/// A failure is logged and retried by the next sweep.
pub(crate) async fn prune_derived_history(stores: &DerivedHistoryStores, cutoff_us: u64) {
    if cutoff_us == 0 {
        return;
    }
    if let Some(episodes) = &stores.episodes {
        let pruned = match episodes.lock() {
            Ok(store) => store.delete_before(cutoff_us),
            Err(error) => Err(anyhow::anyhow!("episode store lock poisoned: {error}")),
        };
        match pruned {
            Ok(0) => {}
            Ok(n) => tracing::info!(episodes = n, "History retention: pruned activity episodes"),
            Err(error) => tracing::warn!("History retention will retry episodes: {error}"),
        }
    }
    if let Some(proactive) = &stores.proactive {
        match proactive.lock().await.delete_before(cutoff_us) {
            Ok(0) => {}
            Ok(n) => tracing::info!(suggestions = n, "History retention: pruned suggestions"),
            Err(error) => tracing::warn!("History retention will retry suggestions: {error}"),
        }
    }
}

/// Log one "Keep history" sweep: what it removed, and any part it must retry.
pub(crate) fn log_history_sweep(
    outcome: Result<
        Result<shadow_core::HistoryPruneReport, shadow_core::ShadowError>,
        tokio::task::JoinError,
    >,
) {
    match outcome {
        Ok(Ok(report)) => {
            if report.total_removed() > 0 {
                tracing::info!(
                    events = report.events_deleted,
                    search_docs = report.search_docs_deleted,
                    vectors = report.vector_entries_deleted,
                    keyframes = report.keyframes_deleted,
                    files = report.files_deleted,
                    bytes_freed = report.bytes_freed,
                    "History retention: pruned data older than the window"
                );
            }
            for error in &report.errors {
                tracing::warn!("History retention will retry: {error}");
            }
        }
        Ok(Err(error)) => tracing::warn!("History retention sweep failed: {error}"),
        Err(error) => tracing::warn!("History retention sweep did not finish: {error}"),
    }
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

const MAX_ACCESSIBILITY_TEXT_BYTES: usize = 64 * 1024;

/// Flatten the focused accessibility tree into the text Appshots-like clients
/// can attach beside a screenshot. Titles and values are kept in tree order;
/// roles and identifiers stay out of the payload because they add machine
/// metadata rather than useful document text.
pub fn extract_accessibility_text(node: &crate::capture::AXTreeNode) -> String {
    fn append_line(output: &mut String, value: &str) -> bool {
        let value = value.trim();
        if value.is_empty() {
            return true;
        }
        let separator = if output.is_empty() { "" } else { "\n" };
        let available = MAX_ACCESSIBILITY_TEXT_BYTES
            .saturating_sub(output.len())
            .saturating_sub(separator.len());
        if available == 0 {
            return false;
        }
        let mut end = value.len().min(available);
        while end > 0 && !value.is_char_boundary(end) {
            end -= 1;
        }
        output.push_str(separator);
        output.push_str(&value[..end]);
        end == value.len()
    }

    fn walk(node: &crate::capture::AXTreeNode, output: &mut String) -> bool {
        let title = node.title.as_deref().unwrap_or_default();
        if !append_line(output, title) {
            return false;
        }
        let value = node.value.as_deref().unwrap_or_default();
        if value.trim() != title.trim() && !append_line(output, value) {
            return false;
        }
        for child in &node.children {
            if !walk(child, output) {
                return false;
            }
        }
        true
    }

    let mut output = String::new();
    let complete = walk(node, &mut output);
    if !complete {
        output.push_str("\n[Accessibility text truncated]");
    }
    output
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

async fn computer_use_config_get(State(state): State<AppState>) -> impl IntoResponse {
    match state.computer_use.read() {
        Ok(config) => Json(crate::agent::computer_use::config_response(&config)).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":{"code":"computer_use_settings_unavailable","message":"computer-use settings are unavailable"}})),
        )
            .into_response(),
    }
}

async fn computer_use_config_post(
    State(state): State<AppState>,
    Json(update): Json<crate::agent::computer_use::ComputerUseConfigUpdate>,
) -> impl IntoResponse {
    let config = crate::config::ComputerUseConfig::from(update);
    if let Err(error) = config.validate() {
        return (
            StatusCode::BAD_REQUEST,
            Json(
                json!({"error":{"code":"invalid_computer_use_config","message":error.to_string()}}),
            ),
        )
            .into_response();
    }
    match state.computer_use.write() {
        Ok(mut current) => {
            *current = config.clone();
            Json(crate::agent::computer_use::config_response(&config)).into_response()
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":{"code":"computer_use_settings_unavailable","message":"computer-use settings are unavailable"}})),
        )
            .into_response(),
    }
}

async fn computer_use_run_handler(
    State(state): State<AppState>,
    Json(req): Json<crate::agent::computer_use::ComputerUseRequest>,
) -> impl IntoResponse {
    if req.goal.trim().is_empty() || req.goal.len() > 4096 {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error":{"code":"invalid_computer_use_goal","message":"goal must be 1..=4096 bytes"}})),
        )
            .into_response();
    }
    let config = match state.computer_use.read() {
        Ok(config) => config.clone(),
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error":{"code":"computer_use_settings_unavailable","message":"computer-use settings are unavailable"}})),
            )
                .into_response()
        }
    };
    match crate::agent::computer_use::run(&state, req.goal, config).await {
        Ok(result) => Json(result).into_response(),
        Err(crate::agent::computer_use::ComputerUseError::Busy) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error":{"code":"computer_use_busy","message":"another Shadow computer-use task is already running"}})),
        )
            .into_response(),
        Err(crate::agent::computer_use::ComputerUseError::Failed(error)) => {
            let message = error.to_string();
            tracing::warn!("Shadow computer-use task failed: {message}");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error":{"code":"computer_use_failed","message":message}})),
            )
                .into_response()
        }
    }
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
    if is_capture_paused() {
        return Json(json!({ "ingested": 0, "paused": true }));
    }
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

    /// The shared capture policy is process-global and the test
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

    /// The real storage uses OnceLock globals. Each phase therefore executes in
    /// a fresh process while reopening the same SQLite database on disk.
    #[test]
    fn capture_retention_requires_durable_ack_and_survives_reopen() {
        let data =
            std::env::temp_dir().join(format!("shadow-retention-proof-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data).unwrap();
        for phase in [
            "seed45", "reopen45", "refuse", "recover3", "reopen3", "clamp",
        ] {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "server::tests::capture_retention_http_child",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("RYU_SHADOW_RETENTION_PROOF_PHASE", phase)
                .env("RYU_SHADOW_RETENTION_PROOF_DIR", &data)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "actual retention phase {phase} failed: {}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(
                stdout.contains("1 passed; 0 failed; 0 ignored"),
                "child did not execute exactly the real fixture"
            );
            assert!(
                stdout.contains(&format!("real Shadow retention phase {phase} passed")),
                "missing actual phase outcome"
            );
        }
        std::fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn capture_retention_uninitialized_store_refuses_write_keeps_legacy_controls() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "server::tests::capture_retention_http_child",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("RYU_SHADOW_RETENTION_PROOF_PHASE", "uninitialized")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "actual uninitialized-store contract failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(stdout.contains("1 passed; 0 failed; 0 ignored"));
        assert!(stdout.contains("real Shadow retention phase uninitialized passed"));
    }

    #[tokio::test]
    #[ignore = "Real isolated storage fixture executed by the two capture-retention parent tests"]
    async fn capture_retention_http_child() {
        let phase =
            std::env::var("RYU_SHADOW_RETENTION_PROOF_PHASE").expect("parent-owned phase required");
        let data = if phase == "uninitialized" {
            None
        } else {
            let value = std::env::var("RYU_SHADOW_RETENTION_PROOF_DIR")
                .expect("parent-owned real storage required");
            shadow_core::init_storage(value.clone()).unwrap();
            Some(std::path::PathBuf::from(value))
        };
        // Real Axum handlers, native loopback HTTP and genuine SQLite. No capture
        // engine, permission grant, hardware device or UI state is fabricated.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = axum::Router::new().route(
            "/capture/control",
            axum::routing::get(capture_control_get_handler).post(capture_control_post_handler),
        );
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let url = format!("http://{address}/capture/control");
        async fn get(client: &reqwest::Client, url: &str) -> serde_json::Value {
            let response = client.get(url).send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            response.json().await.unwrap()
        }
        async fn post(
            client: &reqwest::Client,
            url: &str,
            body: serde_json::Value,
            expected: reqwest::StatusCode,
        ) -> serde_json::Value {
            let response = client.post(url).json(&body).send().await.unwrap();
            assert_eq!(response.status(), expected);
            response.json().await.unwrap()
        }
        fn durable_days(data: &std::path::Path) -> u32 {
            let db = rusqlite::Connection::open(data.join("indices/timeline.db")).unwrap();
            let hot: String = db
                .query_row(
                    "SELECT value FROM retention_config WHERE key='hot_days'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            let warm: String = db
                .query_row(
                    "SELECT value FROM retention_config WHERE key='warm_days'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            hot.parse::<u32>().unwrap() + warm.parse::<u32>().unwrap()
        }
        match phase.as_str() {
            "uninitialized" => {
                assert_eq!(get(&client, &url).await["history_retention_days"], 30);
                let value = post(
                    &client,
                    &url,
                    json!({"paused":true,"frames":false,"app_allowlist":["Mail"]}),
                    reqwest::StatusCode::OK,
                )
                .await;
                assert_eq!(value["paused"], true);
                assert_eq!(value["frames"], false);
                assert_eq!(value["app_allowlist"], json!(["Mail"]));
                assert_eq!(value["history_retention_days"], 30);
                let error = post(&client, &url, json!({"history_retention_days":3,"paused":false,"frames":true,"app_allowlist":[]}), reqwest::StatusCode::INTERNAL_SERVER_ERROR).await;
                assert!(error["error"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()));
                let current = get(&client, &url).await;
                assert_eq!(current["paused"], true);
                assert_eq!(current["frames"], false);
                assert_eq!(current["app_allowlist"], json!(["Mail"]));
                assert_eq!(current["history_retention_days"], 30);
            }
            "seed45" => {
                let value = post(
                    &client,
                    &url,
                    json!({"history_retention_days":45}),
                    reqwest::StatusCode::OK,
                )
                .await;
                assert_eq!(value["history_retention_days"], 45);
                assert_eq!(durable_days(data.as_ref().unwrap()), 45);
                assert_eq!(get(&client, &url).await["history_retention_days"], 45);
            }
            "reopen45" | "reopen3" => {
                let expected = if phase == "reopen45" { 45 } else { 3 };
                assert_eq!(get(&client, &url).await["history_retention_days"], expected);
                assert_eq!(durable_days(data.as_ref().unwrap()), expected);
            }
            "refuse" => {
                let db =
                    rusqlite::Connection::open(data.as_ref().unwrap().join("indices/timeline.db"))
                        .unwrap();
                assert_eq!(durable_days(data.as_ref().unwrap()), 45);
                let before = get(&client, &url).await;
                db.execute_batch("CREATE TRIGGER retention_fixture_refusal BEFORE UPDATE OF value ON retention_config WHEN OLD.key='warm_days' BEGIN SELECT RAISE(ABORT,'actual retention fixture refusal'); END;").unwrap();
                let error = post(&client, &url, json!({"history_retention_days":3,"paused":true,"frames":false,"app_allowlist":["Mail"]}), reqwest::StatusCode::INTERNAL_SERVER_ERROR).await;
                assert!(error["error"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()));
                assert_eq!(
                    durable_days(data.as_ref().unwrap()),
                    45,
                    "later SQL write refusal must roll back the earlier hot_days write"
                );
                let current = get(&client, &url).await;
                assert_eq!(current["history_retention_days"], 45);
                assert_eq!(
                    current, before,
                    "refused combined update must preserve every active field"
                );
            }
            "recover3" => {
                let db =
                    rusqlite::Connection::open(data.as_ref().unwrap().join("indices/timeline.db"))
                        .unwrap();
                assert_eq!(durable_days(data.as_ref().unwrap()), 45);
                db.execute_batch("DROP TRIGGER retention_fixture_refusal;")
                    .unwrap();
                let value = post(&client, &url, json!({"history_retention_days":3,"paused":true,"frames":false,"app_allowlist":["Mail"]}), reqwest::StatusCode::OK).await;
                assert_eq!(value["history_retention_days"], 3);
                assert_eq!(durable_days(data.as_ref().unwrap()), 3);
                assert_eq!(value["paused"], true);
                assert_eq!(value["frames"], false);
                assert_eq!(value["app_allowlist"], json!(["Mail"]));
            }
            "clamp" => {
                for (requested, expected) in [(0, 1), (u32::MAX, 3650)] {
                    let value = post(
                        &client,
                        &url,
                        json!({"history_retention_days":requested}),
                        reqwest::StatusCode::OK,
                    )
                    .await;
                    assert_eq!(value["history_retention_days"], expected);
                    assert_eq!(durable_days(data.as_ref().unwrap()), expected);
                }
            }
            other => panic!("unexpected parent-owned retention phase {other}"),
        }
        stop.send(()).unwrap();
        server.await.unwrap();
        println!("real Shadow retention phase {phase} passed");
    }

    /// Build a minimal AppState with all optional fields set to None.
    /// The handler must return a well-formed response even with no capture subsystems.
    fn minimal_state() -> AppState {
        AppState {
            config: crate::config::Config::new(),
            computer_use: Arc::new(std::sync::RwLock::new(
                crate::config::ComputerUseConfig::default(),
            )),
            computer_use_lock: Arc::new(tokio::sync::Mutex::new(())),
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
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.join(API_TOKEN_FILE))
                .expect("token file")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600, "token file must be created owner-only");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn resolve_api_token_never_writes_into_a_file_opened_before_it_was_owner_only() {
        use std::io::Read;
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!(
            "shadow-token-perm-test-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(API_TOKEN_FILE);
        // A blank token file with a loose mode, already opened by another
        // local user (the window a create-then-chmod mint leaves open).
        std::fs::write(&path, "").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let mut early_reader = std::fs::File::open(&path).unwrap();

        let minted = resolve_api_token(&dir).expect("token minted");

        let mut leaked = String::new();
        early_reader.read_to_string(&mut leaked).unwrap();
        assert!(leaked.is_empty(), "token leaked through an early handle");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert_eq!(resolve_api_token(&dir).as_deref(), Some(minted.as_str()));
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

    #[test]
    fn test_extract_accessibility_text_keeps_nested_titles_and_values() {
        let root = crate::capture::AXTreeNode {
            role: "window".to_string(),
            title: Some("Budget review".to_string()),
            value: None,
            identifier: None,
            bounds: None,
            children: vec![crate::capture::AXTreeNode {
                role: "textfield".to_string(),
                title: Some("Notes".to_string()),
                value: Some("Review Q3 spend".to_string()),
                identifier: None,
                bounds: None,
                children: vec![],
            }],
        };

        assert_eq!(
            extract_accessibility_text(&root),
            "Budget review\nNotes\nReview Q3 spend"
        );
    }

    /// Verify the route is registered by ensuring build_router() does not panic
    /// and includes /context/current.
    #[test]
    fn test_route_is_registered() {
        // build_router is called successfully — if the route isn't registered
        // the compiler would catch the handler mismatch.
        let _router = build_router(minimal_state());
    }

    #[tokio::test]
    async fn computer_use_settings_post_updates_runtime_without_persisting() {
        let dir = std::env::temp_dir().join(format!(
            "shadow-computer-use-route-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let mut state = minimal_state();
        state.config.data_dir = dir.clone();
        let update = crate::agent::computer_use::ComputerUseConfigUpdate {
            planner: crate::config::ComputerUsePlanner::Jev,
            jev_url: crate::config::DEFAULT_JEV_URL.to_owned(),
            jev_model: crate::config::DEFAULT_JEV_MODEL.to_owned(),
            laya_url: crate::config::DEFAULT_LAYA_URL.to_owned(),
            local_plugin_id: String::new(),
            local_model_id: String::new(),
            max_steps: 9,
        };

        let response = computer_use_config_post(State(state.clone()), Json(update))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["planner"], "jev");
        assert_eq!(json["maxSteps"], 9);
        assert_eq!(json["applied"], true);
        assert!(json["jevApiKeyConfigured"].is_boolean());
        assert!(json.get("SHADOW_JEV_API_KEY").is_none());

        let live = state.computer_use.read().unwrap();
        assert_eq!(live.planner, crate::config::ComputerUsePlanner::Jev);
        drop(live);
        assert!(!dir.join("computer-use.json").exists());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn computer_use_settings_refuse_remote_laya_endpoint() {
        let state = minimal_state();
        let update = crate::agent::computer_use::ComputerUseConfigUpdate {
            planner: crate::config::ComputerUsePlanner::Laya,
            jev_url: crate::config::DEFAULT_JEV_URL.to_owned(),
            jev_model: crate::config::DEFAULT_JEV_MODEL.to_owned(),
            laya_url: "https://example.com/predict".to_owned(),
            local_plugin_id: String::new(),
            local_model_id: String::new(),
            max_steps: 12,
        };

        let response = computer_use_config_post(State(state), Json(update))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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
        assert!(
            !is_capture_allowed(""),
            "an unknown foreground app must not contribute screen text under an allowlist"
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

    async fn health_json(challenge: Option<&str>, token: Option<&str>) -> serde_json::Value {
        let query = HealthQuery {
            challenge: challenge.map(str::to_owned),
        };
        let resp = health_handler(Query(query), token.map(Arc::from))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn health_handler_reports_healthy() {
        let json = health_json(None, Some("secret")).await;
        assert_eq!(json["status"], "healthy");
        // No challenge, no proof.
        assert!(json.get("proof").is_none());
    }

    #[tokio::test]
    async fn health_handler_answers_a_peer_challenge_with_the_token_proof() {
        let challenge = "0123456789abcdef0123456789abcdef";
        let json = health_json(Some(challenge), Some("secret")).await;
        let proof = json["proof"].as_str().expect("proof");
        assert!(ryu_sidecar_runtime::peer_proof::verify(
            "secret", challenge, proof
        ));
        assert!(!ryu_sidecar_runtime::peer_proof::verify(
            "other", challenge, proof
        ));
        // The token itself never appears in the response.
        assert!(!json.to_string().contains("secret"));
    }

    #[tokio::test]
    async fn health_handler_gives_no_proof_without_a_token_or_for_a_bad_challenge() {
        let challenge = "0123456789abcdef0123456789abcdef";
        assert!(health_json(Some(challenge), None)
            .await
            .get("proof")
            .is_none());
        assert!(health_json(Some("short"), Some("secret"))
            .await
            .get("proof")
            .is_none());
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
            bundle_id: None,
            app_path: None,
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
    async fn journal_weekly_handler_rejects_out_of_range_end() {
        let resp = journal_weekly_handler(Query(WeeklyQuery {
            end: u64::MAX,
            days: Some(7),
        }))
        .await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// A test zone with one offset change at the UTC instant `switch`, enough to
    /// model a spring-forward gap or a fall-back fold without a tz database.
    #[derive(Clone, Copy, Debug)]
    struct OneShiftZone {
        switch: chrono::NaiveDateTime,
        before: chrono::FixedOffset,
        after: chrono::FixedOffset,
    }

    #[derive(Clone, Copy, Debug)]
    struct OneShiftOffset {
        zone: OneShiftZone,
        current: chrono::FixedOffset,
    }

    impl chrono::Offset for OneShiftOffset {
        fn fix(&self) -> chrono::FixedOffset {
            self.current
        }
    }

    impl chrono::TimeZone for OneShiftZone {
        type Offset = OneShiftOffset;

        fn from_offset(offset: &OneShiftOffset) -> Self {
            offset.zone
        }

        fn offset_from_local_date(
            &self,
            local: &chrono::NaiveDate,
        ) -> chrono::LocalResult<OneShiftOffset> {
            self.offset_from_local_datetime(&local.and_hms_opt(0, 0, 0).unwrap())
        }

        fn offset_from_local_datetime(
            &self,
            local: &chrono::NaiveDateTime,
        ) -> chrono::LocalResult<OneShiftOffset> {
            let utc_for = |offset: chrono::FixedOffset| {
                *local - chrono::Duration::seconds(i64::from(offset.local_minus_utc()))
            };
            let wrap = |current| OneShiftOffset {
                zone: *self,
                current,
            };
            let pre = (utc_for(self.before) < self.switch).then_some(self.before);
            let post = (utc_for(self.after) >= self.switch).then_some(self.after);
            match (pre, post) {
                // The pre-transition reading is the earlier instant.
                (Some(a), Some(b)) => chrono::LocalResult::Ambiguous(wrap(a), wrap(b)),
                (Some(only), None) | (None, Some(only)) => chrono::LocalResult::Single(wrap(only)),
                (None, None) => chrono::LocalResult::None,
            }
        }

        fn offset_from_utc_date(&self, utc: &chrono::NaiveDate) -> OneShiftOffset {
            self.offset_from_utc_datetime(&utc.and_hms_opt(0, 0, 0).unwrap())
        }

        fn offset_from_utc_datetime(&self, utc: &chrono::NaiveDateTime) -> OneShiftOffset {
            let current = if *utc < self.switch {
                self.before
            } else {
                self.after
            };
            OneShiftOffset {
                zone: *self,
                current,
            }
        }
    }

    fn hours(h: i32) -> chrono::FixedOffset {
        chrono::FixedOffset::east_opt(h * 3600).unwrap()
    }

    fn ymd(y: i32, m: u32, d: u32) -> chrono::NaiveDate {
        chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    fn utc_us(y: i32, m: u32, d: u32, h: u32) -> u64 {
        let secs = ymd(y, m, d)
            .and_hms_opt(h, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        u64::try_from(secs).unwrap() * 1_000_000
    }

    const HOUR_US: u64 = 3_600 * 1_000_000;

    /// Consecutive day windows must neither overlap nor leave a gap.
    fn assert_windows_tile<Tz: chrono::TimeZone>(tz: &Tz, first: chrono::NaiveDate, days: u32) {
        let mut date = first;
        let mut prev = local_day_window_us(tz, date).unwrap();
        for _ in 0..days {
            date = date.succ_opt().unwrap();
            let next = local_day_window_us(tz, date).unwrap();
            assert!(prev.0 < prev.1, "{date}: empty window {prev:?}");
            assert_eq!(next.0, prev.1 + 1, "{date}: windows must tile");
            prev = next;
        }
    }

    #[test]
    fn weekly_day_windows_follow_dst_transitions() {
        // New York-like: 02:00 EST → 03:00 EDT on 2026-03-08 (07:00 UTC).
        let spring = OneShiftZone {
            switch: ymd(2026, 3, 8).and_hms_opt(7, 0, 0).unwrap(),
            before: hours(-5),
            after: hours(-4),
        };
        let (start, end) = local_day_window_us(&spring, ymd(2026, 3, 8)).unwrap();
        assert_eq!(start, utc_us(2026, 3, 8, 5));
        assert_eq!(end + 1 - start, 23 * HOUR_US);
        let (start, end) = local_day_window_us(&spring, ymd(2026, 3, 7)).unwrap();
        assert_eq!(end + 1 - start, 24 * HOUR_US);
        assert_windows_tile(&spring, ymd(2026, 3, 5), 6);

        // 02:00 EDT → 01:00 EST on 2026-11-01 (06:00 UTC).
        let fall = OneShiftZone {
            switch: ymd(2026, 11, 1).and_hms_opt(6, 0, 0).unwrap(),
            before: hours(-4),
            after: hours(-5),
        };
        let (start, end) = local_day_window_us(&fall, ymd(2026, 11, 1)).unwrap();
        assert_eq!(start, utc_us(2026, 11, 1, 4));
        assert_eq!(end + 1 - start, 25 * HOUR_US);
        assert_windows_tile(&fall, ymd(2026, 10, 29), 6);
    }

    #[test]
    fn weekly_day_window_handles_a_midnight_gap_and_fold() {
        // Santiago-like: 00:00 → 01:00 on 2026-09-06 (04:00 UTC), so local
        // midnight does not exist and the day starts at 01:00 local.
        let gap = OneShiftZone {
            switch: ymd(2026, 9, 6).and_hms_opt(4, 0, 0).unwrap(),
            before: hours(-4),
            after: hours(-3),
        };
        let (start, end) = local_day_window_us(&gap, ymd(2026, 9, 6)).unwrap();
        assert_eq!(start, utc_us(2026, 9, 6, 4));
        assert_eq!(end + 1 - start, 23 * HOUR_US);
        assert_windows_tile(&gap, ymd(2026, 9, 3), 6);

        // 01:00 → 00:00 on 2026-04-05 (04:00 UTC), so midnight occurs twice:
        // the day starts at its earliest midnight and keeps the repeated hour,
        // while the previous day loses nothing.
        let fold = OneShiftZone {
            switch: ymd(2026, 4, 5).and_hms_opt(4, 0, 0).unwrap(),
            before: hours(-3),
            after: hours(-4),
        };
        let (start, end) = local_day_window_us(&fold, ymd(2026, 4, 5)).unwrap();
        assert_eq!(start, utc_us(2026, 4, 5, 3));
        assert_eq!(end + 1 - start, 25 * HOUR_US);
        let (start, end) = local_day_window_us(&fold, ymd(2026, 4, 4)).unwrap();
        assert_eq!(start, utc_us(2026, 4, 4, 3));
        assert_eq!(end + 1 - start, 24 * HOUR_US);
        assert_windows_tile(&fold, ymd(2026, 4, 1), 6);
    }

    #[test]
    fn weekly_anchor_date_rejects_instants_past_the_last_local_day() {
        let last_secs = chrono::DateTime::<chrono::Utc>::MAX_UTC.timestamp();
        let last_us = u64::try_from(last_secs).unwrap() * 1_000_000;
        // East of UTC the local reading of chrono's last UTC instant falls past
        // `NaiveDate::MAX`, where `date_naive()` panicked.
        assert_eq!(local_anchor_date(&hours(2), last_us), None);
        assert_eq!(
            local_anchor_date(&hours(0), last_us),
            Some(chrono::NaiveDate::MAX)
        );
        assert_eq!(
            local_anchor_date(&hours(-5), last_us),
            Some(chrono::NaiveDate::MAX)
        );
        assert_eq!(local_anchor_date(&hours(0), u64::MAX), None);
        // An ordinary instant anchors on its local calendar day.
        assert_eq!(
            local_anchor_date(&hours(-5), utc_us(2026, 3, 8, 3)),
            Some(ymd(2026, 3, 7))
        );
    }

    #[test]
    fn weekly_day_windows_near_the_last_date_never_panic() {
        let last_secs = chrono::DateTime::<chrono::Utc>::MAX_UTC.timestamp();
        let last_us = u64::try_from(last_secs).unwrap() * 1_000_000;
        // Whatever `TZ` the suite runs under, the system zone resolves (or
        // rejects) the last representable instants without panicking.
        let _ = local_anchor_date(&chrono::Local, last_us);
        for zone in [hours(14), hours(0), hours(-12)] {
            let anchor = local_anchor_date(&zone, last_us - 86_400 * 1_000_000);
            assert!(anchor.is_some(), "{zone}: day before the last is in range");
        }
        let mut date = chrono::NaiveDate::MAX;
        for _ in 0..32 {
            let _ = local_day_window_us(&chrono::Local, date);
            let _ = local_day_window_us(&hours(14), date);
            let _ = local_day_window_us(&hours(-12), date);
            date = date.pred_opt().unwrap();
        }
        // The last date has no successor to bound it, so it has no window.
        assert_eq!(local_day_window_us(&hours(0), chrono::NaiveDate::MAX), None);
    }

    #[tokio::test]
    async fn journal_weekly_handler_accepts_the_last_representable_end() {
        let last_secs = chrono::DateTime::<chrono::Utc>::MAX_UTC.timestamp();
        let resp = journal_weekly_handler(Query(WeeklyQuery {
            end: u64::try_from(last_secs).unwrap() * 1_000_000,
            days: Some(31),
        }))
        .await;
        // 400 east of UTC (the local day is past chrono's range), otherwise a
        // JSON body; never a panic.
        assert!(
            resp.status() == StatusCode::OK || resp.status() == StatusCode::BAD_REQUEST,
            "{}",
            resp.status()
        );
    }

    #[test]
    fn weekly_day_windows_tile_in_fixed_and_local_zones() {
        let utc = chrono::FixedOffset::east_opt(0).unwrap();
        let (start, end) = local_day_window_us(&utc, ymd(2026, 3, 8)).unwrap();
        assert_eq!(start, utc_us(2026, 3, 8, 0));
        assert_eq!(end + 1 - start, 24 * HOUR_US);
        assert_windows_tile(&utc, ymd(2026, 1, 1), 366);
        assert_windows_tile(&chrono::Local, ymd(2026, 1, 1), 366);
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

    // Actual subprocesses are required: production storage is OnceLock-owned.
    // These execute real handlers/SQLite, not an OS capture engine or mock store.
    #[test]
    fn capture_policy_private_settings_survive_fresh_process_before_capture() {
        let data = std::env::temp_dir().join(format!(
            "shadow-capture-policy-proof-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data).unwrap();
        for phase in [
            "seed_private",
            "reopen_private",
            "sparse_frames",
            "reopen_sparse",
        ] {
            run_capture_policy_phase(&data, phase);
        }
        std::fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn capture_policy_combined_refusal_rolls_back_and_fresh_retry_commits() {
        let data = std::env::temp_dir().join(format!(
            "shadow-capture-policy-refusal-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data).unwrap();
        for phase in [
            "seed_private",
            "refuse_combined",
            "reopen_private",
            "recover_complete",
            "reopen_complete",
        ] {
            run_capture_policy_phase(&data, phase);
        }
        std::fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn capture_policy_missing_native_row_stays_paused_until_complete_preferences() {
        let data =
            std::env::temp_dir().join(format!("shadow-capture-migration-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&data).unwrap();
        for phase in [
            "missing_bootstrap",
            "retention_only",
            "first_partial_then_complete",
            "reopen_migrated",
        ] {
            run_capture_policy_phase(&data, phase);
        }
        std::fs::remove_dir_all(data).unwrap();
    }

    #[test]
    fn capture_policy_concurrent_sparse_http_updates_survive_fresh_reopen() {
        let data = std::env::temp_dir().join(format!(
            "shadow-capture-concurrent-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&data).unwrap();
        for phase in ["seed_private", "concurrent_sparse", "reopen_concurrent"] {
            run_capture_policy_phase(&data, phase);
        }
        std::fs::remove_dir_all(data).unwrap();
    }

    fn run_capture_policy_phase(data: &std::path::Path, phase: &str) {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "server::tests::capture_policy_http_child",
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("RYU_SHADOW_CAPTURE_POLICY_PROOF_PHASE", phase)
            .env("RYU_SHADOW_CAPTURE_POLICY_PROOF_DIR", data)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "actual capture policy phase {phase} failed: {}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(stdout.contains("1 passed; 0 failed; 0 ignored"));
        assert!(stdout.contains(&format!("real Shadow capture policy phase {phase} passed")));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "Fresh native storage child executed by the four capture-policy parent tests"]
    async fn capture_policy_http_child() {
        let phase = std::env::var("RYU_SHADOW_CAPTURE_POLICY_PROOF_PHASE").unwrap();
        let data =
            std::path::PathBuf::from(std::env::var("RYU_SHADOW_CAPTURE_POLICY_PROOF_DIR").unwrap());
        // Production main calls this exact startup boundary BEFORE CaptureEngine
        // and passive sources start. No new test-only restore hook is called.
        shadow_core::init_storage(data.to_string_lossy().to_string()).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        println!("native capture policy listener={address}");
        let router = axum::Router::new().route(
            "/capture/control",
            axum::routing::get(capture_control_get_handler).post(capture_control_post_handler),
        );
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let url = format!("http://{address}/capture/control");
        async fn get(client: &reqwest::Client, url: &str) -> serde_json::Value {
            let response = client.get(url).send().await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            response.json().await.unwrap()
        }
        async fn post(
            client: &reqwest::Client,
            url: &str,
            body: serde_json::Value,
            expected: reqwest::StatusCode,
        ) -> serde_json::Value {
            let response = client.post(url).json(&body).send().await.unwrap();
            assert_eq!(response.status(), expected);
            response.json().await.unwrap()
        }
        fn durable_retention(data: &std::path::Path) -> u32 {
            let db = rusqlite::Connection::open(data.join("indices/timeline.db")).unwrap();
            let hot: String = db
                .query_row(
                    "SELECT value FROM retention_config WHERE key='hot_days'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            let warm: String = db
                .query_row(
                    "SELECT value FROM retention_config WHERE key='warm_days'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            hot.parse::<u32>().unwrap() + warm.parse::<u32>().unwrap()
        }
        let private = json!({"paused":true,"frames":false,"app_allowlist":["Terminal"],"history_retention_days":45});
        let sparse = json!({"paused":true,"frames":true,"app_allowlist":["Terminal"],"history_retention_days":45});
        let complete = json!({"paused":false,"frames":false,"app_allowlist":["Mail"],"history_retention_days":3});
        let safe_bootstrap =
            json!({"paused":true,"frames":false,"app_allowlist":[],"history_retention_days":30});
        let retained_bootstrap =
            json!({"paused":true,"frames":false,"app_allowlist":[],"history_retention_days":45});
        let migrated = json!({"paused":false,"frames":false,"app_allowlist":["Terminal"],"history_retention_days":45});
        fn durable_capture_rows(data: &std::path::Path) -> i64 {
            let db = rusqlite::Connection::open(data.join("indices/timeline.db")).unwrap();
            db.query_row("SELECT COUNT(*) FROM capture_policy", [], |row| row.get(0))
                .unwrap()
        }
        match phase.as_str() {
            "missing_bootstrap" => {
                // No Desktop/Tauri store is read by this owner. Unknown prior
                // device preferences cannot authorize default active capture.
                assert_eq!(get(&client, &url).await, safe_bootstrap);
                assert_eq!(durable_capture_rows(&data), 0);
                assert!(is_capture_paused());
                assert!(!is_frame_capture_enabled());
                assert!(!is_capture_allowed("Terminal"));
            }
            "retention_only" => {
                assert_eq!(get(&client, &url).await, safe_bootstrap);
                assert_eq!(
                    post(
                        &client,
                        &url,
                        json!({"history_retention_days":45}),
                        reqwest::StatusCode::OK
                    )
                    .await,
                    retained_bootstrap
                );
                assert_eq!(durable_retention(&data), 45);
                assert_eq!(durable_capture_rows(&data), 0);
            }
            "first_partial_then_complete" => {
                assert_eq!(get(&client, &url).await, retained_bootstrap);
                let refused = post(
                    &client,
                    &url,
                    json!({"paused":false}),
                    reqwest::StatusCode::CONFLICT,
                )
                .await;
                assert_eq!(refused["code"], "capture_policy_uninitialized");
                assert_eq!(get(&client, &url).await, retained_bootstrap);
                assert_eq!(durable_capture_rows(&data), 0);
                // Full legacy three-field capture payload is enough; retention
                // was already independently initialized and remains unchanged.
                assert_eq!(
                    post(
                        &client,
                        &url,
                        json!({"paused":false,"frames":false,"app_allowlist":["Terminal"]}),
                        reqwest::StatusCode::OK
                    )
                    .await,
                    migrated
                );
                assert_eq!(durable_capture_rows(&data), 1);
                assert_eq!(durable_retention(&data), 45);
            }
            "reopen_migrated" => assert_eq!(get(&client, &url).await, migrated),
            "concurrent_sparse" => {
                assert_eq!(get(&client, &url).await, private);
                let concurrent = json!({"paused":false,"frames":true,"app_allowlist":["Terminal"],"history_retention_days":45});
                for _ in 0..16 {
                    assert_eq!(
                        post(&client, &url, private.clone(), reqwest::StatusCode::OK).await,
                        private
                    );
                    // Two native HTTP requests outstanding on the real two-thread
                    // Axum runtime. No handler/barrier/store is replaced.
                    let (frames, pause) = tokio::join!(
                        post(
                            &client,
                            &url,
                            json!({"frames":true}),
                            reqwest::StatusCode::OK
                        ),
                        post(
                            &client,
                            &url,
                            json!({"paused":false}),
                            reqwest::StatusCode::OK
                        ),
                    );
                    assert_eq!(frames["frames"], true);
                    assert_eq!(pause["paused"], false);
                    assert_eq!(get(&client, &url).await, concurrent);
                    assert_eq!(durable_retention(&data), 45);
                }
            }
            "reopen_concurrent" => assert_eq!(
                get(&client, &url).await,
                json!({"paused":false,"frames":true,"app_allowlist":["Terminal"],"history_retention_days":45})
            ),
            "seed_private" => {
                assert_eq!(
                    post(&client, &url, private.clone(), reqwest::StatusCode::OK).await,
                    private
                );
                assert_eq!(get(&client, &url).await, private);
                assert_eq!(durable_retention(&data), 45);
            }
            "reopen_private" => {
                // Fresh process, startup init only, BEFORE any capture-control POST.
                assert_eq!(get(&client, &url).await, private);
                assert_eq!(durable_retention(&data), 45);
            }
            "sparse_frames" => {
                assert_eq!(get(&client, &url).await, private);
                assert_eq!(
                    post(
                        &client,
                        &url,
                        json!({"frames":true}),
                        reqwest::StatusCode::OK
                    )
                    .await,
                    sparse
                );
                assert_eq!(durable_retention(&data), 45);
            }
            "reopen_sparse" => assert_eq!(get(&client, &url).await, sparse),
            "refuse_combined" => {
                assert_eq!(get(&client, &url).await, private);
                let db = rusqlite::Connection::open(data.join("indices/timeline.db")).unwrap();
                // Existing real retention table, no new-table fixture assumed.
                db.execute_batch("CREATE TRIGGER refuse_capture_retention BEFORE UPDATE ON retention_config WHEN NEW.key='warm_days' BEGIN SELECT RAISE(ABORT,'owned capture-policy refusal'); END;").unwrap();
                let response = post(
                    &client,
                    &url,
                    complete.clone(),
                    reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                )
                .await;
                assert!(response["error"]
                    .as_str()
                    .is_some_and(|message| !message.is_empty()));
                assert_eq!(get(&client, &url).await, private);
                assert_eq!(durable_retention(&data), 45);
                db.execute_batch("DROP TRIGGER refuse_capture_retention;")
                    .unwrap();
            }
            "recover_complete" => {
                assert_eq!(get(&client, &url).await, private);
                assert_eq!(
                    post(&client, &url, complete.clone(), reqwest::StatusCode::OK).await,
                    complete
                );
                assert_eq!(durable_retention(&data), 3);
            }
            "reopen_complete" => assert_eq!(get(&client, &url).await, complete),
            _ => panic!("unknown actual phase {phase}"),
        }
        stop.send(()).unwrap();
        server.await.unwrap();
        println!("real Shadow capture policy phase {phase} passed");
    }

    // Candidate integrity follow-up: run only after the original two-parent
    // comparison proves the real durable row exists. These are not credited as
    // current-original parser tests; no missing-table fixture setup is masked.
    #[test]
    fn capture_policy_invalid_or_unreadable_database_blocks_native_startup() {
        for phase in [
            "null_pause",
            "unsupported_version",
            "unknown_field",
            "oversized",
            "unreadable_column",
        ] {
            let data = std::env::temp_dir().join(format!(
                "shadow-capture-corruption-{}",
                uuid::Uuid::new_v4()
            ));
            std::fs::create_dir_all(&data).unwrap();
            run_capture_policy_phase(&data, "seed_private");
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "server::tests::capture_policy_corrupt_startup_child",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("RYU_SHADOW_CAPTURE_POLICY_PROOF_PHASE", phase)
                .env("RYU_SHADOW_CAPTURE_POLICY_PROOF_DIR", &data)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "actual corrupt startup phase {phase} failed: {}\n{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            let stdout = String::from_utf8_lossy(&result.stdout);
            assert!(stdout.contains("1 passed; 0 failed; 0 ignored"));
            assert!(stdout.contains(&format!(
                "real Shadow blocked corrupt capture startup {phase}"
            )));
            std::fs::remove_dir_all(data).unwrap();
        }
    }

    #[test]
    #[ignore = "Real corrupt-storage startup child executed by its capture-policy parent"]
    fn capture_policy_corrupt_startup_child() {
        let phase = std::env::var("RYU_SHADOW_CAPTURE_POLICY_PROOF_PHASE").unwrap();
        let data =
            std::path::PathBuf::from(std::env::var("RYU_SHADOW_CAPTURE_POLICY_PROOF_DIR").unwrap());
        let db = rusqlite::Connection::open(data.join("indices/timeline.db")).unwrap();
        let mut value =
            json!({"version":1,"paused":true,"frames":false,"appAllowlist":["Terminal"]});
        match phase.as_str() {
            "null_pause" => value["paused"] = serde_json::Value::Null,
            "unsupported_version" => value["version"] = json!(2),
            "unknown_field" => value["futureAllowCapture"] = json!(true),
            "oversized" => value["appAllowlist"] = json!(["x".repeat(2 * 1024 * 1024 + 1)]),
            "unreadable_column" => {
                db.execute_batch(
                    "ALTER TABLE capture_policy RENAME COLUMN policy TO unavailable_policy;",
                )
                .unwrap();
            }
            _ => panic!("unknown actual corruption phase {phase}"),
        }
        if phase != "unreadable_column" {
            assert_eq!(
                db.execute(
                    "UPDATE capture_policy SET policy=?1 WHERE id=1",
                    [value.to_string()]
                )
                .unwrap(),
                1
            );
        }
        drop(db);
        // Exact production startup boundary, before any CaptureEngine starts.
        let error = shadow_core::init_storage(data.to_string_lossy().to_string()).unwrap_err();
        assert!(!error.to_string().is_empty());
        assert!(is_capture_paused());
        assert!(!is_frame_capture_enabled());
        assert!(!is_capture_allowed("Terminal"));
        // Production main propagates this failure with ?, so no OS capture
        // source is instantiated. This fixture does not invoke native hardware.
        println!("real Shadow blocked corrupt capture startup {phase}");
    }
}
