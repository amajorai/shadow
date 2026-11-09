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

mod agent_routes;
mod api_token;
mod clips;
mod context;
mod history;
mod live;
mod router;

use agent_routes::*;
use api_token::*;
use clips::*;
pub use context::*;
use history::*;
use live::*;
use router::*;

#[cfg(test)]
mod tests;

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
    let listener = bind_listener(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_requested())
        .await?;

    Ok(())
}

/// Bind the HTTP listener with Nagle off on every accepted connection. The SSE
/// streams and the WebSocket feed send small frames; with Nagle on each one waits
/// for the client's delayed ACK (about 40 ms on Linux, up to 200 ms on Windows).
async fn bind_listener(
    addr: SocketAddr,
) -> std::io::Result<impl axum::serve::Listener<Io = tokio::net::TcpStream, Addr = SocketAddr>> {
    use axum::serve::ListenerExt as _;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    Ok(listener.tap_io(|stream: &mut tokio::net::TcpStream| {
        if let Err(error) = stream.set_nodelay(true) {
            tracing::trace!(%error, "could not set TCP_NODELAY on an accepted connection");
        }
    }))
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
