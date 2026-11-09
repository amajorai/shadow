//! Meeting capture and clip handlers.

use super::*;

// ─── Meeting recorder ───────────────────────────────────────────────────────────
//
// Core drives device-local meeting capture through these endpoints: it owns the
// meeting session + transcription + notes, and asks Shadow (the local sensor) to
// stream mixed mic + system-loopback WAV chunks back to
// `POST /api/meetings/:id/chunk`. See `capture::meeting`.

/// POST /meeting/start — begin recording `meeting_id`, uploading chunks to
/// `ingest_url` (defaults to Core on loopback when omitted).
#[derive(Deserialize)]
pub(super) struct MeetingStartRequest {
    pub(super) meeting_id: String,
    /// Core endpoint to POST captured WAV chunks to. When omitted, defaults to
    /// `http://127.0.0.1:7980/api/meetings/<id>/chunk`.
    #[serde(default)]
    pub(super) ingest_url: Option<String>,
}

pub(super) async fn meeting_start_handler(
    Json(req): Json<MeetingStartRequest>,
) -> impl IntoResponse {
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

pub(super) fn resolve_meeting_ingest_url(req: &MeetingStartRequest) -> Result<String, String> {
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
pub(super) async fn meeting_stop_handler() -> impl IntoResponse {
    crate::capture::meeting::stop();
    Json(json!({ "recording": false }))
}

/// GET /meeting/status — whether a meeting is recording, and which one.
pub(super) async fn meeting_status_handler() -> impl IntoResponse {
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
pub(super) async fn clips_start_handler(
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
pub(super) async fn clips_ingest_handler(
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
pub(super) async fn clips_list_handler() -> impl IntoResponse {
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
pub(super) async fn clips_stop_handler(Path(_id): Path<String>) -> impl IntoResponse {
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
pub(super) async fn clips_pause_handler(Path(_id): Path<String>) -> impl IntoResponse {
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
pub(super) async fn clips_resume_handler(Path(_id): Path<String>) -> impl IntoResponse {
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
pub(super) async fn clips_sources_handler() -> impl IntoResponse {
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
pub(super) async fn clips_context_handler(Path(id): Path<String>) -> impl IntoResponse {
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
pub(super) struct ClipFrameQuery {
    #[serde(rename = "atMs", default)]
    pub(super) at_ms: u64,
}

/// GET /clips/{id}/frame?atMs= — a single JPEG frame at the requested moment.
pub(super) async fn clips_frame_handler(
    Path(id): Path<String>,
    Query(q): Query<ClipFrameQuery>,
) -> Response {
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
pub(super) struct DiagnosticsBody {
    #[serde(default)]
    pub(super) events: Vec<crate::capture::clip::DiagnosticEvent>,
}

pub(super) async fn clips_diagnostics_handler(
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
pub(super) async fn clips_file_handler(Path(id): Path<String>) -> Response {
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
