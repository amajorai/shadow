//! The route table, health and stop.

use super::*;

pub(super) fn build_router(state: AppState) -> Router {
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
pub(super) async fn health_handler(
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

pub(super) async fn stop_handler() -> impl IntoResponse {
    // Stop gracefully after returning the response: the server shuts down and
    // `main` closes capture, which seals the open audio and video segments.
    tokio::spawn(async {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        stop_signal().notify_one();
    });
    Json(json!({ "stopping": true }))
}
