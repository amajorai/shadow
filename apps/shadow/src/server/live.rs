//! The WebSocket feed, ingest, synthesis, feedback, patterns and intent handlers.

use super::*;

// ─── WebSocket ─────────────────────────────────────────────────────────────────

pub(super) async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_websocket)
}

pub(super) async fn handle_websocket(mut socket: WebSocket) {
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

pub(super) async fn ingest_handler(Json(req): Json<IngestRequest>) -> impl IntoResponse {
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
pub(super) struct SynthesizeRequest {
    pub(super) actions: Vec<serde_json::Value>,
}

pub(super) async fn synthesize_handler(
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
pub(super) struct FeedbackRequest {
    pub(super) suggestion_type: String,
    pub(super) kind: String,
}

pub(super) async fn feedback_handler(
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
pub(super) struct PatternsQuery {
    pub(super) q: Option<String>,
    pub(super) app: Option<String>,
    pub(super) limit: Option<usize>,
}

pub(super) async fn patterns_handler(
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
pub(super) struct IntentRequest {
    pub(super) query: String,
}

pub(super) async fn intent_handler(
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
