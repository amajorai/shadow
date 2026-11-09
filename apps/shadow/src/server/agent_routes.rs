//! Agent, computer-use, summary, memory, directive and procedure handlers.

use super::*;

// ─── Agent ─────────────────────────────────────────────────────────────────────

pub(super) async fn agent_handler(
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

pub(super) async fn agent_tools_handler(State(state): State<AppState>) -> impl IntoResponse {
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

pub(super) async fn computer_use_config_get(State(state): State<AppState>) -> impl IntoResponse {
    match state.computer_use.read() {
        Ok(config) => Json(crate::agent::computer_use::config_response(&config)).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":{"code":"computer_use_settings_unavailable","message":"computer-use settings are unavailable"}})),
        )
            .into_response(),
    }
}

pub(super) async fn computer_use_config_post(
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

pub(super) async fn computer_use_run_handler(
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

pub(super) async fn proactive_handler(State(state): State<AppState>) -> impl IntoResponse {
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

pub(super) async fn summaries_handler(State(state): State<AppState>) -> impl IntoResponse {
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

pub(super) async fn summary_by_id_handler(
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

pub(super) async fn generate_summary_handler(
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

pub(super) async fn memory_query_handler(Query(query): Query<SearchQuery>) -> impl IntoResponse {
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

pub(super) async fn memory_store_handler(Json(body): Json<serde_json::Value>) -> impl IntoResponse {
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

pub(super) async fn directives_handler() -> impl IntoResponse {
    match crate::memory::MEMORY_STORE.get() {
        Some(store) => match store.lock().unwrap().list_active(None) {
            Ok(directives) => Json(json!({ "directives": directives })),
            Err(e) => Json(json!({ "error": e.to_string() })),
        },
        None => Json(json!({ "directives": [] })),
    }
}

pub(super) async fn create_directive_handler(
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
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

pub(super) async fn procedures_handler(State(state): State<AppState>) -> impl IntoResponse {
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

pub(super) async fn run_procedure_handler(
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
