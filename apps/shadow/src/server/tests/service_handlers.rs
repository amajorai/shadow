use super::*;

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
    let path = std::env::temp_dir().join(format!("shadow-srv-proactive-{}", uuid::Uuid::new_v4()));
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
