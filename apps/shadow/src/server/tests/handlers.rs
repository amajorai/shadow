use super::*;

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
async fn the_listener_turns_nagle_off_on_accepted_connections() {
    use axum::serve::Listener as _;
    let mut listener = bind_listener(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let _client = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (accepted, _) = listener.accept().await;
    assert!(accepted.nodelay().unwrap());
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

fn timeline_entry(app: Option<&str>, title: Option<&str>) -> shadow_core::timeline::TimelineEntry {
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
