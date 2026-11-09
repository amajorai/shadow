use super::*;

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
            "server::tests::capture_policy::capture_policy_http_child",
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
    let complete =
        json!({"paused":false,"frames":false,"app_allowlist":["Mail"],"history_retention_days":3});
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
                "server::tests::capture_policy::capture_policy_corrupt_startup_child",
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
    let mut value = json!({"version":1,"paused":true,"frames":false,"appAllowlist":["Terminal"]});
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
