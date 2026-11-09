use super::*;

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
                "server::tests::retention::capture_retention_http_child",
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
            "server::tests::retention::capture_retention_http_child",
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
            let error = post(
                &client,
                &url,
                json!({"history_retention_days":3,"paused":false,"frames":true,"app_allowlist":[]}),
                reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            )
            .await;
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
            let db = rusqlite::Connection::open(data.as_ref().unwrap().join("indices/timeline.db"))
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
            let db = rusqlite::Connection::open(data.as_ref().unwrap().join("indices/timeline.db"))
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
