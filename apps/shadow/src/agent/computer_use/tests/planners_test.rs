use std::sync::{Arc, Mutex};

use axum::body::to_bytes;
use axum::extract::{Path, Request, State};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

use super::super::planners::{
    choice_value, decide_with_jev, decide_with_laya, decide_with_local_model_at,
    parse_choice_response, read_planner_json,
};
use super::super::MAX_PLANNER_RESPONSE_BYTES;
use crate::config::ComputerUseConfig;

#[derive(Debug)]
struct PlannerRequest {
    provider: String,
    authorization: Option<String>,
    payload: Value,
}

async fn planner_test_endpoint(
    Path(provider): Path<String>,
    State(requests): State<Arc<Mutex<Vec<PlannerRequest>>>>,
    request: Request,
) -> Json<Value> {
    let authorization = request
        .headers()
        .get(reqwest::header::AUTHORIZATION)
        .and_then(|header| header.to_str().ok())
        .map(str::to_owned);
    let bytes = to_bytes(request.into_body(), MAX_PLANNER_RESPONSE_BYTES)
        .await
        .unwrap();
    let payload = serde_json::from_slice(&bytes).unwrap();
    let local = provider == "local";
    requests.lock().unwrap().push(PlannerRequest {
        provider,
        authorization,
        payload,
    });
    if local {
        Json(json!({"operation":"DONE","target":null}))
    } else {
        Json(json!({
            "answers": {
                "operation": {
                    "choice": "DONE",
                    "probabilities": { "DONE": 0.99, "BLOCKED": 0.01 }
                }
            }
        }))
    }
}

fn choice(choice: &str, probabilities: Value) -> Value {
    json!({"choice":choice,"probabilities":probabilities})
}

#[test]
fn choice_validation_accepts_only_the_highest_offered_choice() {
    let operations = vec!["CLICK".to_owned(), "DONE".to_owned()];
    let answer = json!({"answers":{"operation":choice("CLICK",json!({"CLICK":0.9,"DONE":0.1})),"click_target":choice("1",json!({"1":0.8,"2":0.2}))}});
    let parsed = parse_choice_response(&answer, &operations).unwrap();
    assert_eq!(parsed.operation, "CLICK");
    assert_eq!(parsed.target.as_deref(), Some("1"));
}

#[test]
fn choice_validation_rejects_unknown_or_malformed_output() {
    let operations = vec!["CLICK".to_owned(), "DONE".to_owned()];
    let unknown = json!({"answers":{"operation":choice("SHELL",json!({"SHELL":1.0}))}});
    assert!(parse_choice_response(&unknown, &operations).is_err());
    let malformed = json!({"choice":"CLICK","probabilities":{"CLICK":0.4,"DONE":0.4}});
    assert!(choice_value(&malformed).is_err());
}

#[test]
fn choice_validation_rejects_a_lower_probability_choice() {
    let answer = choice("CLICK", json!({"CLICK":0.4,"DONE":0.6}));
    assert!(choice_value(&answer).is_err());
}

#[tokio::test]
async fn jev_laya_and_local_adapters_send_only_their_scoped_credentials() {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route("/{provider}", post(planner_test_endpoint))
        .with_state(Arc::clone(&requests));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let mut config = ComputerUseConfig::default();
    config.jev_url = format!("http://{address}/jev");
    config.jev_model = "jev-test-model".to_owned();
    config.laya_url = format!("http://{address}/laya");
    config.validate().unwrap();
    let operations = vec!["DONE".to_owned(), "BLOCKED".to_owned()];
    let questions = json!({"operation":{"choices":operations}});

    let jev = decide_with_jev(
        &config,
        "jev-test-key",
        "finish the task",
        json!({"app":"editor"}),
        questions.clone(),
        &operations,
    )
    .await
    .unwrap();
    let laya = decide_with_laya(
        &config,
        Some("local-node-test-token"),
        "finish the task",
        json!({"app":"editor"}),
        questions,
        &operations,
    )
    .await
    .unwrap();
    config.local_plugin_id = "@ryu/needle3".to_owned();
    config.local_model_id = "needle3".to_owned();
    let local = decide_with_local_model_at(
        &config,
        &format!("http://{address}/local"),
        Some("local-node-test-token"),
        "finish the task",
        json!({"app":"editor"}),
        json!({"operation":{"choices":["DONE","BLOCKED"]}}),
    )
    .await
    .unwrap();

    assert_eq!(jev.operation, "DONE");
    assert_eq!(laya.operation, "DONE");
    assert_eq!(local.operation, "DONE");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].provider, "jev");
    assert_eq!(
        requests[0].authorization.as_deref(),
        Some("Bearer jev-test-key")
    );
    assert_eq!(requests[0].payload["model"], "jev-test-model");
    assert_eq!(requests[0].payload["state"]["task"], "finish the task");
    assert_eq!(requests[1].provider, "laya");
    assert_eq!(
        requests[1].authorization.as_deref(),
        Some("Bearer local-node-test-token")
    );
    assert_eq!(requests[1].payload["model"], "router");
    assert_eq!(requests[1].payload["state"]["task"], "finish the task");
    assert_eq!(requests[2].provider, "local");
    assert_eq!(
        requests[2].authorization.as_deref(),
        Some("Bearer local-node-test-token")
    );
    assert_eq!(requests[2].payload["model_id"], "needle3");
    assert_eq!(requests[2].payload["state"]["task"], "finish the task");

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn planner_response_reader_rejects_oversized_bodies() {
    use axum::routing::get;

    let payload = "x".repeat(MAX_PLANNER_RESPONSE_BYTES + 1);
    let app = axum::Router::new().route(
        "/",
        get(move || {
            let payload = payload.clone();
            async move { payload }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let response = reqwest::Client::new()
        .get(format!("http://{address}/"))
        .send()
        .await
        .unwrap();

    assert!(read_planner_json(response).await.is_err());
    server.abort();
    let _ = server.await;
}
