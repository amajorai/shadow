//! Tests for the AX tool error contract and argument validation. None of these
//! synthesize input: every case is refused before any input would be sent.

use serde_json::json;

use super::click::allows_hid_fallback;
use super::input::{parse_hotkey, parse_scroll};
use super::*;
use crate::agent::AgentTool;

fn code_of(err: &GhostError) -> &'static str {
    err.code.as_str()
}

#[test]
fn blocked_hotkeys_are_refused_with_a_single_code_prefix() {
    let blocked = if cfg!(target_os = "macos") {
        "Cmd+Q"
    } else {
        "Alt+F4"
    };
    let err = parse_hotkey(blocked).expect_err("a system shortcut must be refused");
    assert_eq!(code_of(&err), "POLICY_DENIED");
    let text = err.to_string();
    assert!(text.starts_with("POLICY_DENIED: "), "{text}");
    assert_eq!(text.matches("POLICY_DENIED").count(), 1, "{text}");
}

#[test]
fn unknown_or_empty_hotkeys_are_invalid_args() {
    assert_eq!(code_of(&parse_hotkey("").unwrap_err()), "INVALID_ARGS");
    assert_eq!(code_of(&parse_hotkey(" + ").unwrap_err()), "INVALID_ARGS");
    assert_eq!(
        code_of(&parse_hotkey("Ctrl+NoSuchKey").unwrap_err()),
        "INVALID_ARGS"
    );
    assert_eq!(parse_hotkey("Ctrl+C").unwrap(), vec!["Ctrl", "C"]);
}

#[test]
fn key_errors_map_to_their_own_codes() {
    let key_err =
        ghost_hands::check_hotkey(&["ctrl", "nosuchkey"], ghost_hands::KeyOptions::default())
            .unwrap_err();
    let mapped = input_error(&anyhow::Error::new(key_err));
    assert_eq!(code_of(&mapped), "INVALID_ARGS");
    let other = input_error(&anyhow::anyhow!("SendInput failed"));
    assert_eq!(code_of(&other), "ACTION_FAILED");
    assert_eq!(other.to_string(), "ACTION_FAILED: SendInput failed");
}

#[test]
fn typing_failures_report_counts_only() {
    let err = typing_failed("hunter2".chars().count());
    assert_eq!(
        err.to_string(),
        "ACTION_FAILED: typing failed (7 characters)"
    );
    assert!(!err.to_string().contains("hunter2"));
}

#[test]
fn scroll_requires_coordinates_and_a_known_direction() {
    let missing_xy = parse_scroll(&json!({ "direction": "down" })).unwrap_err();
    assert_eq!(code_of(&missing_xy), "INVALID_ARGS");
    let missing_y = parse_scroll(&json!({ "direction": "down", "x": 5 })).unwrap_err();
    assert_eq!(code_of(&missing_y), "INVALID_ARGS");
    let sideways = json!({ "direction": "sideways", "x": 1, "y": 2 });
    let bad_dir = parse_scroll(&sideways).unwrap_err();
    assert_eq!(code_of(&bad_dir), "INVALID_ARGS");
    let params = json!({ "direction": "up", "x": 1, "y": 2 });
    assert_eq!(parse_scroll(&params).unwrap(), (1, 2, "up", 3));
}

#[tokio::test]
async fn scroll_tool_refuses_missing_coordinates_before_any_input() {
    let result = AXScrollTool
        .execute(json!({ "direction": "down" }))
        .await
        .unwrap();
    let error = result.error.expect("missing x/y must be an error");
    assert!(error.starts_with("INVALID_ARGS: "), "{error}");
    assert_eq!(result.result["success"], json!(false));
    assert_eq!(
        result.result["error"]["disposition"]["retry"],
        json!("safe")
    );
}

#[tokio::test]
async fn wait_and_inspect_require_a_description() {
    let wait = AXWaitTool
        .execute(json!({ "condition": "" }))
        .await
        .unwrap();
    assert!(wait.error.unwrap().starts_with("INVALID_ARGS: "));
    let inspect = AXInspectTool.execute(json!({})).await.unwrap();
    assert!(inspect.error.unwrap().starts_with("INVALID_ARGS: "));
    let element_at = AXElementAtTool.execute(json!({ "x": 3 })).await.unwrap();
    assert!(element_at.error.unwrap().starts_with("INVALID_ARGS: "));
}

#[test]
fn only_undelivered_unsupported_errors_fall_back_to_hid() {
    assert!(allows_hid_fallback(&GhostError::not_supported("no press")));
    assert!(allows_hid_fallback(&GhostError::new(
        ErrorCode::PlatformNotSupported,
        "no backend"
    )));
    assert!(!allows_hid_fallback(
        &GhostError::not_supported("no press").with_delivery(Delivery::DeliveredUnverified)
    ));
    assert!(!allows_hid_fallback(&GhostError::stale_ref("moved")));
    assert!(!allows_hid_fallback(&GhostError::new(
        ErrorCode::ActionFailed,
        "press failed"
    )));
}

#[test]
fn click_no_longer_offers_the_ignored_strategy_param() {
    let definition = AXClickTool.definition();
    assert!(definition.parameters["properties"]
        .get("strategy")
        .is_none());
    assert_eq!(definition.parameters["required"], json!(["description"]));
}

#[test]
fn hid_successes_report_delivered_unverified() {
    assert_eq!(
        hid_disposition(),
        json!({ "delivery": "delivered_unverified", "retry": "unsafe" })
    );
}
