use super::verify::click_point;
use super::*;
use ghost_eyes::Bounds;
use serde_json::{json, Value};

#[path = "candidates_test.rs"]
mod candidates;
#[path = "input_test.rs"]
mod input;
#[path = "planners_test.rs"]
mod planners;

#[test]
fn computer_use_result_serializes_all_planners_and_terminal_statuses() {
    // This crate is mirrored independently of plugins-store. Keep the actual
    // serde regression here; the plugin's own Node suite checks its schema
    // against these same public wire values without an absent-file include.
    for (planner, planner_wire) in [
        (ComputerUsePlanner::Native, "native"),
        (ComputerUsePlanner::Jev, "jev"),
        (ComputerUsePlanner::Local, "local"),
        (ComputerUsePlanner::Laya, "laya"),
    ] {
        for (status, status_wire) in [
            (ComputerUseStatus::Completed, "completed"),
            (ComputerUseStatus::Blocked, "blocked"),
            (ComputerUseStatus::StepLimit, "step_limit"),
        ] {
            let result = ComputerUseResult {
                planner,
                status,
                steps: vec![ComputerUseStep::new(
                    "CLICK",
                    Some("1".to_owned()),
                    "clicked",
                )],
                message: "Serialized fixture; no desktop action was executed.".to_owned(),
            };
            assert_eq!(
                serde_json::to_value(&result).expect("computer-use result serialization"),
                json!({
                    "planner": planner_wire,
                    "status": status_wire,
                    "steps": [{"operation": "CLICK", "target": "1", "outcome": "clicked"}],
                    "message": "Serialized fixture; no desktop action was executed.",
                }),
                "planner and terminal status must keep their published response shape",
            );
        }
    }
    // Targetless terminal runs remain successful typed responses, not a
    // fabricated action step or an omitted status/message.
    let blocked = ComputerUseResult {
        planner: ComputerUsePlanner::Local,
        status: ComputerUseStatus::Blocked,
        steps: Vec::new(),
        message: "No supported current control.".to_owned(),
    };
    assert_eq!(
        serde_json::to_value(blocked).expect("targetless result serialization"),
        json!({
            "planner": "local", "status": "blocked", "steps": [],
            "message": "No supported current control.",
        }),
    );
}

#[test]
fn status_serialization_keeps_only_the_three_published_values() {
    let wire: Vec<Value> = [
        ComputerUseStatus::Completed,
        ComputerUseStatus::Blocked,
        ComputerUseStatus::StepLimit,
    ]
    .iter()
    .map(|status| serde_json::to_value(status).unwrap())
    .collect();
    assert_eq!(
        wire,
        vec![json!("completed"), json!("blocked"), json!("step_limit")]
    );
}

#[test]
fn a_failed_step_keeps_earlier_steps_and_blocks_the_run() {
    let config = ComputerUseConfig::default();
    let mut done = ComputerUseStep::new("CLICK", Some("Next".to_owned()), "clicked");
    done.method = Some("AXPress".to_owned());
    done.verified = Some(false);
    let error = GhostError::stale_ref("the planned accessibility target disappeared");
    let failure = super::act::failed("TYPE_TEXT", Some("Name".to_owned()), &error);
    let result = super::act::blocked(&config, vec![done], failure);
    let wire = serde_json::to_value(&result).unwrap();
    assert_eq!(wire["status"], "blocked");
    let steps = wire["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["method"], "AXPress");
    assert_eq!(steps[0]["verified"], false);
    assert!(steps[0].get("error").is_none());
    assert_eq!(
        steps[1]["error"],
        json!({"code": "STALE_REF", "disposition": {"delivery": "not_delivered", "retry": "safe"}})
    );
    assert!(steps[1].get("method").is_none());
    assert!(wire["message"].as_str().unwrap().contains("STALE_REF"));
}

#[test]
fn click_point_rejects_out_of_display_targets() {
    let bounds = Bounds {
        x: 1910,
        y: 20,
        width: 80,
        height: 20,
    };
    assert!(click_point(&bounds, 1920, 1080).is_err());
}
