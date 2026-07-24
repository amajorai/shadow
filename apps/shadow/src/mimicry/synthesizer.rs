use anyhow::{Context, Result};

use ghost_core::learning::LearnedEvent;

use crate::llm::{orchestrator::LlmOrchestrator, LlmMessage, LlmRequest};
use crate::mimicry::types::{ProcedureStep, ProcedureTemplate, StepFailureAction};
use crate::utils::{extract_json, wall_micros};

/// Converts a recorded `LearnedEvent` sequence into a reusable `ProcedureTemplate`
/// by asking the LLM to generalize the raw actions into parameterized steps.
pub struct ProcedureSynthesizer;

impl ProcedureSynthesizer {
    /// Synthesize a `ProcedureTemplate` from a list of recorded events.
    ///
    /// Requires at least 2 events. Falls back to heuristic synthesis when the
    /// LLM is unavailable or returns unparseable output.
    pub async fn synthesize(
        events: &[LearnedEvent],
        orchestrator: &LlmOrchestrator,
    ) -> Result<ProcedureTemplate> {
        if events.len() < 2 {
            anyhow::bail!("Need at least 2 events to synthesize a procedure");
        }

        let descriptions = events_to_descriptions(events);
        let inferred_app = events
            .iter()
            .filter_map(|e| e.app_name.as_deref())
            .next()
            .unwrap_or("")
            .to_string();

        let prompt = build_synthesis_prompt(&descriptions, &inferred_app);

        let resp = orchestrator
            .generate(LlmRequest {
                messages: vec![LlmMessage::user(prompt)],
                temperature: 0.2,
                max_tokens: 1024,
                ..Default::default()
            })
            .await;

        match resp {
            Ok(r) if r.content.is_some() => {
                let text = r.content.unwrap();
                match parse_template(&text, events, &inferred_app) {
                    Some(t) => Ok(t),
                    None => {
                        tracing::warn!("LLM synthesis parse failed; falling back to heuristic");
                        Ok(heuristic_synthesis(events, &inferred_app))
                    }
                }
            }
            _ => {
                tracing::warn!("LLM unavailable for synthesis; using heuristic");
                Ok(heuristic_synthesis(events, &inferred_app))
            }
        }
    }
}

// ---- helpers ----------------------------------------------------------------

fn events_to_descriptions(events: &[LearnedEvent]) -> Vec<String> {
    events.iter().map(|e| describe_event(e)).collect()
}

fn describe_event(e: &LearnedEvent) -> String {
    let app = e.app_name.as_deref().unwrap_or("app");
    match e.event_type.as_str() {
        "click" => {
            let label = e
                .element_name
                .as_deref()
                .or(e.element_id.as_deref())
                .unwrap_or("element");
            let role = e.element_role.as_deref().unwrap_or("control");
            format!("Clicked '{}' {} in {}", label, role, app)
        }
        "type" => {
            let text = e.key.as_deref().unwrap_or("text");
            format!("Typed '{}' in {}", text, app)
        }
        "hotkey" => {
            let key = e.key.as_deref().unwrap_or("key");
            format!("Pressed hotkey {} in {}", key, app)
        }
        "scroll" => format!("Scrolled in {}", app),
        "app_switch" => {
            format!("Switched to {}", app)
        }
        other => format!("{} in {}", other, app),
    }
}

fn build_synthesis_prompt(descriptions: &[String], app: &str) -> String {
    let steps_text = descriptions
        .iter()
        .enumerate()
        .map(|(i, d)| format!("{}. {}", i + 1, d))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "Convert these recorded actions in {app} into a generalized, reusable procedure.\n\
         Replace specific values (search queries, file names, email addresses) with \
         {{{{PLACEHOLDER}}}} parameters.\n\
         Respond with JSON only (no markdown fences):\n\
         {{\"name\":\"...\",\"description\":\"...\",\"steps\":[\
         {{\"tool_name\":\"ax_click|ax_type|ax_hotkey|ax_scroll|ax_wait|ax_focus_app\",\
         \"description\":\"...\",\"tool_args\":{{...}},\
         \"on_failure\":\"abort|skip|retry|escalate\"}}]}}\n\n\
         Recorded actions:\n{steps_text}",
        app = app,
        steps_text = steps_text,
    )
}

fn parse_template(
    text: &str,
    events: &[LearnedEvent],
    inferred_app: &str,
) -> Option<ProcedureTemplate> {
    let json_str = extract_json(text)?;
    let v: serde_json::Value = serde_json::from_str(&json_str).ok()?;

    let name = v["name"].as_str()?.to_string();
    let description = v["description"].as_str().unwrap_or(&name).to_string();
    let steps_arr = v["steps"].as_array()?;

    let steps: Vec<ProcedureStep> = steps_arr
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let tool_name = s["tool_name"].as_str()?.to_string();
            let desc = s["description"].as_str().unwrap_or("step").to_string();
            let tool_args = s["tool_args"].clone();
            let on_failure = match s["on_failure"].as_str().unwrap_or("abort") {
                "skip" => StepFailureAction::Skip,
                "retry" => StepFailureAction::Retry,
                "escalate" => StepFailureAction::Escalate,
                _ => StepFailureAction::Abort,
            };
            Some(ProcedureStep {
                step_number: (i + 1) as u32,
                description: desc,
                tool_name,
                tool_args,
                verification: None,
                on_failure,
            })
        })
        .collect();

    if steps.is_empty() {
        return None;
    }

    Some(ProcedureTemplate {
        id: uuid::Uuid::new_v4().to_string(),
        name,
        app_name: inferred_app.to_string(),
        description,
        steps,
        preconditions: vec![],
        success_count: 0,
        failure_count: 0,
        last_used: wall_micros(),
        created_at: wall_micros(),
    })
}

/// Fallback: build a template directly from events without LLM.
fn heuristic_synthesis(events: &[LearnedEvent], inferred_app: &str) -> ProcedureTemplate {
    let steps: Vec<ProcedureStep> = events
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let (tool_name, tool_args) = event_to_tool(e);
            ProcedureStep {
                step_number: (i + 1) as u32,
                description: describe_event(e),
                tool_name,
                tool_args,
                verification: None,
                on_failure: StepFailureAction::Skip,
            }
        })
        .collect();

    let name = format!(
        "Recorded procedure in {}",
        events
            .first()
            .and_then(|e| e.app_name.as_deref())
            .unwrap_or("app")
    );

    ProcedureTemplate {
        id: uuid::Uuid::new_v4().to_string(),
        name: name.clone(),
        app_name: inferred_app.to_string(),
        description: name,
        steps,
        preconditions: vec![],
        success_count: 0,
        failure_count: 0,
        last_used: wall_micros(),
        created_at: wall_micros(),
    }
}

fn event_to_tool(e: &LearnedEvent) -> (String, serde_json::Value) {
    match e.event_type.as_str() {
        "click" => {
            let mut args = serde_json::json!({});
            if let Some(name) = &e.element_name {
                args["query"] = serde_json::json!(name);
            } else if let (Some(x), Some(y)) = (e.x, e.y) {
                args["x"] = serde_json::json!(x);
                args["y"] = serde_json::json!(y);
            }
            ("ax_click".to_string(), args)
        }
        "type" => {
            let text = e.key.as_deref().unwrap_or("");
            ("ax_type".to_string(), serde_json::json!({"text": text}))
        }
        "hotkey" => {
            let key = e.key.as_deref().unwrap_or("");
            ("ax_hotkey".to_string(), serde_json::json!({"keys": key}))
        }
        "scroll" => (
            "ax_scroll".to_string(),
            serde_json::json!({"direction": "down"}),
        ),
        "app_switch" => {
            let app = e.app_name.as_deref().unwrap_or("app");
            ("ax_focus_app".to_string(), serde_json::json!({"app": app}))
        }
        _ => ("ax_wait".to_string(), serde_json::json!({"ms": 500})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(event_type: &str) -> LearnedEvent {
        LearnedEvent {
            ts_ms: 0,
            event_type: event_type.to_string(),
            x: None,
            y: None,
            key: None,
            element_role: None,
            element_name: None,
            element_id: None,
            app_name: Some("Mail".to_string()),
        }
    }

    #[test]
    fn describe_event_click_prefers_element_name() {
        let mut e = ev("click");
        e.element_name = Some("Send".to_string());
        e.element_role = Some("button".to_string());
        assert_eq!(describe_event(&e), "Clicked 'Send' button in Mail");
    }

    #[test]
    fn describe_event_click_falls_back_to_element_id_then_defaults() {
        let mut e = ev("click");
        e.element_id = Some("btn-42".to_string());
        assert_eq!(describe_event(&e), "Clicked 'btn-42' control in Mail");

        let bare = ev("click");
        assert_eq!(describe_event(&bare), "Clicked 'element' control in Mail");
    }

    #[test]
    fn describe_event_covers_type_hotkey_scroll_switch_and_other() {
        let mut typed = ev("type");
        typed.key = Some("hello".to_string());
        assert_eq!(describe_event(&typed), "Typed 'hello' in Mail");

        let mut hk = ev("hotkey");
        hk.key = Some("cmd+s".to_string());
        assert_eq!(describe_event(&hk), "Pressed hotkey cmd+s in Mail");

        assert_eq!(describe_event(&ev("scroll")), "Scrolled in Mail");
        assert_eq!(describe_event(&ev("app_switch")), "Switched to Mail");
        assert_eq!(describe_event(&ev("drag")), "drag in Mail");
    }

    #[test]
    fn describe_event_defaults_app_when_missing() {
        let mut e = ev("scroll");
        e.app_name = None;
        assert_eq!(describe_event(&e), "Scrolled in app");
    }

    #[test]
    fn events_to_descriptions_maps_each_event() {
        let events = vec![ev("scroll"), ev("app_switch")];
        let descs = events_to_descriptions(&events);
        assert_eq!(descs, vec!["Scrolled in Mail", "Switched to Mail"]);
    }

    #[test]
    fn build_synthesis_prompt_numbers_steps_and_names_app() {
        let prompt = build_synthesis_prompt(&["Clicked X".to_string(), "Typed Y".to_string()], "Mail");
        assert!(prompt.contains("recorded actions in Mail"));
        assert!(prompt.contains("1. Clicked X"));
        assert!(prompt.contains("2. Typed Y"));
        assert!(prompt.contains("{{PLACEHOLDER}}"));
    }

    #[test]
    fn event_to_tool_click_with_name_uses_query() {
        let mut e = ev("click");
        e.element_name = Some("Compose".to_string());
        let (tool, args) = event_to_tool(&e);
        assert_eq!(tool, "ax_click");
        assert_eq!(args["query"], serde_json::json!("Compose"));
    }

    #[test]
    fn event_to_tool_click_without_name_uses_coordinates() {
        let mut e = ev("click");
        e.x = Some(10);
        e.y = Some(20);
        let (tool, args) = event_to_tool(&e);
        assert_eq!(tool, "ax_click");
        assert_eq!(args["x"], serde_json::json!(10));
        assert_eq!(args["y"], serde_json::json!(20));
    }

    #[test]
    fn event_to_tool_maps_type_hotkey_scroll_switch_and_default() {
        let mut typed = ev("type");
        typed.key = Some("hi".to_string());
        assert_eq!(event_to_tool(&typed), ("ax_type".to_string(), serde_json::json!({"text": "hi"})));

        let mut hk = ev("hotkey");
        hk.key = Some("cmd+c".to_string());
        assert_eq!(event_to_tool(&hk), ("ax_hotkey".to_string(), serde_json::json!({"keys": "cmd+c"})));

        let (scroll_tool, scroll_args) = event_to_tool(&ev("scroll"));
        assert_eq!(scroll_tool, "ax_scroll");
        assert_eq!(scroll_args["direction"], serde_json::json!("down"));

        let (switch_tool, switch_args) = event_to_tool(&ev("app_switch"));
        assert_eq!(switch_tool, "ax_focus_app");
        assert_eq!(switch_args["app"], serde_json::json!("Mail"));

        let (default_tool, _) = event_to_tool(&ev("unknown"));
        assert_eq!(default_tool, "ax_wait");
    }

    #[test]
    fn heuristic_synthesis_builds_a_step_per_event() {
        let events = vec![ev("scroll"), ev("app_switch")];
        let template = heuristic_synthesis(&events, "Mail");
        assert_eq!(template.steps.len(), 2);
        assert_eq!(template.app_name, "Mail");
        assert!(template.name.contains("Mail"));
        assert_eq!(template.steps[0].step_number, 1);
        assert_eq!(template.steps[1].step_number, 2);
        // Heuristic steps use Skip on failure.
        assert!(matches!(template.steps[0].on_failure, StepFailureAction::Skip));
    }

    #[test]
    fn parse_template_extracts_named_steps() {
        let text = "{\"name\":\"Compose email\",\"description\":\"send a mail\",\"steps\":[\
            {\"tool_name\":\"ax_click\",\"description\":\"click compose\",\"tool_args\":{},\"on_failure\":\"retry\"}\
        ]}";
        let events = vec![ev("click"), ev("type")];
        let template = parse_template(text, &events, "Mail").unwrap();
        assert_eq!(template.name, "Compose email");
        assert_eq!(template.description, "send a mail");
        assert_eq!(template.steps.len(), 1);
        assert_eq!(template.steps[0].tool_name, "ax_click");
        assert!(matches!(template.steps[0].on_failure, StepFailureAction::Retry));
    }

    #[test]
    fn parse_template_returns_none_when_no_steps() {
        let text = "{\"name\":\"Empty\",\"steps\":[]}";
        assert!(parse_template(text, &[], "Mail").is_none());
    }

    #[test]
    fn parse_template_returns_none_on_invalid_json() {
        assert!(parse_template("not json", &[], "Mail").is_none());
    }
}
