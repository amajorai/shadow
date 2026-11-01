//! The bounded plan-act loop: observe, decide, verify the target, then act.
//!
//! Setup failures (validation, an unavailable accessibility backend, and any
//! failure before the first step) are errors, which the server maps to 503.
//! Once a step has run or been chosen, a failure ends the run as `blocked`: the
//! steps are kept and the failing step carries `error`.

use std::time::Duration;

use anyhow::{bail, Context, Result};
use ghost_eyes::accessibility::resolve::node_at_path;
use ghost_eyes::{ElementLocator, GhostError};

use super::candidates::{build_action_space, candidates};
use super::effector::{Desktop, Effector};
use super::input::{self, StepReport};
use super::observe::{observe, page_changed, window_target};
use super::planners::{decide, generate_field_text};
use super::verify::{click_point, verify_target};
use super::{
    ActionHistory, Candidate, ComputerUseResult, ComputerUseStatus, ComputerUseStep,
    ComputerUseStepError, Decision, WindowState, MAX_GOAL_BYTES,
};
use crate::config::{ComputerUseConfig, ComputerUsePlanner};
use crate::server::AppState;

pub(super) async fn run_inner(
    state: &AppState,
    goal: String,
    config: ComputerUseConfig,
) -> Result<ComputerUseResult> {
    let goal = goal.trim();
    if goal.is_empty() || goal.len() > MAX_GOAL_BYTES {
        bail!("goal must be 1..={MAX_GOAL_BYTES} bytes");
    }
    config.validate()?;
    if config.planner == ComputerUsePlanner::Jev
        && std::env::var("SHADOW_JEV_API_KEY")
            .ok()
            .is_none_or(|key| key.trim().is_empty())
    {
        bail!("Jev is selected, but its API key is not configured in Shadow settings");
    }
    if config.planner == ComputerUsePlanner::Native && state.orchestrator.is_none() {
        bail!("Shadow's text model is not configured");
    }
    let fx = Desktop::new().context("accessibility tree is unavailable")?;
    run_loop(state, &fx, goal, &config).await
}

async fn run_loop<E: Effector>(
    state: &AppState,
    fx: &E,
    goal: &str,
    config: &ComputerUseConfig,
) -> Result<ComputerUseResult> {
    let mut history = Vec::new();
    let mut steps = Vec::new();
    for _ in 0..config.max_steps {
        let observed = match observe(state, fx).await {
            Ok(observed) => observed,
            Err(error) if steps.is_empty() => return Err(error.into()),
            Err(error) => return Ok(blocked(config, steps, failed("OBSERVE", None, &error))),
        };
        let (candidates, cut) = candidates(&observed.tree);
        let truncated = cut || !observed.complete;
        let (operations, questions) =
            build_action_space(&observed, &candidates, truncated, goal, &history);
        let decision = match decide(state, config, goal, &operations, &questions).await {
            Ok(decision) => decision,
            Err(error) if steps.is_empty() => return Err(error),
            Err(error) => {
                let error = GhostError::from_anyhow(&error);
                return Ok(blocked(config, steps, failed("PLAN", None, &error)));
            }
        };

        match decision.operation.as_str() {
            "DONE" => {
                return Ok(finish(
                    config,
                    ComputerUseStatus::Completed,
                    steps,
                    "The planner reports the requested task is complete.",
                ));
            }
            "BLOCKED" => {
                return Ok(finish(
                    config,
                    ComputerUseStatus::Blocked,
                    steps,
                    "The planner could not find a supported next action.",
                ));
            }
            "WAIT" => {
                tokio::time::sleep(Duration::from_millis(650)).await;
                steps.push(ComputerUseStep::new(
                    &decision.operation,
                    None,
                    "waited 650 ms",
                ));
                history.push(ActionHistory {
                    operation: decision.operation,
                    target: None,
                    page_changed: false,
                });
            }
            _ => {
                let label = target_of(&decision, &candidates).map(|c| c.label.clone());
                match execute(state, fx, goal, &decision, &observed, &candidates).await {
                    Ok((step, changed)) => {
                        steps.push(step);
                        history.push(ActionHistory {
                            operation: decision.operation,
                            target: label,
                            page_changed: changed,
                        });
                    }
                    Err(error) => {
                        let step = failed(&decision.operation, label, &error);
                        return Ok(blocked(config, steps, step));
                    }
                }
            }
        }
    }

    let message = format!(
        "Stopped after the configured {}-step limit.",
        config.max_steps
    );
    Ok(finish(
        config,
        ComputerUseStatus::StepLimit,
        steps,
        &message,
    ))
}

fn finish(
    config: &ComputerUseConfig,
    status: ComputerUseStatus,
    steps: Vec<ComputerUseStep>,
    message: &str,
) -> ComputerUseResult {
    ComputerUseResult {
        planner: config.planner,
        status,
        steps,
        message: message.to_owned(),
    }
}

/// A failed step. The message carries only the error code and its (value-free) text.
pub(super) fn failed(
    operation: &str,
    target: Option<String>,
    error: &GhostError,
) -> ComputerUseStep {
    let mut step = ComputerUseStep::new(operation, target, format!("failed: {error}"));
    step.error = Some(ComputerUseStepError::from(error));
    step
}

/// Ends the run as `blocked`, keeping every step; `last` carries the failure.
pub(super) fn blocked(
    config: &ComputerUseConfig,
    mut steps: Vec<ComputerUseStep>,
    last: ComputerUseStep,
) -> ComputerUseResult {
    let code = last
        .error
        .as_ref()
        .map_or("ACTION_FAILED", |error| error.code.as_str());
    let message = format!("Stopped: the {} step failed ({code}).", last.operation);
    steps.push(last);
    finish(config, ComputerUseStatus::Blocked, steps, &message)
}

fn target_of<'a>(decision: &Decision, candidates: &'a [Candidate]) -> Option<&'a Candidate> {
    let id = decision.target.as_deref()?;
    candidates.iter().find(|candidate| candidate.id == id)
}

/// Runs one acting operation. Returns the step and whether the page changed.
async fn execute<E: Effector>(
    state: &AppState,
    fx: &E,
    goal: &str,
    decision: &Decision,
    observed: &WindowState,
    candidates: &[Candidate],
) -> Result<(ComputerUseStep, bool), GhostError> {
    let operation = decision.operation.as_str();
    let (target, report) = match operation {
        "SCROLL_UP" | "SCROLL_DOWN" => {
            let up = operation == "SCROLL_UP";
            (None, input::scroll(fx, observed.tree.bounds.as_ref(), up)?)
        }
        "CLICK" | "TYPE_TEXT" => {
            let candidate = target_of(decision, candidates).ok_or_else(|| {
                GhostError::invalid_args("the planner selected a target outside the action space")
            })?;
            let report = act_on(state, fx, goal, operation, observed, candidate).await?;
            (Some(candidate.label.clone()), report)
        }
        _ => {
            return Err(GhostError::invalid_args(
                "the planner returned an unsupported operation",
            ))
        }
    };
    let settle = if target.is_some() { 350 } else { 250 };
    tokio::time::sleep(Duration::from_millis(settle)).await;
    // The action already ran: a window that cannot be read now counts as changed.
    let changed = observe(state, fx)
        .await
        .map_or(true, |after| page_changed(observed, &after));
    let outcome = if changed && operation != "TYPE_TEXT" {
        format!("{}; page state changed", report.outcome)
    } else {
        report.outcome.to_owned()
    };
    let mut step = ComputerUseStep::new(operation, target, outcome);
    step.method = Some(report.method);
    step.verified = report.verified;
    Ok((step, changed))
}

async fn act_on<E: Effector>(
    state: &AppState,
    fx: &E,
    goal: &str,
    operation: &str,
    observed: &WindowState,
    candidate: &Candidate,
) -> Result<StepReport, GhostError> {
    if operation == "TYPE_TEXT" && !candidate.editable {
        return Err(GhostError::invalid_args(
            "the planner selected a non-editable target for TYPE_TEXT",
        ));
    }
    let mut current = observe(state, fx).await?;
    let mut path = verify_target(observed, &current, candidate)?;
    let mut text = None;
    if operation == "TYPE_TEXT" {
        let orchestrator = state.orchestrator.as_ref().ok_or_else(|| {
            GhostError::internal("Shadow's text model is unavailable for field entry")
        })?;
        let generated = generate_field_text(orchestrator, goal, observed, candidate)
            .await
            .map_err(|_| GhostError::internal("generating the field value failed"))?;
        // Generating text takes time: re-read and re-resolve before typing.
        current = observe(state, fx).await?;
        path = verify_target(observed, &current, candidate)?;
        text = Some(generated);
    }
    let node = node_at_path(&current.tree, &path)
        .ok_or_else(|| GhostError::stale_ref("the planned accessibility target disappeared"))?;
    let (width, height) = fx.display_size();
    let point = node
        .bounds
        .as_ref()
        .and_then(|bounds| click_point(bounds, width, height).ok());
    let locator = ElementLocator {
        pid: current.pid,
        window: window_target(current.pid),
        path,
        identity: candidate.identity.clone(),
    };
    match text {
        None => input::click(fx, &locator, point).await,
        Some(text) => {
            let before = if node.secure {
                None
            } else {
                node.value.clone()
            };
            input::type_into(fx, &locator, &text, before.as_deref(), point).await
        }
    }
}
