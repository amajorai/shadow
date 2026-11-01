//! Bounded computer-use planning for Shadow.
//!
//! Every planner chooses from the current accessibility snapshot. Returned
//! operation and target ids are validated against that snapshot, then the
//! focused window and target are re-read before input is sent.
//!
//! Actions run on the accessibility (semantic) path first. Physical input is a
//! fallback only after the semantic action is unsupported and not delivered,
//! and only on a desktop session (`ghost_hands::mode::hid_permitted`).

mod act;
mod candidates;
mod effector;
mod input;
mod observe;
mod planners;
mod verify;

use std::time::Duration;

use serde::{Deserialize, Serialize};

use ghost_eyes::{AXTreeNode, Disposition, ElementIdentity, ErrorCode, GhostError};

use crate::config::{ComputerUseConfig, ComputerUsePlanner};
use crate::server::AppState;

const MAX_GOAL_BYTES: usize = 4096;
const MAX_CANDIDATES: usize = 40;
const MAX_LABEL_CHARS: usize = 240;
const MAX_FIELD_VALUE_CHARS: usize = 600;
const MAX_TEXT_CHARS: usize = 2000;
const MAX_PLANNER_RESPONSE_BYTES: usize = 64 * 1024;
const MODEL_TIMEOUT: Duration = Duration::from_secs(25);
const RUN_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputerUseRequest {
    pub goal: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComputerUseResult {
    pub planner: ComputerUsePlanner,
    pub status: ComputerUseStatus,
    pub steps: Vec<ComputerUseStep>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerUseStatus {
    Completed,
    Blocked,
    StepLimit,
}

#[derive(Debug)]
pub enum ComputerUseError {
    Busy,
    Failed(anyhow::Error),
}

impl std::fmt::Display for ComputerUseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy => {
                formatter.write_str("another Shadow computer-use task is already running")
            }
            Self::Failed(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for ComputerUseError {}

/// One executed (or failed) step. `method`, `verified` and `error` are additive
/// and omitted when absent, so older clients see the same three keys.
#[derive(Debug, Clone, Serialize)]
pub struct ComputerUseStep {
    pub operation: String,
    pub target: Option<String>,
    pub outcome: String,
    /// The accessibility or input method that ran (e.g. "AXPress", "hid.click").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Whether the effect was read back from the element.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verified: Option<bool>,
    /// Set on the step that ended the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ComputerUseStepError>,
}

impl ComputerUseStep {
    pub(crate) fn new(operation: &str, target: Option<String>, outcome: impl Into<String>) -> Self {
        Self {
            operation: operation.to_owned(),
            target,
            outcome: outcome.into(),
            method: None,
            verified: None,
            error: None,
        }
    }
}

/// The failure of a step. Never carries typed text or a field value.
#[derive(Debug, Clone, Serialize)]
pub struct ComputerUseStepError {
    pub code: ErrorCode,
    pub disposition: Disposition,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

impl From<&GhostError> for ComputerUseStepError {
    fn from(error: &GhostError) -> Self {
        Self {
            code: error.code,
            disposition: error.disposition(),
            suggestion: error.suggestion.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComputerUseConfigResponse {
    pub applied: bool,
    pub planner: ComputerUsePlanner,
    pub jev_url: String,
    pub jev_model: String,
    pub laya_url: String,
    pub local_plugin_id: String,
    pub local_model_id: String,
    pub max_steps: u8,
    pub jev_api_key_configured: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ComputerUseConfigUpdate {
    pub planner: ComputerUsePlanner,
    pub jev_url: String,
    pub jev_model: String,
    pub laya_url: String,
    pub local_plugin_id: String,
    pub local_model_id: String,
    pub max_steps: u8,
}

impl From<ComputerUseConfigUpdate> for ComputerUseConfig {
    fn from(update: ComputerUseConfigUpdate) -> Self {
        Self {
            version: 1,
            planner: update.planner,
            jev_url: update.jev_url,
            jev_model: update.jev_model,
            laya_url: update.laya_url,
            local_plugin_id: update.local_plugin_id,
            local_model_id: update.local_model_id,
            max_steps: update.max_steps,
        }
    }
}

#[derive(Debug, Clone)]
struct WindowState {
    app_name: String,
    title: String,
    /// The window's process; 0 when the tracker could not tell (no pid check).
    pid: i32,
    tree: AXTreeNode,
    /// Whether the walk finished within its limits.
    complete: bool,
}

#[derive(Debug, Clone)]
struct Candidate {
    id: String,
    label: String,
    /// Canonical role key (`AXTreeNode::stable_role`).
    role: String,
    /// Empty for secure fields.
    value: String,
    secure: bool,
    path: Vec<u32>,
    editable: bool,
    identity: ElementIdentity,
}

#[derive(Debug, Clone)]
struct Decision {
    operation: String,
    target: Option<String>,
}

#[derive(Debug, Clone)]
struct ActionHistory {
    operation: String,
    target: Option<String>,
    page_changed: bool,
}

pub async fn run(
    state: &AppState,
    goal: String,
    config: ComputerUseConfig,
) -> std::result::Result<ComputerUseResult, ComputerUseError> {
    let _run_guard = state
        .computer_use_lock
        .try_lock()
        .map_err(|_| ComputerUseError::Busy)?;
    tokio::time::timeout(RUN_TIMEOUT, act::run_inner(state, goal, config))
        .await
        .map_err(|_| ComputerUseError::Failed(anyhow::anyhow!("computer-use task timed out")))?
        .map_err(ComputerUseError::Failed)
}

pub fn config_response(config: &ComputerUseConfig) -> ComputerUseConfigResponse {
    ComputerUseConfigResponse {
        applied: true,
        planner: config.planner,
        jev_url: config.jev_url.clone(),
        jev_model: config.jev_model.clone(),
        laya_url: config.laya_url.clone(),
        local_plugin_id: config.local_plugin_id.clone(),
        local_model_id: config.local_model_id.clone(),
        max_steps: config.max_steps,
        jev_api_key_configured: std::env::var("SHADOW_JEV_API_KEY")
            .ok()
            .is_some_and(|key| !key.trim().is_empty()),
    }
}

#[cfg(test)]
mod tests;
