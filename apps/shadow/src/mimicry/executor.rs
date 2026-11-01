use anyhow::Result;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

use super::types::{MimicryProgress, ProcedureStep, StepFailureAction, UndoSnapshot};
use super::undo_manager::ExecutionUndoManager;
use crate::agent::tools::ax::{
    AXClickTool, AXFocusAppTool, AXHotkeyTool, AXScrollTool, AXTreeQueryTool, AXTypeTool,
    AXWaitTool,
};
use crate::agent::tools::{AgentTool, ToolResult};
use crate::intelligence::safety::SafetyGate;

/// A step whose tool reported a failure, with whether the contract error says a
/// retry is safe (`disposition.retry == "safe"`: the input was not delivered).
#[derive(Debug)]
pub(crate) struct StepFailed {
    message: String,
    retry_safe: bool,
}

impl std::fmt::Display for StepFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StepFailed {}

/// Whether a failed tool result's contract error allows a retry. A result without
/// a disposition is treated as unsafe: nothing proves the input was not delivered.
pub(crate) fn retry_is_safe(result: &serde_json::Value) -> bool {
    result["error"]["disposition"]["retry"].as_str() == Some("safe")
}

/// Executes task plan steps using the AX tool suite.
pub struct LocalExecutor {
    tools: Vec<Arc<dyn AgentTool>>,
    safety_gate: Arc<SafetyGate>,
    pub undo: std::sync::Mutex<ExecutionUndoManager>,
}

impl LocalExecutor {
    pub fn new(safety_gate: Arc<SafetyGate>) -> Self {
        Self {
            tools: vec![
                Arc::new(AXClickTool),
                Arc::new(AXTypeTool),
                Arc::new(AXHotkeyTool),
                Arc::new(AXScrollTool),
                Arc::new(AXWaitTool),
                Arc::new(AXFocusAppTool),
                Arc::new(AXTreeQueryTool),
            ],
            safety_gate,
            undo: std::sync::Mutex::new(ExecutionUndoManager::new()),
        }
    }

    /// Execute a single step. Returns Ok(result) or Err on failure.
    pub async fn execute_step(&self, step: &ProcedureStep) -> Result<serde_json::Value> {
        match self
            .safety_gate
            .check(&step.tool_name, &step.tool_args)
            .await
        {
            Err(error) => return Err(error),
            Ok(true) => {
                anyhow::bail!(
                    "Step {} requires explicit user approval and was not executed",
                    step.step_number
                )
            }
            Ok(false) => {}
        }
        self.execute_step_unchecked(step).await
    }

    async fn execute_step_unchecked(&self, step: &ProcedureStep) -> Result<serde_json::Value> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.definition().name == step.tool_name)
            .ok_or_else(|| anyhow::anyhow!("Unknown tool: {}", step.tool_name))?;

        let result: ToolResult = tool.execute(step.tool_args.clone()).await?;

        if let Some(err) = result.error {
            return Err(StepFailed {
                message: format!("Step {} failed: {}", step.step_number, err),
                retry_safe: retry_is_safe(&result.result),
            }
            .into());
        }

        Ok(result.result)
    }

    /// Execute all steps in a plan, streaming progress.
    pub async fn execute_plan(
        &self,
        steps: &[ProcedureStep],
        tx: &UnboundedSender<MimicryProgress>,
    ) -> Result<u32> {
        let mut completed = 0u32;

        for step in steps {
            // Safety gate check before every step
            match self
                .safety_gate
                .check(&step.tool_name, &step.tool_args)
                .await
            {
                Err(e) => {
                    let reason = e.to_string();
                    tracing::warn!(
                        "Safety gate hard-blocked step {}: {}",
                        step.step_number,
                        reason
                    );
                    let _ = tx.send(MimicryProgress::StepBlocked {
                        step: step.step_number,
                        reason: reason.clone(),
                    });
                    return Err(anyhow::anyhow!(
                        "Step {} blocked by safety gate: {}",
                        step.step_number,
                        reason
                    ));
                }
                Ok(true) => {
                    let reason = format!(
                        "Step {} requires explicit user approval and was not executed",
                        step.step_number
                    );
                    let _ = tx.send(MimicryProgress::StepBlocked {
                        step: step.step_number,
                        reason: reason.clone(),
                    });
                    return Err(anyhow::anyhow!(reason));
                }
                Ok(false) => {} // safe
            }

            let _ = tx.send(MimicryProgress::StepStarted {
                step: step.step_number,
                description: step.description.clone(),
            });

            // Push undo snapshot only for steps that mutate state
            const READ_ONLY_TOOLS: &[&str] = &[
                "ax_tree_query",
                "ax_inspect",
                "ax_element_at",
                "ax_read_text",
                "ax_list_apps",
            ];
            if !READ_ONLY_TOOLS.contains(&step.tool_name.as_str()) {
                let pre_hash = self.capture_ax_hash().await.unwrap_or(0);
                // SwitchBack restores the app that was active before a focus change.
                let app_context = if step.tool_name == "ax_focus_app" {
                    active_app_name().await
                } else {
                    None
                };
                if let Ok(mut undo) = self.undo.lock() {
                    let scroll_dy =
                        step.tool_args["direction"]
                            .as_str()
                            .map(|d| if d == "up" { -1i32 } else { 1 });
                    undo.push_step(
                        step.step_number as usize,
                        &step.tool_name,
                        pre_hash,
                        app_context,
                        None,
                        scroll_dy,
                    );
                }
            }

            // Capture AX snapshot before the step when verification is requested
            let before_hash = if step.verification.is_some() {
                self.capture_ax_snapshot(step.step_number)
                    .await
                    .map(|s| hash_value(&s.ax_tree))
            } else {
                None
            };

            match self.execute_step_unchecked(step).await {
                Ok(result) => {
                    // AX hash verification
                    if let Some(condition) = &step.verification {
                        if let Err(e) = self.verify_with_hash(condition, &result, before_hash).await
                        {
                            tracing::warn!("Step {} verification warning: {}", step.step_number, e);
                        }
                    }

                    completed += 1;
                    let _ = tx.send(MimicryProgress::StepCompleted {
                        step: step.step_number,
                        result,
                    });
                }
                Err(e) => {
                    let err_str = e.to_string();
                    let retry_safe = e
                        .downcast_ref::<StepFailed>()
                        .is_some_and(|failed| failed.retry_safe);
                    let _ = tx.send(MimicryProgress::StepFailed {
                        step: step.step_number,
                        error: err_str.clone(),
                    });

                    match step.on_failure {
                        StepFailureAction::Abort => {
                            return Err(anyhow::anyhow!(
                                "Step {} aborted: {}",
                                step.step_number,
                                err_str
                            ));
                        }
                        StepFailureAction::Skip => {
                            tracing::warn!("Step {} skipped: {}", step.step_number, err_str);
                            continue;
                        }
                        StepFailureAction::Retry if !retry_safe => {
                            // The input may already have been delivered (a press sent, some
                            // characters typed): replaying it could submit or type twice.
                            return Err(anyhow::anyhow!(
                                "Step {} failed and was not retried because its input may \
                                 already have been delivered: {}",
                                step.step_number,
                                err_str
                            ));
                        }
                        StepFailureAction::Retry => {
                            // Retry once; the step goes through the safety gate again.
                            match self.execute_step(step).await {
                                Ok(result) => {
                                    completed += 1;
                                    let _ = tx.send(MimicryProgress::StepCompleted {
                                        step: step.step_number,
                                        result,
                                    });
                                }
                                Err(e2) => {
                                    return Err(anyhow::anyhow!(
                                        "Step {} failed after retry: {}",
                                        step.step_number,
                                        e2
                                    ));
                                }
                            }
                        }
                        StepFailureAction::Escalate => {
                            let _ = tx.send(MimicryProgress::Replanning {
                                reason: err_str.clone(),
                            });
                            return Err(anyhow::anyhow!("Escalated: {}", err_str));
                        }
                    }
                }
            }

            // Brief pause between steps
            tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
        }

        Ok(completed)
    }

    /// Capture the current AX tree as an UndoSnapshot.
    async fn capture_ax_snapshot(&self, step: u32) -> Option<UndoSnapshot> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.definition().name == "ax_tree_query")?;
        let result = tool.execute(serde_json::json!({})).await.ok()?;
        if result.error.is_some() {
            return None;
        }
        Some(UndoSnapshot {
            step,
            ax_tree: result.result,
        })
    }

    /// Capture a hash of the current AX tree state.
    async fn capture_ax_hash(&self) -> Option<u64> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.definition().name == "ax_tree_query")?;
        let result = tool.execute(serde_json::json!({})).await.ok()?;
        if result.error.is_some() {
            return None;
        }
        Some(hash_value(&result.result))
    }

    /// Verify a step result, using AX tree hash comparison when verification is set.
    async fn verify_with_hash(
        &self,
        condition: &str,
        result: &serde_json::Value,
        before_hash: Option<u64>,
    ) -> Result<()> {
        // String-contains check
        if condition.starts_with("contains:") {
            let expected = &condition["contains:".len()..];
            let result_str = serde_json::to_string(result).unwrap_or_default();
            if !result_str.contains(expected) {
                anyhow::bail!(
                    "Verification failed: result does not contain '{}'",
                    expected
                );
            }
        }

        // AX tree change check: warn if tree is unchanged after the step
        if let Some(before) = before_hash {
            if let Some(after) = self.capture_ax_hash().await {
                if before == after {
                    tracing::warn!(
                        "AX tree unchanged after step with verification condition '{}'; \
                         the action may not have had its intended effect",
                        condition
                    );
                }
            }
        }

        Ok(())
    }

    /// Undo the last executed step by dispatching the stored reversal strategy.
    /// Failures are logged, never discarded silently.
    pub async fn undo_last_step(&self) {
        let strategy = {
            if let Ok(mut undo) = self.undo.lock() {
                undo.pop_reversal()
            } else {
                None
            }
        };
        let Some(strategy) = strategy else {
            return;
        };
        use super::undo_manager::UndoStrategy;
        let (tool, args) = match strategy {
            UndoStrategy::UndoShortcut => {
                // The platform undo chord: Cmd+Z on macOS, Ctrl+Z elsewhere.
                let keys = if cfg!(target_os = "macos") {
                    "cmd+z"
                } else {
                    "ctrl+z"
                };
                ("ax_hotkey", serde_json::json!({ "keys": keys }))
            }
            UndoStrategy::SwitchBack(app) => {
                ("ax_focus_app", serde_json::json!({ "app_name": app }))
            }
            UndoStrategy::ReverseScroll { dy, .. } => {
                let dir = if dy < 0 { "up" } else { "down" };
                // The scroll point is not recorded; reverse at the display centre.
                let (w, h) = crate::capture::screen::get_primary_display_size();
                let (x, y) = (w / 2, h / 2);
                (
                    "ax_scroll",
                    serde_json::json!({ "direction": dir, "x": x, "y": y }),
                )
            }
            UndoStrategy::Manual(reason) => {
                tracing::warn!("Cannot auto-undo last step: {}", reason);
                return;
            }
        };
        let Some(t) = self.tools.iter().find(|t| t.definition().name == tool) else {
            tracing::warn!("Undo failed: tool {} is not available", tool);
            return;
        };
        match t.execute(args).await {
            Ok(result) => {
                if let Some(err) = result.error {
                    tracing::warn!("Undo via {} failed: {}", tool, err);
                }
            }
            Err(e) => tracing::warn!("Undo via {} failed: {}", tool, e),
        }
    }
}

/// Name of the currently active application, if the window tracker can tell.
async fn active_app_name() -> Option<String> {
    use crate::capture::WindowTracker;
    let tracker = crate::capture::window::PlatformWindowTracker::new().ok()?;
    tracker
        .get_active_app()
        .await
        .map(|app| app.name)
        .filter(|name| !name.is_empty())
}

impl Default for LocalExecutor {
    fn default() -> Self {
        Self::new(Arc::new(SafetyGate::new(None)))
    }
}

/// FNV-1a-inspired hash for quick AX tree change detection.
fn hash_value(v: &serde_json::Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    serde_json::to_string(v)
        .unwrap_or_default()
        .hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
#[path = "executor_test.rs"]
mod tests;
