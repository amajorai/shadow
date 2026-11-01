//! Accessibility and input tools exposed to Shadow's agent runtime.
//!
//! Errors are reported in `ToolResult.error` as `"CODE: message"` with a single
//! code prefix (the `ghost_eyes` contract codes). Physical input runs only when
//! `ghost_hands::mode::hid_permitted(true)` allows it (a desktop session), the
//! hotkey policy always applies (no `force` is exposed), and typed text is never
//! echoed in a result or an error.

mod click;
mod input;
mod read;
mod system;

use ghost_eyes::{Delivery, Disposition, ErrorCode, GhostError};
use serde_json::{json, Value};

use crate::agent::ToolResult;

pub use click::AXClickTool;
pub use input::{AXFocusAppTool, AXHotkeyTool, AXScrollTool, AXTypeTool};
pub use read::{AXElementAtTool, AXInspectTool, AXReadTextTool, AXTreeQueryTool, AXWaitTool};
pub use system::{AXListAppsTool, CaptureLiveScreenshotTool};

/// Hint string listing available AX/automation tools for the mimicry planner prompt.
pub const AVAILABLE_TOOLS_HINT: &str = "\
ax_click, ax_type, ax_hotkey, ax_scroll, ax_wait, ax_focus_app, \
ax_read_text, ax_inspect, ax_element_at, ax_list_apps, ax_tree_query, \
capture_live_screenshot";

/// A successful tool result.
pub(crate) fn ok(tool: &str, result: Value) -> ToolResult {
    ToolResult {
        tool_name: tool.to_string(),
        result,
        error: None,
    }
}

/// A failed tool result carrying the contract error (`error` is `"CODE: message"`).
pub(crate) fn fail(tool: &str, err: &GhostError) -> ToolResult {
    ToolResult {
        tool_name: tool.to_string(),
        result: json!({ "success": false, "error": err.to_json() }),
        error: Some(err.to_string()),
    }
}

/// Maps an input or accessibility error onto the contract. Hotkey policy and
/// planning errors keep their own code; anything else is `ACTION_FAILED` with an
/// unknown delivery (the input may have partly happened).
///
/// Never call this with a `type_text` error: the Linux backend echoes the
/// characters it could not type. Use [`typing_failed`] instead.
pub(crate) fn input_error(e: &anyhow::Error) -> GhostError {
    if let Some(key) = e.downcast_ref::<ghost_hands::KeyError>() {
        let code = match key.code() {
            "POLICY_DENIED" => ErrorCode::PolicyDenied,
            _ => ErrorCode::InvalidArgs,
        };
        return GhostError::new(code, key.message());
    }
    if e.downcast_ref::<ghost_hands::mode::PlanError>().is_some() {
        return hid_not_allowed();
    }
    if let Some(ghost) = e.downcast_ref::<GhostError>() {
        return ghost.clone();
    }
    GhostError::new(ErrorCode::ActionFailed, e.to_string())
}

/// The generic typing failure: character counts only, never the text.
pub(crate) fn typing_failed(chars: usize) -> GhostError {
    GhostError::new(
        ErrorCode::ActionFailed,
        format!("typing failed ({chars} characters)"),
    )
}

/// The refusal for physical input outside a desktop session.
pub(crate) fn hid_not_allowed() -> GhostError {
    GhostError::policy_denied("physical input is not allowed here")
}

/// `Err` when physical (HID) input is not permitted in this session.
pub(crate) fn require_hid() -> Result<(), GhostError> {
    if ghost_hands::mode::hid_permitted(true) {
        Ok(())
    } else {
        Err(hid_not_allowed())
    }
}

/// Disposition of an HID-only success: delivered, effect unverified.
pub(crate) fn hid_disposition() -> Value {
    serde_json::to_value(Disposition::from(Delivery::DeliveredUnverified)).unwrap_or(Value::Null)
}

/// A required integer parameter that fits in `i32`.
pub(crate) fn required_i32(params: &Value, name: &str) -> Result<i32, GhostError> {
    params[name]
        .as_i64()
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(|| GhostError::invalid_args(format!("`{name}` is required (an integer)")))
}

/// A required non-empty string parameter.
pub(crate) fn required_str<'a>(params: &'a Value, name: &str) -> Result<&'a str, GhostError> {
    params[name]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| {
            GhostError::invalid_args(format!("`{name}` is required (a non-empty string)"))
        })
}

#[cfg(test)]
mod tests;
