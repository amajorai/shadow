//! Input tools: type, hotkey, scroll and app focus.

use anyhow::Result;
use async_trait::async_trait;
use ghost_eyes::{ErrorCode, GhostError};
use serde_json::json;

use super::{
    fail, hid_disposition, input_error, ok, require_hid, required_i32, required_str, typing_failed,
};
use crate::agent::{AgentTool, Tool, ToolResult};

/// Type text into focused element.
pub struct AXTypeTool;

#[async_trait]
impl AgentTool for AXTypeTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_type".to_string(),
            description: "Type text into the currently focused UI element".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "press_enter": {"type": "boolean", "default": false}
                },
                "required": ["text"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        const NAME: &str = "ax_type";
        let Some(text) = params["text"].as_str() else {
            return Ok(fail(NAME, &GhostError::invalid_args("`text` is required")));
        };
        let press_enter = params["press_enter"].as_bool().unwrap_or(false);
        if let Err(e) = require_hid() {
            return Ok(fail(NAME, &e));
        }
        let typed_chars = text.chars().count();
        // The error is replaced: some backends echo the characters they could not type.
        if ghost_hands::type_text(text, false).is_err() {
            return Ok(fail(NAME, &typing_failed(typed_chars)));
        }
        if press_enter {
            if let Err(e) = ghost_hands::press_key("return", &[]) {
                let err = input_error(&e).after_delivery(ghost_eyes::Delivery::DeliveredUnverified);
                return Ok(fail(NAME, &err));
            }
        }
        Ok(ok(
            NAME,
            json!({
                "success": true,
                "typed_chars": typed_chars,
                "via": "hid",
                "disposition": hid_disposition(),
            }),
        ))
    }
}

/// Splits `"Ctrl+Shift+T"` into its parts and applies the hotkey policy, so a
/// blocked or unknown combo is refused before any input is attempted.
pub(super) fn parse_hotkey(keys: &str) -> Result<Vec<&str>, GhostError> {
    let parts: Vec<&str> = keys
        .split('+')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return Err(GhostError::invalid_args(
            "`keys` is required (e.g. 'Ctrl+C')",
        ));
    }
    ghost_hands::check_hotkey(&parts, ghost_hands::KeyOptions::default())
        .map_err(|e| input_error(&anyhow::Error::new(e)))?;
    Ok(parts)
}

/// Synthesize keyboard shortcuts.
pub struct AXHotkeyTool;

#[async_trait]
impl AgentTool for AXHotkeyTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_hotkey".to_string(),
            description:
                "Synthesize a keyboard shortcut (e.g. 'Ctrl+C', 'Cmd+Shift+T'). \
                          System shortcuts such as quitting apps or locking the screen are refused."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "string",
                        "description": "Key combination like 'Ctrl+C', 'Cmd+Shift+T', 'F5'"
                    }
                },
                "required": ["keys"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        const NAME: &str = "ax_hotkey";
        let keys = params["keys"].as_str().unwrap_or("");
        let parts = match parse_hotkey(keys) {
            Ok(parts) => parts,
            Err(e) => return Ok(fail(NAME, &e)),
        };
        if let Err(e) = require_hid() {
            return Ok(fail(NAME, &e));
        }
        // The default `send_hotkey` re-applies the policy and sends exactly the checked combo.
        if let Err(e) = ghost_hands::send_hotkey(&parts) {
            return Ok(fail(NAME, &input_error(&e)));
        }
        Ok(ok(
            NAME,
            json!({
                "success": true,
                "keys": keys,
                "via": "hid",
                "disposition": hid_disposition(),
            }),
        ))
    }
}

/// Scroll at screen coordinates.
pub struct AXScrollTool;

const SCROLL_DIRECTIONS: [&str; 4] = ["up", "down", "left", "right"];

/// Validated scroll arguments: `(x, y, direction, amount)`.
pub(super) fn parse_scroll(
    params: &serde_json::Value,
) -> Result<(i32, i32, &str, i32), GhostError> {
    let direction = params["direction"].as_str().unwrap_or("");
    if !SCROLL_DIRECTIONS.contains(&direction) {
        return Err(GhostError::invalid_args(
            "`direction` must be one of up, down, left, right",
        ));
    }
    let x = required_i32(params, "x")?;
    let y = required_i32(params, "y")?;
    let amount = params["amount"]
        .as_i64()
        .map_or(3, |a| a.clamp(1, 100) as i32);
    Ok((x, y, direction, amount))
}

#[async_trait]
impl AgentTool for AXScrollTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_scroll".to_string(),
            description: "Scroll up, down, left or right at specific screen coordinates"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                    "amount": {"type": "integer", "default": 3, "description": "Number of scroll steps"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"}
                },
                "required": ["direction", "x", "y"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        const NAME: &str = "ax_scroll";
        let (x, y, direction, amount) = match parse_scroll(&params) {
            Ok(args) => args,
            Err(e) => return Ok(fail(NAME, &e)),
        };
        if let Err(e) = require_hid() {
            return Ok(fail(NAME, &e));
        }
        if let Err(e) = ghost_hands::scroll(x, y, direction, amount) {
            return Ok(fail(NAME, &input_error(&e)));
        }
        Ok(ok(
            NAME,
            json!({
                "success": true,
                "x": x,
                "y": y,
                "direction": direction,
                "amount": amount,
                "via": "hid",
                "disposition": hid_disposition(),
            }),
        ))
    }
}

/// Bring an app to the foreground.
pub struct AXFocusAppTool;

#[async_trait]
impl AgentTool for AXFocusAppTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_focus_app".to_string(),
            description: "Bring an application window to the foreground by name".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Application name or window title substring"}
                },
                "required": ["app_name"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        const NAME: &str = "ax_focus_app";
        let app_name = match required_str(&params, "app_name") {
            Ok(name) => name,
            Err(e) => return Ok(fail(NAME, &e)),
        };
        if !ghost_hands::focus_app(app_name) {
            let err = GhostError::new(
                ErrorCode::AppNotFound,
                "no running application window matched the name",
            );
            return Ok(fail(NAME, &err));
        }
        Ok(ok(NAME, json!({ "success": true })))
    }
}
