//! `ax_click`: ground a description, then press the element.
//!
//! An accessibility hit is pressed semantically (`perform(Press)` through
//! `ghost_eyes`). Physical input is used only for a vision (coordinate) hit, or
//! as a reported fallback when the element has no press action, and only when
//! `hid_permitted(true)` allows it.

use anyhow::Result;
use async_trait::async_trait;
use ghost_eyes::accessibility::AXTree as _;
use ghost_eyes::{ActionOptions, Delivery, Disposition, ElementAction, ErrorCode, GhostError};
use serde_json::{json, Value};

use super::{fail, hid_disposition, input_error, ok, require_hid, required_str};
use crate::agent::{AgentTool, Tool, ToolResult};
use crate::intelligence::grounding::{AxTarget, GroundingResult};

const NAME: &str = "ax_click";

/// Click UI element by description.
pub struct AXClickTool;

#[async_trait]
impl AgentTool for AXClickTool {
    fn definition(&self) -> Tool {
        Tool {
            name: NAME.to_string(),
            description: "Click a UI element matching a description. Accessibility matches \
                          (exact, then fuzzy) are pressed through the accessibility API; a \
                          vision match falls back to a mouse click on a desktop session."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }),
        }
    }

    async fn execute(&self, params: Value) -> Result<ToolResult> {
        let description = match required_str(&params, "description") {
            Ok(d) => d,
            Err(e) => return Ok(fail(NAME, &e)),
        };

        let (sw, sh) = crate::capture::screen::get_primary_display_size();
        let frame = crate::capture::screen::quick_screenshot(0).await.ok();
        let screenshot_data = frame.as_ref().map(|f| f.data.as_slice()).unwrap_or(&[]);

        let oracle = crate::intelligence::GroundingOracle::new()?;
        let grounding = match oracle.ground(description, screenshot_data, sw, sh).await {
            Ok(g) => g,
            Err(_) => {
                let err = GhostError::not_found("no element matched the description");
                return Ok(fail(NAME, &err));
            }
        };

        let base = json!({
            "strategy": format!("{:?}", grounding.strategy),
            "confidence": grounding.confidence,
        });
        Ok(match grounding.ax_target.clone() {
            Some(target) => semantic_click(&target, base).await,
            None => {
                let x = (grounding.x * sw as f32) as i32;
                let y = (grounding.y * sh as f32) as i32;
                vision_click(&grounding, x, y, base)
            }
        })
    }
}

/// Whether a semantic failure allows a physical fallback (nothing was delivered
/// and the element or platform has no semantic press).
pub(super) fn allows_hid_fallback(err: &GhostError) -> bool {
    matches!(
        err.code,
        ErrorCode::ActionNotSupported | ErrorCode::PlatformNotSupported
    ) && err.delivery == Delivery::NotDelivered
}

fn merge(mut base: Value, extra: Value) -> Value {
    if let (Some(obj), Value::Object(extra)) = (base.as_object_mut(), extra) {
        obj.extend(extra);
    }
    base
}

async fn semantic_click(target: &AxTarget, base: Value) -> ToolResult {
    let (x, y) = target.center;
    let ax = match crate::capture::accessibility::PlatformAXTree::new() {
        Ok(ax) => ax,
        Err(e) => return fail(NAME, &GhostError::from_anyhow(&e)),
    };
    let outcome = ax
        .eyes()
        .perform(
            &target.locator,
            &ElementAction::Press,
            &ActionOptions::default(),
        )
        .await;
    match outcome {
        Ok(outcome) => ok(
            NAME,
            merge(
                base,
                json!({
                    "success": true,
                    "x": x,
                    "y": y,
                    "via": "semantic",
                    "method": outcome.method,
                    "disposition": Disposition::from(outcome.delivery),
                    "verification": outcome.verification,
                }),
            ),
        ),
        Err(err) if allows_hid_fallback(&err) => {
            if let Err(denied) = require_hid() {
                return fail(NAME, &err.with_suggestion(denied.message));
            }
            hid_click(
                x,
                y,
                merge(base, json!({ "fallback_reason": err.code.as_str() })),
            )
        }
        Err(err) => fail(NAME, &err),
    }
}

fn vision_click(grounding: &GroundingResult, x: i32, y: i32, base: Value) -> ToolResult {
    if let Err(e) = require_hid() {
        return fail(NAME, &e);
    }
    tracing::debug!(
        "ax_click: vision grounding hit (confidence {:.2})",
        grounding.confidence
    );
    hid_click(x, y, base)
}

fn hid_click(x: i32, y: i32, base: Value) -> ToolResult {
    // ghost_hands dispatches to SendInput (Windows) / CGEvent (macOS) / XTEST (Linux).
    if let Err(e) = ghost_hands::mouse_click(x, y, ghost_hands::MouseButton::Left, 1) {
        return fail(NAME, &input_error(&e));
    }
    ok(
        NAME,
        merge(
            base,
            json!({
                "success": true,
                "x": x,
                "y": y,
                "via": "hid",
                "disposition": hid_disposition(),
            }),
        ),
    )
}
