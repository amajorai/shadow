//! System tools: running-app listing and live screenshot capture.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::agent::{AgentTool, Tool, ToolResult};

/// List running applications.
pub struct AXListAppsTool;

#[async_trait]
impl AgentTool for AXListAppsTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_list_apps".to_string(),
            description: "List all running applications with their process IDs".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn execute(&self, _params: serde_json::Value) -> Result<ToolResult> {
        use ghost_eyes::accessibility::AXTree as _;
        let ax = match crate::capture::accessibility::PlatformAXTree::new() {
            Ok(ax) => ax,
            Err(e) => {
                return Ok(super::fail(
                    "ax_list_apps",
                    &ghost_eyes::GhostError::from_anyhow(&e),
                ))
            }
        };
        let apps = ax.eyes().list_apps().await;
        Ok(super::ok("ax_list_apps", json!({ "apps": apps })))
    }
}

/// Capture current screen as base64 PNG.
pub struct CaptureLiveScreenshotTool;

#[async_trait]
impl AgentTool for CaptureLiveScreenshotTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "capture_live_screenshot".to_string(),
            description: "Capture the current screen and return it as a base64 PNG for analysis"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "display_id": {"type": "integer", "default": 0}
                }
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let display_id = params["display_id"].as_u64().unwrap_or(0) as u32;

        let frame = crate::capture::screen::quick_screenshot(display_id)
            .await
            .map_err(|e| anyhow::anyhow!("Screenshot failed: {}", e))?;

        // Convert BGRA → RGBA for PNG encoding
        let mut rgba = vec![0u8; frame.data.len()];
        for i in (0..frame.data.len()).step_by(4) {
            rgba[i] = frame.data[i + 2]; // R
            rgba[i + 1] = frame.data[i + 1]; // G
            rgba[i + 2] = frame.data[i]; // B
            rgba[i + 3] = frame.data[i + 3]; // A
        }
        let img = image::RgbaImage::from_raw(frame.width, frame.height, rgba)
            .ok_or_else(|| anyhow::anyhow!("Image buffer size mismatch"))?;

        let mut png_bytes = Vec::new();
        image::DynamicImage::ImageRgba8(img).write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )?;

        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_bytes);

        Ok(ToolResult {
            tool_name: "capture_live_screenshot".to_string(),
            result: json!({
                "png_base64": b64,
                "width":  frame.width,
                "height": frame.height,
                "display_id": display_id
            }),
            error: None,
        })
    }
}
