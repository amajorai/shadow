//! Read-only accessibility tools: tree query, wait, text, inspect and element-at.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use ghost_eyes::accessibility::AXTree as EyesTree;
use ghost_eyes::{Delivery, GhostError};

use super::{fail, ok, required_i32, required_str};
use crate::agent::{AgentTool, Tool, ToolResult};
use crate::capture::accessibility::{AXTree, AXTreeNode, PlatformAXTree};

/// The focused window's tree, or the contract error for the tool result.
async fn focused_tree() -> Result<AXTreeNode, GhostError> {
    let ax = PlatformAXTree::new().map_err(|e| GhostError::from_anyhow(&e))?;
    ax.get_focused_tree()
        .await
        .map_err(|e| GhostError::from_anyhow(&e))
}

/// Query accessibility tree.
pub struct AXTreeQueryTool;

#[async_trait]
impl AgentTool for AXTreeQueryTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_tree_query".to_string(),
            description: "Query the accessibility tree for the focused app, optionally filtered by role or title".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "Optional filter by role or title substring"}
                }
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let filter = params["filter"].as_str();
        let tree = match focused_tree().await {
            Ok(tree) => tree,
            Err(e) => return Ok(fail("ax_tree_query", &e)),
        };

        let tree_json = serde_json::to_value(&tree)?;
        let result = if let Some(f) = filter {
            filter_ax_tree(&tree_json, f)
        } else {
            tree_json
        };

        Ok(ok("ax_tree_query", json!({ "tree": result })))
    }
}

fn filter_ax_tree(node: &serde_json::Value, filter: &str) -> serde_json::Value {
    let filter_lower = filter.to_lowercase();
    let role = node["role"].as_str().unwrap_or("").to_lowercase();
    let title = node["title"].as_str().unwrap_or("").to_lowercase();
    if role.contains(&filter_lower) || title.contains(&filter_lower) {
        return node.clone();
    }
    if let Some(children) = node["children"].as_array() {
        let filtered: Vec<_> = children
            .iter()
            .filter_map(|c| {
                let r = filter_ax_tree(c, filter);
                if r.is_null() {
                    None
                } else {
                    Some(r)
                }
            })
            .collect();
        if !filtered.is_empty() {
            return json!({ "role": node["role"], "title": node["title"], "children": filtered });
        }
    }
    serde_json::Value::Null
}

/// Wait for an element condition with timeout.
pub struct AXWaitTool;

#[async_trait]
impl AgentTool for AXWaitTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_wait".to_string(),
            description: "Wait for a UI element to appear or match a condition, with timeout"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "Description of the element to wait for"},
                    "timeout_ms": {"type": "integer", "default": 5000}
                },
                "required": ["condition"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        const NAME: &str = "ax_wait";
        let condition = match required_str(&params, "condition") {
            Ok(c) => c,
            Err(e) => return Ok(fail(NAME, &e)),
        };
        let timeout_ms = params["timeout_ms"].as_u64().unwrap_or(5000).min(120_000);
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(timeout_ms);
        let ax = match PlatformAXTree::new() {
            Ok(ax) => ax,
            Err(e) => return Ok(fail(NAME, &GhostError::from_anyhow(&e))),
        };

        loop {
            if let Some(element) = ax.find_element(condition).await {
                return Ok(ok(NAME, json!({ "found": true, "element": element })));
            }

            if tokio::time::Instant::now() >= deadline {
                let err = GhostError::timeout(
                    format!("no element matched the condition within {timeout_ms} ms"),
                    Delivery::NotDelivered,
                );
                let mut result = fail(NAME, &err);
                if let Some(obj) = result.result.as_object_mut() {
                    obj.insert("found".to_string(), json!(false));
                    obj.insert("timed_out".to_string(), json!(true));
                }
                return Ok(result);
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }
    }
}

/// Extract all text from an element subtree.
pub struct AXReadTextTool;

#[async_trait]
impl AgentTool for AXReadTextTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_read_text".to_string(),
            description: "Extract all visible text from the focused window's accessibility tree"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "filter": {"type": "string", "description": "Optional filter by element role"}
                }
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let filter = params["filter"].as_str();
        let tree = match focused_tree().await {
            Ok(tree) => tree,
            Err(e) => return Ok(fail("ax_read_text", &e)),
        };

        let mut texts = vec![];
        collect_texts(&tree, filter, &mut texts);

        Ok(ok("ax_read_text", json!({ "texts": texts })))
    }
}

fn collect_texts(node: &AXTreeNode, role_filter: Option<&str>, out: &mut Vec<String>) {
    let matches = role_filter
        .map(|f| node.role.to_lowercase().contains(&f.to_lowercase()))
        .unwrap_or(true);

    if matches {
        if let Some(v) = &node.value {
            if !v.is_empty() {
                out.push(v.clone());
            }
        }
        if let Some(t) = &node.title {
            if !t.is_empty() {
                out.push(t.clone());
            }
        }
    }
    for child in &node.children {
        collect_texts(child, role_filter, out);
    }
}

/// Detailed element inspection.
pub struct AXInspectTool;

#[async_trait]
impl AgentTool for AXInspectTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_inspect".to_string(),
            description: "Get detailed accessibility attributes for a specific element matching a description".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        const NAME: &str = "ax_inspect";
        let description = match required_str(&params, "description") {
            Ok(d) => d,
            Err(e) => return Ok(fail(NAME, &e)),
        };
        let ax = match PlatformAXTree::new() {
            Ok(ax) => ax,
            Err(e) => return Ok(fail(NAME, &GhostError::from_anyhow(&e))),
        };
        match ax.find_element(description).await {
            Some(element) => Ok(ok(NAME, json!({ "element": element }))),
            None => Ok(fail(
                NAME,
                &GhostError::not_found("no element matched the description"),
            )),
        }
    }
}

/// Get element at screen coordinates.
pub struct AXElementAtTool;

#[async_trait]
impl AgentTool for AXElementAtTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "ax_element_at".to_string(),
            description: "Get the accessibility element at specific screen coordinates".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"}
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        const NAME: &str = "ax_element_at";
        let (x, y) = match (required_i32(&params, "x"), required_i32(&params, "y")) {
            (Ok(x), Ok(y)) => (x, y),
            (Err(e), _) | (_, Err(e)) => return Ok(fail(NAME, &e)),
        };
        let ax = match PlatformAXTree::new() {
            Ok(ax) => ax,
            Err(e) => return Ok(fail(NAME, &GhostError::from_anyhow(&e))),
        };
        match EyesTree::element_at(ax.eyes(), x, y).await {
            Some(node) => Ok(ok(
                NAME,
                json!({ "x": x, "y": y, "element": AXTreeNode::from(&node) }),
            )),
            None => Ok(fail(
                NAME,
                &GhostError::not_found(format!("no accessibility element at ({x}, {y})")),
            )),
        }
    }
}
