use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use super::super::{AgentTool, Tool, ToolResult};

/// Query semantic memory by category and text.
pub struct GetKnowledgeTool;

#[async_trait]
impl AgentTool for GetKnowledgeTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "get_knowledge".to_string(),
            description: "Query semantic memory for stored facts, preferences, patterns, relationships, or skills".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["preference", "fact", "pattern", "relationship", "skill"],
                        "description": "Type of memory to query"
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to search for"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let query = params["query"].as_str().unwrap_or("");
        let category = params["category"].as_str();

        match crate::memory::MEMORY_STORE.get() {
            Some(store) => {
                let entries = store.lock().unwrap().query(category, query)?;
                Ok(ToolResult {
                    tool_name: "get_knowledge".to_string(),
                    result: json!({ "entries": entries }),
                    error: None,
                })
            }
            None => Ok(ToolResult {
                tool_name: "get_knowledge".to_string(),
                result: json!({ "entries": [] }),
                error: Some("Memory store not initialized".to_string()),
            }),
        }
    }
}

/// List active directives.
pub struct GetDirectivesTool;

#[async_trait]
impl AgentTool for GetDirectivesTool {
    fn definition(&self) -> Tool {
        Tool {
            name: "get_directives".to_string(),
            description: "List all active directives (reminders, habits, automations, watches)"
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "type_filter": {
                        "type": "string",
                        "enum": ["reminder", "habit", "automation", "watch"],
                        "description": "Optional filter by directive type"
                    }
                }
            }),
        }
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult> {
        let type_filter = params["type_filter"].as_str();

        match crate::memory::MEMORY_STORE.get() {
            Some(store) => {
                let directives = store.lock().unwrap().list_active(type_filter)?;
                Ok(ToolResult {
                    tool_name: "get_directives".to_string(),
                    result: json!({ "directives": directives }),
                    error: None,
                })
            }
            None => Ok(ToolResult {
                tool_name: "get_directives".to_string(),
                result: json!({ "directives": [] }),
                error: Some("Memory store not initialized".to_string()),
            }),
        }
    }
}
