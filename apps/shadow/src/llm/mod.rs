pub mod ollama;
pub mod openai_compat;
pub mod orchestrator;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A single message in an LLM conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Text(s) => s,
            Self::Parts(_) => "",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl LlmMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: MessageContent::text(content),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: MessageContent::text(content),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: MessageContent::text(content),
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: MessageContent::Text(format!(
                "{{\"tool_call_id\":\"{}\",\"content\":{}}}",
                tool_call_id.into(),
                serde_json::to_string(&content.into()).unwrap_or_default()
            )),
        }
    }
}

/// Tool definition passed to the LLM (JSON Schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// A tool call returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Request to an LLM provider.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub messages: Vec<LlmMessage>,
    pub tools: Vec<ToolDefinition>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub stream: bool,
}

impl Default for LlmRequest {
    fn default() -> Self {
        Self {
            messages: vec![],
            tools: vec![],
            temperature: 0.7,
            max_tokens: 4096,
            stream: false,
        }
    }
}

/// Response from an LLM provider.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// LLM provider trait — implemented by OpenAI-compat client.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate(&self, req: LlmRequest) -> anyhow::Result<LlmResponse>;

    /// Stream text tokens; returns the full response at the end.
    async fn stream(
        &self,
        req: LlmRequest,
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> anyhow::Result<LlmResponse>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors_set_role_and_text() {
        assert_eq!(LlmMessage::system("s").role, "system");
        assert_eq!(LlmMessage::user("u").role, "user");
        assert_eq!(LlmMessage::assistant("a").role, "assistant");
        assert_eq!(LlmMessage::user("hello").content.as_str(), "hello");
    }

    #[test]
    fn message_content_as_str_is_empty_for_parts() {
        let parts = MessageContent::Parts(vec![ContentPart {
            part_type: "text".to_string(),
            text: Some("ignored".to_string()),
        }]);
        assert_eq!(parts.as_str(), "");
        assert_eq!(MessageContent::text("x").as_str(), "x");
    }

    #[test]
    fn tool_result_embeds_id_and_json_encoded_content() {
        let msg = LlmMessage::tool_result("call_1", "the result");
        assert_eq!(msg.role, "tool");
        let s = msg.content.as_str();
        assert!(s.contains("\"tool_call_id\":\"call_1\""));
        assert!(s.contains("\"the result\""));
    }

    #[test]
    fn text_content_serializes_untagged_as_string() {
        let msg = LlmMessage::user("hi");
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(v["role"], serde_json::json!("user"));
        // Untagged MessageContent::Text serializes as a bare string.
        assert_eq!(v["content"], serde_json::json!("hi"));
    }

    #[test]
    fn llm_request_defaults() {
        let req = LlmRequest::default();
        assert!(req.messages.is_empty());
        assert!(req.tools.is_empty());
        assert!((req.temperature - 0.7).abs() < 1e-6);
        assert_eq!(req.max_tokens, 4096);
        assert!(!req.stream);
    }

    #[test]
    fn content_part_skips_none_text_on_serialize() {
        let part = ContentPart {
            part_type: "image".to_string(),
            text: None,
        };
        let v = serde_json::to_value(&part).unwrap();
        assert_eq!(v["type"], serde_json::json!("image"));
        assert!(v.get("text").is_none(), "None text must be skipped");
    }
}
