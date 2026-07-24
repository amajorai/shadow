use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Shadow sidecar configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Root data directory for all Shadow data.
    pub data_dir: PathBuf,

    /// HTTP server port.
    pub port: u16,

    /// Capture settings.
    pub capture: CaptureConfig,

    /// LLM provider settings.
    pub llm: LlmConfig,

    /// Retention policy settings.
    pub retention: RetentionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    /// Enable screen capture.
    pub screen: bool,

    /// Enable input monitoring (keyboard, mouse).
    pub input: bool,

    /// Enable audio capture.
    pub audio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// OpenAI-compatible base URL (e.g. "http://localhost:11434/v1" for Ollama).
    pub base_url: String,

    /// Model name (e.g. "qwen2.5:7b", "llama3.2", "gpt-4o-mini").
    pub model: String,

    /// API key — empty string for local providers.
    pub api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionConfig {
    /// Days to keep full-resolution video (hot tier).
    pub hot_days: u32,

    /// Days to keep keyframes after video deletion (warm tier).
    pub warm_days: u32,

    /// Maximum total storage in gigabytes before enforced cleanup.
    pub max_storage_gb: u64,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            screen: true,
            input: true,
            audio: true,
        }
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        let base_url = std::env::var("SHADOW_LLM_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
        let model = std::env::var("SHADOW_LLM_MODEL").unwrap_or_else(|_| "qwen2.5:7b".to_string());
        let api_key = std::env::var("SHADOW_LLM_API_KEY").unwrap_or_default();
        Self {
            base_url,
            model,
            api_key,
        }
    }
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            hot_days: 7,
            warm_days: 23,
            max_storage_gb: 50,
        }
    }
}

impl Config {
    /// Create a new configuration with defaults, respecting environment variables.
    pub fn new() -> Self {
        let data_dir = std::env::var("SHADOW_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join(".shadow")
            });

        let port = std::env::var("SHADOW_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3030);

        Self {
            data_dir,
            port,
            capture: CaptureConfig::default(),
            llm: LlmConfig::default(),
            retention: RetentionConfig::default(),
        }
    }

    /// Path to the memory database.
    pub fn memory_db_path(&self) -> PathBuf {
        self.data_dir.join("indices").join("memory.db")
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a Config without touching process env (Config::new reads env vars).
    fn fixture(data_dir: &str) -> Config {
        Config {
            data_dir: PathBuf::from(data_dir),
            port: 3030,
            capture: CaptureConfig::default(),
            llm: LlmConfig {
                base_url: "http://localhost:11434/v1".to_string(),
                model: "qwen2.5:7b".to_string(),
                api_key: String::new(),
            },
            retention: RetentionConfig::default(),
        }
    }

    #[test]
    fn capture_config_default_enables_all_sources() {
        let c = CaptureConfig::default();
        assert!(c.screen && c.input && c.audio);
    }

    #[test]
    fn retention_config_defaults() {
        let r = RetentionConfig::default();
        assert_eq!(r.hot_days, 7);
        assert_eq!(r.warm_days, 23);
        assert_eq!(r.max_storage_gb, 50);
    }

    #[test]
    fn memory_db_path_is_under_indices() {
        let cfg = fixture("/var/shadow");
        assert_eq!(
            cfg.memory_db_path(),
            PathBuf::from("/var/shadow")
                .join("indices")
                .join("memory.db")
        );
    }

    #[test]
    fn config_serde_round_trips() {
        let cfg = fixture("/data");
        let json = serde_json::to_string(&cfg).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.port, 3030);
        assert_eq!(back.data_dir, PathBuf::from("/data"));
        assert_eq!(back.llm.model, "qwen2.5:7b");
        assert_eq!(back.retention.warm_days, 23);
        assert!(back.capture.audio);
    }
}
