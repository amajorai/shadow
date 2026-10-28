use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const DEFAULT_JEV_URL: &str = "https://api.typesafe.ai/v1/systemone";
pub const DEFAULT_JEV_MODEL: &str = "jev-latest";
pub const DEFAULT_LAYA_URL: &str = "http://127.0.0.1:7980/api/ext/@ryu/laya/predict";

/// The planner that owns Shadow's computer-use action loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerUsePlanner {
    /// Keep the existing Shadow LLM tool-calling path.
    Native,
    /// Use Jev's indexed-action decision contract for fast UI actions.
    Jev,
    /// Use a model adapter advertised by an enabled Ryu plugin.
    Local,
    /// Legacy Laya endpoint configuration retained for persisted installs.
    Laya,
}

impl Default for ComputerUsePlanner {
    fn default() -> Self {
        Self::Native
    }
}

/// Runtime configuration for Shadow's optional computer-use planner.
///
/// Jev credentials are deliberately not persisted here. Core injects the
/// encrypted Jev secret into Shadow's process environment; Laya calls stay on
/// the local Core loopback proxy and use the node bearer only for that hop.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ComputerUseConfig {
    #[serde(default = "default_config_version")]
    pub version: u32,
    #[serde(default)]
    pub planner: ComputerUsePlanner,
    #[serde(default = "default_jev_url", alias = "jev_url")]
    pub jev_url: String,
    #[serde(default = "default_jev_model", alias = "jev_model")]
    pub jev_model: String,
    #[serde(default = "default_laya_url", alias = "laya_url")]
    pub laya_url: String,
    #[serde(default, alias = "local_plugin_id")]
    pub local_plugin_id: String,
    #[serde(default, alias = "local_model_id")]
    pub local_model_id: String,
    #[serde(default = "default_max_steps", alias = "max_steps")]
    pub max_steps: u8,
}

fn default_jev_url() -> String {
    DEFAULT_JEV_URL.to_owned()
}

fn default_jev_model() -> String {
    DEFAULT_JEV_MODEL.to_owned()
}

/// The Laya predict route on the local Core. `jevUrl`, `jevModel` and `layaUrl`
/// are changed in `computer-use.json`, not through the environment.
fn default_laya_url() -> String {
    let core_url = std::env::var("RYU_CORE_URL")
        .ok()
        .filter(|value| is_loopback_base_url(value))
        .unwrap_or_else(|| "http://127.0.0.1:7980".to_owned());
    format!(
        "{}/api/ext/@ryu/laya/predict",
        core_url.trim_end_matches('/')
    )
}

fn default_max_steps() -> u8 {
    12
}

fn default_config_version() -> u32 {
    1
}

impl Default for ComputerUseConfig {
    fn default() -> Self {
        Self {
            version: default_config_version(),
            planner: ComputerUsePlanner::default(),
            jev_url: default_jev_url(),
            jev_model: default_jev_model(),
            laya_url: default_laya_url(),
            local_plugin_id: String::new(),
            local_model_id: String::new(),
            max_steps: default_max_steps(),
        }
    }
}

impl ComputerUseConfig {
    pub const MAX_STEPS: u8 = 20;

    pub fn normalized(mut self) -> Self {
        if self.jev_url.trim().is_empty() {
            self.jev_url = default_jev_url();
        }
        if self.jev_model.trim().is_empty() {
            self.jev_model = default_jev_model();
        }
        if self.laya_url.trim().is_empty() {
            self.laya_url = default_laya_url();
        }
        self.max_steps = self.max_steps.clamp(1, Self::MAX_STEPS);
        self
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.version != 1 {
            anyhow::bail!("unsupported computer-use settings version {}", self.version);
        }
        validate_endpoint(&self.jev_url, "jev_url")?;
        validate_loopback_endpoint(&self.laya_url, "laya_url")?;
        if self.jev_url.len() > 2048 || self.laya_url.len() > 2048 {
            anyhow::bail!("computer-use endpoint URLs must be at most 2048 bytes");
        }
        if self.planner == ComputerUsePlanner::Local {
            validate_local_plugin_id(&self.local_plugin_id)?;
            if self.local_model_id.trim().is_empty()
                || self.local_model_id.len() > 256
                || self.local_model_id.chars().any(char::is_control)
            {
                anyhow::bail!("local_model_id must be 1..=256 printable characters");
            }
        }
        if self.jev_model.trim().is_empty() || self.jev_model.len() > 128 {
            anyhow::bail!("jev_model must be 1..=128 characters");
        }
        if !(1..=Self::MAX_STEPS).contains(&self.max_steps) {
            anyhow::bail!("max_steps must be between 1 and {}", Self::MAX_STEPS);
        }
        Ok(())
    }

    pub fn local_planner_url(&self) -> anyhow::Result<String> {
        if self.planner != ComputerUsePlanner::Local {
            anyhow::bail!("a local planner is not selected");
        }
        validate_local_plugin_id(&self.local_plugin_id)?;
        let base = std::env::var("RYU_CORE_URL")
            .ok()
            .filter(|value| is_loopback_base_url(value))
            .unwrap_or_else(|| "http://127.0.0.1:7980".to_owned());
        Ok(format!(
            "{}/api/ext/{}/computer-use",
            base.trim_end_matches('/'),
            self.local_plugin_id
        ))
    }

    /// Load persisted settings from Shadow's own data directory.
    pub fn load(data_dir: &std::path::Path) -> Self {
        let path = data_dir.join("computer-use.json");
        if std::fs::metadata(&path)
            .map(|metadata| metadata.len() > 64 * 1024)
            .unwrap_or(false)
        {
            tracing::warn!(
                "ignoring oversized Shadow computer-use settings at {}",
                path.display()
            );
            return Self::default();
        }
        match std::fs::read_to_string(&path) {
            Ok(raw) => match serde_json::from_str::<Self>(&raw) {
                Ok(value) => {
                    let value = value.normalized();
                    match value.validate() {
                        Ok(()) => value,
                        Err(error) => {
                            tracing::warn!(
                                "ignoring invalid Shadow computer-use settings at {}: {error}",
                                path.display()
                            );
                            Self::default()
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(
                        "ignoring invalid Shadow computer-use settings at {}: {error}",
                        path.display()
                    );
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Persist only non-secret settings atomically.
    pub fn save(&self, data_dir: &std::path::Path) -> anyhow::Result<()> {
        self.validate()?;
        std::fs::create_dir_all(data_dir)?;
        let path = data_dir.join("computer-use.json");
        let temp = data_dir.join("computer-use.json.tmp");
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(&temp, bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&temp, &path)?;
        Ok(())
    }
}

fn validate_local_plugin_id(value: &str) -> anyhow::Result<()> {
    let valid_segment = |segment: &str| {
        !segment.is_empty()
            && segment.len() <= 128
            && !segment.contains("..")
            && !segment.starts_with('.')
            && !segment.starts_with('-')
            && !segment.ends_with('.')
            && segment
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
    };
    let valid = if let Some(scoped) = value.strip_prefix('@') {
        scoped.split_once('/').is_some_and(|(scope, name)| {
            !name.contains('/') && valid_segment(scope) && valid_segment(name)
        })
    } else {
        valid_segment(value)
    };
    if !valid || value.len() > 128 {
        anyhow::bail!("local_plugin_id is not a valid Ryu plugin id");
    }
    Ok(())
}

fn validate_endpoint(value: &str, field: &str) -> anyhow::Result<()> {
    let url =
        reqwest::Url::parse(value).map_err(|_| anyhow::anyhow!("{field} must be a valid URL"))?;
    if url.username() != "" || url.password().is_some() || url.query().is_some() {
        anyhow::bail!("{field} must not contain credentials or a query string");
    }
    let Some(host) = url.host_str() else {
        anyhow::bail!("{field} must include a host");
    };
    match url.scheme() {
        "https" => {}
        "http" if matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1") => {}
        _ => anyhow::bail!("{field} must use HTTPS; HTTP is allowed only for localhost"),
    }
    Ok(())
}

fn validate_loopback_endpoint(value: &str, field: &str) -> anyhow::Result<()> {
    validate_endpoint(value, field)?;
    let url = reqwest::Url::parse(value)?;
    let host = url.host_str().unwrap_or_default();
    if !matches!(
        host.trim_matches(['[', ']']),
        "localhost" | "127.0.0.1" | "::1"
    ) {
        anyhow::bail!("{field} must use the local Core loopback address");
    }
    Ok(())
}

fn is_loopback_base_url(value: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(value) else {
        return false;
    };
    matches!(
        url.host_str().unwrap_or_default(),
        "localhost" | "127.0.0.1" | "[::1]" | "::1"
    ) && matches!(url.scheme(), "http" | "https")
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && matches!(url.path(), "" | "/")
}

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

    /// Computer-use planner and its non-secret endpoint settings.
    #[serde(default)]
    pub computer_use: ComputerUseConfig,

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
        let data_dir = crate::paths::data_dir();

        let port = std::env::var("SHADOW_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3030);
        let computer_use = ComputerUseConfig::load(&data_dir);

        Self {
            data_dir,
            port,
            capture: CaptureConfig::default(),
            llm: LlmConfig::default(),
            computer_use,
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
#[path = "config_tests.rs"]
mod tests;
