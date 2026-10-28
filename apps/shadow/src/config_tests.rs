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
        computer_use: ComputerUseConfig::default(),
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
    assert_eq!(back.computer_use.planner, ComputerUsePlanner::Native);
    assert_eq!(back.computer_use.version, 1);
}

#[test]
fn computer_use_defaults_are_safe_and_bounded() {
    let config = ComputerUseConfig::default();
    assert_eq!(config.planner, ComputerUsePlanner::Native);
    assert_eq!(config.max_steps, 12);
    assert!(config.validate().is_ok());
}

#[test]
fn computer_use_rejects_cleartext_remote_endpoints() {
    let mut config = ComputerUseConfig::default();
    config.jev_url = "http://example.com/choose".to_owned();
    assert!(config.validate().is_err());
    config.jev_url = "http://127.0.0.1:8099/choose".to_owned();
    assert!(config.validate().is_ok());
}

#[test]
fn local_planner_requires_a_safe_plugin_id_and_model_id() {
    let mut config = ComputerUseConfig::default();
    config.planner = ComputerUsePlanner::Local;
    config.local_plugin_id = "@ryu/needle3".to_owned();
    config.local_model_id = "needle3".to_owned();
    assert!(config.validate().is_ok());

    config.local_plugin_id = "@ryu/../../outside".to_owned();
    assert!(config.validate().is_err());
    config.local_plugin_id = "@ryu/needle3".to_owned();
    config.local_model_id = "bad\nmodel".to_owned();
    assert!(config.validate().is_err());
}

#[test]
fn laya_endpoint_cannot_send_the_node_bearer_to_a_remote_host() {
    let mut config = ComputerUseConfig::default();
    config.laya_url = "https://api.example.com/predict".to_owned();
    assert!(config.validate().is_err());
    config.laya_url = "http://127.0.0.1:8980/api/ext/@ryu/laya/predict".to_owned();
    assert!(config.validate().is_ok());
}

#[test]
fn computer_use_normalizes_step_limit() {
    let mut config = ComputerUseConfig::default();
    config.max_steps = 255;
    assert_eq!(config.normalized().max_steps, ComputerUseConfig::MAX_STEPS);
}

#[test]
fn computer_use_settings_save_and_load_versioned_data() {
    let dir = std::env::temp_dir().join(format!(
        "shadow-computer-use-{}",
        uuid::Uuid::new_v4().simple()
    ));
    let mut config = ComputerUseConfig::default();
    config.planner = ComputerUsePlanner::Jev;
    config.max_steps = 7;

    config.save(&dir).unwrap();
    let loaded = ComputerUseConfig::load(&dir);

    assert_eq!(loaded.version, 1);
    assert_eq!(loaded.planner, ComputerUsePlanner::Jev);
    assert_eq!(loaded.max_steps, 7);
    let serialized = std::fs::read_to_string(dir.join("computer-use.json")).unwrap();
    assert!(!serialized.contains("SHADOW_JEV_API_KEY"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn computer_use_settings_recover_to_defaults_when_corrupt_or_unknown_version() {
    for raw in [
        "not-json",
        r#"{"version":99,"planner":"jev","max_steps":8}"#,
        r#"{"version":1,"planner":"native","unknown":"value"}"#,
    ] {
        let dir = std::env::temp_dir().join(format!(
            "shadow-computer-use-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("computer-use.json"), raw).unwrap();
        assert_eq!(ComputerUseConfig::load(&dir), ComputerUseConfig::default());
        let _ = std::fs::remove_dir_all(dir);
    }
}
