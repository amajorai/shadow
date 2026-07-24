use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Adaptive thresholds adjusted by user feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustParams {
    /// Minimum confidence to push a suggestion immediately.
    pub confidence_threshold: f32,
    /// Minimum score for inbox delivery (vs drop).
    pub inbox_threshold: f32,
    /// Score penalty per recent repetition of the same suggestion type.
    pub repetition_penalty: f32,
    /// Per-type cooldown in seconds (increased on dismissals).
    pub cooldown_by_type: HashMap<String, u64>,
    /// Per-type weight multiplier (increased on thumbs-up).
    pub preferred_types: HashMap<String, f32>,
}

impl Default for TrustParams {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.82,
            inbox_threshold: 0.55,
            repetition_penalty: 0.05,
            cooldown_by_type: HashMap::new(),
            preferred_types: HashMap::new(),
        }
    }
}

/// How the user responded to a proactive suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackKind {
    ThumbsUp,
    ThumbsDown,
    Dismiss,
    Snooze,
}

/// Adjusts delivery parameters based on accumulated user feedback.
pub struct TrustTuner {
    params: TrustParams,
    persist_path: Option<std::path::PathBuf>,
}

impl TrustTuner {
    const CONFIDENCE_MIN: f32 = 0.40;
    const CONFIDENCE_MAX: f32 = 0.95;
    const INBOX_MIN: f32 = 0.20;
    const INBOX_MAX: f32 = 0.80;
    const REPETITION_MAX: f32 = 0.30;
    const COOLDOWN_INCREMENT_SECS: u64 = 15 * 60; // 15 min
    const COOLDOWN_CAP_SECS: u64 = 60 * 60; // 1 hr

    pub fn new() -> Self {
        Self {
            params: TrustParams::default(),
            persist_path: None,
        }
    }

    /// Load from JSON file, falling back to defaults on error.
    pub fn load(path: &std::path::Path) -> Self {
        let params = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self {
            params,
            persist_path: Some(path.to_path_buf()),
        }
    }

    /// Current parameters (read-only snapshot).
    pub fn params(&self) -> &TrustParams {
        &self.params
    }

    /// Apply a feedback signal and optionally persist.
    pub fn apply_feedback(&mut self, kind: FeedbackKind, suggestion_type: &str) {
        match kind {
            FeedbackKind::ThumbsUp => {
                self.params.confidence_threshold =
                    (self.params.confidence_threshold - 0.02).max(Self::CONFIDENCE_MIN);
                // Boost preference weight for this type
                let weight = self
                    .params
                    .preferred_types
                    .entry(suggestion_type.to_string())
                    .or_insert(1.0);
                *weight = (*weight + 0.1).min(2.0);
            }
            FeedbackKind::ThumbsDown => {
                self.params.confidence_threshold =
                    (self.params.confidence_threshold + 0.03).min(Self::CONFIDENCE_MAX);
                self.params.repetition_penalty =
                    (self.params.repetition_penalty + 0.02).min(Self::REPETITION_MAX);
            }
            FeedbackKind::Dismiss | FeedbackKind::Snooze => {
                let cooldown = self
                    .params
                    .cooldown_by_type
                    .entry(suggestion_type.to_string())
                    .or_insert(0);
                *cooldown =
                    (*cooldown + Self::COOLDOWN_INCREMENT_SECS).min(Self::COOLDOWN_CAP_SECS);
            }
        }

        self.params.inbox_threshold = self
            .params
            .inbox_threshold
            .clamp(Self::INBOX_MIN, Self::INBOX_MAX);

        self.save_if_path();
    }

    /// Effective push threshold for a suggestion type (accounts for preference boost).
    pub fn push_threshold_for(&self, suggestion_type: &str) -> f32 {
        let base = self.params.confidence_threshold;
        let boost = self
            .params
            .preferred_types
            .get(suggestion_type)
            .copied()
            .unwrap_or(1.0);
        // Boost lowers the threshold (user prefers this type)
        (base / boost).max(Self::CONFIDENCE_MIN)
    }

    /// Returns the configured cooldown for this suggestion type (0 = none).
    pub fn cooldown_for(&self, suggestion_type: &str) -> u64 {
        self.params
            .cooldown_by_type
            .get(suggestion_type)
            .copied()
            .unwrap_or(0)
    }

    fn save_if_path(&self) {
        let _ = self.persist();
    }

    pub fn persist(&self) -> Result<()> {
        if let Some(path) = &self.persist_path {
            let json = serde_json::to_string_pretty(&self.params)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, json)?;
        }
        Ok(())
    }
}

impl Default for TrustTuner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let t = TrustTuner::new();
        assert!((t.params().confidence_threshold - 0.82).abs() < 1e-6);
        assert!((t.params().inbox_threshold - 0.55).abs() < 1e-6);
        assert!(t.params().preferred_types.is_empty());
        assert!(t.params().cooldown_by_type.is_empty());
    }

    #[test]
    fn thumbs_up_lowers_threshold_and_boosts_preference() {
        let mut t = TrustTuner::new();
        t.apply_feedback(FeedbackKind::ThumbsUp, "reply_draft");
        assert!((t.params().confidence_threshold - 0.80).abs() < 1e-6);
        assert!((t.params().preferred_types["reply_draft"] - 1.1).abs() < 1e-6);
    }

    #[test]
    fn thumbs_up_confidence_floor_is_respected() {
        let mut t = TrustTuner::new();
        // 0.82 - 21*0.02 would go well below the 0.40 floor.
        for _ in 0..40 {
            t.apply_feedback(FeedbackKind::ThumbsUp, "x");
        }
        assert!(t.params().confidence_threshold >= 0.40 - 1e-6);
        // Preference weight caps at 2.0.
        assert!((t.params().preferred_types["x"] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn thumbs_down_raises_threshold_and_repetition_penalty() {
        let mut t = TrustTuner::new();
        t.apply_feedback(FeedbackKind::ThumbsDown, "reply_draft");
        assert!((t.params().confidence_threshold - 0.85).abs() < 1e-6);
        assert!((t.params().repetition_penalty - 0.07).abs() < 1e-6);
    }

    #[test]
    fn thumbs_down_confidence_ceiling_is_respected() {
        let mut t = TrustTuner::new();
        for _ in 0..40 {
            t.apply_feedback(FeedbackKind::ThumbsDown, "x");
        }
        assert!(t.params().confidence_threshold <= 0.95 + 1e-6);
        assert!(t.params().repetition_penalty <= 0.30 + 1e-6);
    }

    #[test]
    fn dismiss_and_snooze_increase_cooldown_and_cap_it() {
        let mut t = TrustTuner::new();
        t.apply_feedback(FeedbackKind::Dismiss, "digest");
        assert_eq!(t.cooldown_for("digest"), 15 * 60);
        t.apply_feedback(FeedbackKind::Snooze, "digest");
        assert_eq!(t.cooldown_for("digest"), 30 * 60);
        // Cap at 1 hour.
        for _ in 0..10 {
            t.apply_feedback(FeedbackKind::Dismiss, "digest");
        }
        assert_eq!(t.cooldown_for("digest"), 60 * 60);
        // Untracked type has no cooldown.
        assert_eq!(t.cooldown_for("other"), 0);
    }

    #[test]
    fn push_threshold_lowers_with_preference_boost() {
        let mut t = TrustTuner::new();
        let base = t.push_threshold_for("digest");
        assert!((base - 0.82).abs() < 1e-6);
        t.apply_feedback(FeedbackKind::ThumbsUp, "digest");
        // Preferred type divides the (now-lowered) threshold by its weight.
        assert!(t.push_threshold_for("digest") < base);
        assert!(t.push_threshold_for("digest") >= 0.40 - 1e-6);
    }

    #[test]
    fn load_falls_back_to_defaults_on_missing_file_then_persists() {
        let path = std::env::temp_dir().join(format!("shadow-trust-{}.json", uuid::Uuid::new_v4()));
        let mut t = TrustTuner::load(&path);
        // Missing file → defaults.
        assert!((t.params().confidence_threshold - 0.82).abs() < 1e-6);
        // apply_feedback persists because load() set a persist path.
        t.apply_feedback(FeedbackKind::ThumbsUp, "digest");
        assert!(path.exists(), "feedback must persist to disk");

        // Reloading recovers the saved state.
        let reloaded = TrustTuner::load(&path);
        assert!(reloaded.params().preferred_types.contains_key("digest"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn new_tuner_does_not_write_to_disk() {
        let mut t = TrustTuner::new();
        t.apply_feedback(FeedbackKind::ThumbsUp, "x");
        // No persist path → persist() is a no-op and does not error.
        assert!(t.persist().is_ok());
    }
}
