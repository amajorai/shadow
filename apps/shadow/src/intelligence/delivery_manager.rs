use std::sync::{Arc, Mutex};

use super::proactive::ProactiveSuggestion;
use super::trust_tuner::{FeedbackKind, TrustTuner};

/// How to deliver a suggestion to the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryDecision {
    /// Surface immediately (push notification / overlay).
    Push,
    /// Store silently for user to browse later.
    Inbox,
    /// Discard — not worth showing.
    Drop,
}

/// Coordinates suggestion delivery and feedback recording.
pub struct DeliveryManager {
    trust: Arc<Mutex<TrustTuner>>,
    push_enabled: bool,
}

impl DeliveryManager {
    pub fn new(trust: Arc<Mutex<TrustTuner>>, push_enabled: bool) -> Self {
        Self {
            trust,
            push_enabled,
        }
    }

    /// Decide how to deliver a suggestion based on current trust parameters.
    pub fn deliver(&self, suggestion: &ProactiveSuggestion) -> DeliveryDecision {
        let type_str = suggestion.suggestion_type.as_str();

        let (push_threshold, inbox_threshold) = {
            if let Ok(tuner) = self.trust.lock() {
                let push = tuner.push_threshold_for(&type_str);
                let inbox = tuner.params().inbox_threshold;
                (push, inbox)
            } else {
                (0.82, 0.55)
            }
        };

        if !self.push_enabled {
            if suggestion.confidence >= inbox_threshold {
                return DeliveryDecision::Inbox;
            }
            return DeliveryDecision::Drop;
        }

        if suggestion.confidence >= push_threshold {
            DeliveryDecision::Push
        } else if suggestion.confidence >= inbox_threshold {
            DeliveryDecision::Inbox
        } else {
            DeliveryDecision::Drop
        }
    }

    /// Record user feedback, updating trust tuner parameters.
    pub fn record_feedback(&self, kind: FeedbackKind, suggestion_type: &str) {
        if let Ok(mut tuner) = self.trust.lock() {
            tuner.apply_feedback(kind, suggestion_type);
        }
    }

    pub fn push_enabled(&self) -> bool {
        self.push_enabled
    }

    pub fn set_push_enabled(&mut self, enabled: bool) {
        self.push_enabled = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intelligence::proactive::{SuggestionDisposition, SuggestionType};

    fn suggestion(confidence: f32) -> ProactiveSuggestion {
        ProactiveSuggestion {
            id: "s1".to_string(),
            suggestion_type: SuggestionType::Reminder,
            title: "t".to_string(),
            body: "b".to_string(),
            confidence,
            disposition: SuggestionDisposition::InboxOnly,
            created_at: 1,
            metadata: serde_json::Value::Object(Default::default()),
        }
    }

    fn manager(push_enabled: bool) -> DeliveryManager {
        DeliveryManager::new(Arc::new(Mutex::new(TrustTuner::new())), push_enabled)
    }

    #[test]
    fn high_confidence_pushes_when_push_enabled() {
        // Default push threshold is 0.82.
        assert_eq!(manager(true).deliver(&suggestion(0.9)), DeliveryDecision::Push);
    }

    #[test]
    fn mid_confidence_goes_to_inbox() {
        assert_eq!(manager(true).deliver(&suggestion(0.6)), DeliveryDecision::Inbox);
    }

    #[test]
    fn low_confidence_drops() {
        assert_eq!(manager(true).deliver(&suggestion(0.2)), DeliveryDecision::Drop);
    }

    #[test]
    fn push_disabled_never_pushes_even_at_high_confidence() {
        let m = manager(false);
        assert_eq!(m.deliver(&suggestion(0.99)), DeliveryDecision::Inbox);
        assert_eq!(m.deliver(&suggestion(0.1)), DeliveryDecision::Drop);
    }

    #[test]
    fn push_enabled_accessor_and_setter() {
        let mut m = manager(true);
        assert!(m.push_enabled());
        m.set_push_enabled(false);
        assert!(!m.push_enabled());
    }

    #[test]
    fn record_feedback_updates_trust_params() {
        let trust = Arc::new(Mutex::new(TrustTuner::new()));
        let m = DeliveryManager::new(Arc::clone(&trust), true);
        m.record_feedback(FeedbackKind::ThumbsUp, "reminder");
        // ThumbsUp lowers confidence threshold from the 0.82 default.
        assert!(trust.lock().unwrap().params().confidence_threshold < 0.82);
    }
}
