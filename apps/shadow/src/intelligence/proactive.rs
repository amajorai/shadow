use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

use super::context::{ContextSynthesizer, EpisodeStore};
use super::trust_tuner::TrustTuner;
use crate::llm::{orchestrator::LlmOrchestrator, LlmMessage, LlmRequest};
use crate::utils::wall_micros;

const FAST_TICK_INTERVAL: Duration = Duration::from_secs(10 * 60); // 10 min
const FAST_TICKS_PER_DEEP: u32 = 3; // deep every 3rd fast tick = 30 min
const EPISODE_COOLDOWN_SECS: u64 = 3 * 60; // 3 min between episode synths
const DAILY_COOLDOWN_SECS: u64 = 60 * 60; // 1 hr between daily synths
const BACKOFF_BASE_SECS: u64 = 30;
const MAX_BACKOFF_SECS: u64 = 30 * 60; // max 30 min backoff

/// Types of proactive suggestions Shadow can generate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionType {
    Followup,
    MeetingPrep,
    WorkloadPattern,
    Reminder,
    ContextSwitch,
    DailyDigest,
}

impl SuggestionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Followup => "followup",
            Self::MeetingPrep => "meetingprep",
            Self::WorkloadPattern => "workloadpattern",
            Self::Reminder => "reminder",
            Self::ContextSwitch => "contextswitch",
            Self::DailyDigest => "dailydigest",
        }
    }
}

/// Disposition: how urgently to surface this suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionDisposition {
    PushNow,
    InboxOnly,
    Drop,
}

/// A single proactive suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveSuggestion {
    pub id: String,
    pub suggestion_type: SuggestionType,
    pub title: String,
    pub body: String,
    pub confidence: f32,
    pub disposition: SuggestionDisposition,
    pub created_at: u64,
    pub metadata: serde_json::Value,
}

/// Tick type for policy engine scoring.
#[derive(Debug, Clone, Copy)]
enum TickType {
    Fast,
    Deep,
}

/// Generates proactive suggestions from behavioral context.
pub struct ProactiveAnalyzer {
    orchestrator: Arc<LlmOrchestrator>,
    context: ContextSynthesizer,
}

impl ProactiveAnalyzer {
    pub fn new(
        orchestrator: Arc<LlmOrchestrator>,
        episode_store: Option<Arc<std::sync::Mutex<EpisodeStore>>>,
    ) -> Self {
        let context = match episode_store {
            Some(store) => ContextSynthesizer::with_store(store, Arc::clone(&orchestrator)),
            None => ContextSynthesizer::new(),
        };
        Self {
            orchestrator,
            context,
        }
    }

    /// Fast tick: analyzes recent episodes (last 30 min) for live-context suggestions.
    pub async fn analyze_fast(&self) -> Result<Vec<ProactiveSuggestion>> {
        let episodes = self.context.get_recent_episodes(20)?;
        if episodes.is_empty() {
            return Ok(vec![]);
        }

        let episode_summary = episodes
            .iter()
            .map(|e| {
                format!(
                    "- {} ({}-{}): {}",
                    e.app_name,
                    e.start_us / 1_000_000,
                    e.end_us / 1_000_000,
                    e.summary
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are Shadow, a personal AI assistant. Based on the user's recent activity, \
             generate 1-2 immediate, actionable suggestions.\n\n\
             Recent episodes:\n{}\n\n\
             Respond with a JSON array of suggestions, each with: \
             type (followup|meeting_prep|workload_pattern|reminder|context_switch|daily_digest), \
             title (short), body (1-2 sentences), confidence (0.0-1.0).\n\
             Example: [{{\"type\":\"followup\",\"title\":\"...\",\"body\":\"...\",\"confidence\":0.8}}]",
            episode_summary
        );

        self.run_llm_analysis(prompt, 400).await
    }

    /// Deep tick: analyzes full episode history and daily patterns for strategic suggestions.
    pub async fn analyze_deep(&self) -> Result<Vec<ProactiveSuggestion>> {
        // Refresh episode store with LLM summaries for the past 2 hours
        let now = wall_micros();
        let two_hours_ago = now.saturating_sub(2 * 60 * 60 * 1_000_000);
        let _ = self
            .context
            .refresh_and_store_episodes(two_hours_ago, now)
            .await;

        let episodes = self.context.get_recent_episodes(50)?;
        if episodes.is_empty() {
            return Ok(vec![]);
        }

        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        let episode_summary = episodes
            .iter()
            .map(|e| format!("- {}: {}", e.app_name, e.summary))
            .collect::<Vec<_>>()
            .join("\n");

        let daily_context = match self.context.build_daily_summary(&today) {
            Ok(daily) => {
                let top_apps = daily
                    .top_apps
                    .iter()
                    .take(5)
                    .map(|a| format!("{}: {}min", a.app_name, a.duration_ms / 60_000))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Top apps today: {}", top_apps)
            }
            Err(_) => String::new(),
        };

        let prompt = format!(
            "You are Shadow, a personal AI assistant. Based on the user's full activity \
             history today, generate 1-3 strategic, pattern-based suggestions.\n\n\
             Episode history:\n{}\n\n{}\n\n\
             Respond with a JSON array of suggestions, each with: \
             type (followup|meeting_prep|workload_pattern|reminder|context_switch|daily_digest), \
             title (short), body (1-2 sentences), confidence (0.0-1.0).\n\
             Example: [{{\"type\":\"workload_pattern\",\"title\":\"...\",\"body\":\"...\",\"confidence\":0.7}}]",
            episode_summary, daily_context
        );

        self.run_llm_analysis(prompt, 600).await
    }

    async fn run_llm_analysis(
        &self,
        prompt: String,
        max_tokens: u32,
    ) -> Result<Vec<ProactiveSuggestion>> {
        let response = self
            .orchestrator
            .generate(LlmRequest {
                messages: vec![LlmMessage::user(prompt)],
                temperature: 0.4,
                max_tokens,
                ..Default::default()
            })
            .await?;

        let content = response.content.unwrap_or_default();
        Ok(parse_suggestions(&content))
    }
}

/// Multi-factor policy engine: assigns final disposition to a suggestion.
fn score_suggestion(
    s: &ProactiveSuggestion,
    tick_type: TickType,
    trust: Option<&TrustTuner>,
) -> SuggestionDisposition {
    // Interrupt cost by suggestion type (higher = more disruptive)
    let interrupt_cost: f32 = match s.suggestion_type {
        SuggestionType::Reminder | SuggestionType::MeetingPrep => 0.15,
        SuggestionType::Followup | SuggestionType::WorkloadPattern => 0.25,
        SuggestionType::ContextSwitch => 0.45,
        SuggestionType::DailyDigest => 0.10,
    };

    let effective_score = s.confidence - interrupt_cost * 0.5;

    let type_str = s.suggestion_type.as_str();

    // Use TrustTuner params if available, else fall back to hardcoded defaults
    let (push_threshold, inbox_threshold) = if let Some(tuner) = trust {
        let push_base = match tick_type {
            TickType::Fast => tuner.push_threshold_for(&type_str),
            TickType::Deep => (tuner.push_threshold_for(&type_str) - 0.17).max(0.40),
        };
        (push_base, tuner.params().inbox_threshold)
    } else {
        let push = match tick_type {
            TickType::Fast => 0.72,
            TickType::Deep => 0.55,
        };
        (push, 0.30)
    };

    if effective_score >= push_threshold {
        SuggestionDisposition::PushNow
    } else if effective_score >= inbox_threshold {
        SuggestionDisposition::InboxOnly
    } else {
        SuggestionDisposition::Drop
    }
}

/// Apply policy engine to a list of raw suggestions, filtering out Drop entries.
fn apply_policy(
    suggestions: Vec<ProactiveSuggestion>,
    tick_type: TickType,
    trust: Option<&TrustTuner>,
) -> Vec<ProactiveSuggestion> {
    suggestions
        .into_iter()
        .filter_map(|mut s| {
            let disp = score_suggestion(&s, tick_type, trust);
            if matches!(disp, SuggestionDisposition::Drop) {
                None
            } else {
                s.disposition = disp;
                Some(s)
            }
        })
        .collect()
}

fn parse_suggestions(content: &str) -> Vec<ProactiveSuggestion> {
    let json_str = extract_json_array(content).unwrap_or_default();
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&json_str).unwrap_or_default();

    let now = wall_micros();
    parsed
        .into_iter()
        .filter_map(|v| {
            let suggestion_type = match v["type"].as_str()? {
                "followup" => SuggestionType::Followup,
                "meeting_prep" => SuggestionType::MeetingPrep,
                "workload_pattern" => SuggestionType::WorkloadPattern,
                "reminder" => SuggestionType::Reminder,
                "context_switch" => SuggestionType::ContextSwitch,
                "daily_digest" => SuggestionType::DailyDigest,
                _ => SuggestionType::Followup,
            };
            let confidence = v["confidence"].as_f64().unwrap_or(0.5) as f32;

            Some(ProactiveSuggestion {
                id: uuid::Uuid::new_v4().to_string(),
                suggestion_type,
                title: v["title"].as_str()?.to_string(),
                body: v["body"].as_str()?.to_string(),
                confidence,
                disposition: SuggestionDisposition::InboxOnly, // overwritten by policy engine
                created_at: now,
                metadata: serde_json::Value::Object(Default::default()),
            })
        })
        .collect()
}

fn extract_json_array(s: &str) -> Option<String> {
    let start = s.find('[')?;
    let end = s.rfind(']')?;
    if end > start {
        Some(s[start..=end].to_string())
    } else {
        None
    }
}

/// Stores proactive suggestions in SQLite.
pub struct ProactiveStore {
    conn: rusqlite::Connection,
}

impl ProactiveStore {
    pub fn new(db_path: &std::path::Path) -> Result<Self> {
        let conn = rusqlite::Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS proactive_suggestions (
                id TEXT PRIMARY KEY,
                suggestion_type TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                confidence REAL NOT NULL,
                disposition TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );",
        )?;
        Ok(Self { conn })
    }

    pub fn store(&self, suggestion: &ProactiveSuggestion) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO proactive_suggestions \
             (id, suggestion_type, title, body, confidence, disposition, created_at, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                suggestion.id,
                format!("{:?}", suggestion.suggestion_type),
                suggestion.title,
                suggestion.body,
                suggestion.confidence as f64,
                format!("{:?}", suggestion.disposition),
                suggestion.created_at as i64,
                serde_json::to_string(&suggestion.metadata).unwrap_or_default()
            ],
        )?;
        Ok(())
    }

    pub fn list_recent(&self, limit: usize) -> Result<Vec<ProactiveSuggestion>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, suggestion_type, title, body, confidence, disposition, created_at, metadata \
             FROM proactive_suggestions ORDER BY created_at DESC LIMIT ?1",
        )?;

        let rows = stmt.query_map([limit as i64], |row| {
            Ok(ProactiveSuggestion {
                id: row.get(0)?,
                suggestion_type: parse_suggestion_type(row.get::<_, String>(1)?.as_str()),
                title: row.get(2)?,
                body: row.get(3)?,
                confidence: row.get::<_, f64>(4)? as f32,
                disposition: parse_disposition(row.get::<_, String>(5)?.as_str()),
                created_at: row.get::<_, i64>(6)? as u64,
                metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn get(&self, id: &str) -> Result<Option<ProactiveSuggestion>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, suggestion_type, title, body, confidence, disposition, created_at, metadata \
             FROM proactive_suggestions WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map([id], |row| {
            Ok(ProactiveSuggestion {
                id: row.get(0)?,
                suggestion_type: parse_suggestion_type(row.get::<_, String>(1)?.as_str()),
                title: row.get(2)?,
                body: row.get(3)?,
                confidence: row.get::<_, f64>(4)? as f32,
                disposition: parse_disposition(row.get::<_, String>(5)?.as_str()),
                created_at: row.get::<_, i64>(6)? as u64,
                metadata: serde_json::from_str(&row.get::<_, String>(7)?)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
            })
        })?;
        Ok(rows.next().and_then(|r| r.ok()))
    }

    /// Return titles of suggestions created within the last `within_secs` seconds.
    pub fn recent_titles(&self, within_secs: u64) -> Vec<String> {
        let cutoff = (wall_micros().saturating_sub(within_secs * 1_000_000)) as i64;
        self.conn
            .prepare("SELECT title FROM proactive_suggestions WHERE created_at >= ?1")
            .ok()
            .and_then(|mut stmt| {
                stmt.query_map([cutoff], |row| row.get::<_, String>(0))
                    .ok()
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
            })
            .unwrap_or_default()
    }
}

fn parse_suggestion_type(s: &str) -> SuggestionType {
    match s {
        s if s.contains("MeetingPrep") => SuggestionType::MeetingPrep,
        s if s.contains("WorkloadPattern") => SuggestionType::WorkloadPattern,
        s if s.contains("Reminder") => SuggestionType::Reminder,
        s if s.contains("ContextSwitch") => SuggestionType::ContextSwitch,
        s if s.contains("DailyDigest") => SuggestionType::DailyDigest,
        _ => SuggestionType::Followup,
    }
}

fn parse_disposition(s: &str) -> SuggestionDisposition {
    match s {
        s if s.contains("PushNow") => SuggestionDisposition::PushNow,
        s if s.contains("Drop") => SuggestionDisposition::Drop,
        _ => SuggestionDisposition::InboxOnly,
    }
}

/// Tracks cooldown timestamps and date for rollover detection.
struct CooldownGuard {
    last_episode_synth_us: u64,
    last_daily_synth_us: u64,
    last_date: String,
}

impl CooldownGuard {
    fn new() -> Self {
        Self {
            last_episode_synth_us: 0,
            last_daily_synth_us: 0,
            last_date: String::new(),
        }
    }

    fn can_episode_synth(&self, now_us: u64) -> bool {
        (now_us.saturating_sub(self.last_episode_synth_us)) / 1_000_000 >= EPISODE_COOLDOWN_SECS
    }

    fn can_daily_synth(&self, now_us: u64) -> bool {
        (now_us.saturating_sub(self.last_daily_synth_us)) / 1_000_000 >= DAILY_COOLDOWN_SECS
    }

    /// Returns true if the calendar date has rolled over since last check.
    fn check_date_rollover(&mut self, today: &str) -> bool {
        if self.last_date.is_empty() {
            self.last_date = today.to_string();
            return false;
        }
        if self.last_date != today {
            self.last_date = today.to_string();
            return true;
        }
        false
    }
}

/// Background task that runs proactive analysis on a dual-tick (fast + deep) schedule.
pub async fn run_proactive_heartbeat(
    orchestrator: Arc<LlmOrchestrator>,
    store: Arc<Mutex<ProactiveStore>>,
    episode_store: Option<Arc<std::sync::Mutex<EpisodeStore>>>,
    trust: Option<Arc<std::sync::Mutex<TrustTuner>>>,
) {
    let analyzer = ProactiveAnalyzer::new(Arc::clone(&orchestrator), episode_store);
    let mut cooldowns = CooldownGuard::new();
    let mut backoff_secs = 0u64;
    let mut fast_tick_count = 0u32;

    loop {
        let sleep = FAST_TICK_INTERVAL + Duration::from_secs(backoff_secs);
        tokio::time::sleep(sleep).await;

        let now = wall_micros();
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        let date_rolled = cooldowns.check_date_rollover(&today);

        fast_tick_count += 1;
        let run_deep = fast_tick_count >= FAST_TICKS_PER_DEEP || date_rolled;
        if run_deep {
            fast_tick_count = 0;
        }

        // ── Fast tick: live context (episode synth cooldown applies) ────────────
        if cooldowns.can_episode_synth(now) {
            tracing::debug!("Proactive: fast tick");
            match analyzer.analyze_fast().await {
                Ok(raw) => {
                    let suggestions = {
                        let trust_ref = trust.as_ref().and_then(|t| t.lock().ok());
                        apply_policy(raw, TickType::Fast, trust_ref.as_deref())
                    };
                    let s = store.lock().await;
                    for suggestion in &suggestions {
                        if let Err(e) = s.store(suggestion) {
                            tracing::warn!("Failed to store proactive suggestion: {}", e);
                        }
                    }
                    if !suggestions.is_empty() {
                        tracing::info!("Proactive fast tick: {} suggestions", suggestions.len());
                    }
                    cooldowns.last_episode_synth_us = now;
                    backoff_secs = 0;
                }
                Err(e) => {
                    tracing::debug!("Proactive fast tick skipped: {}", e);
                    backoff_secs = (backoff_secs * 2 + BACKOFF_BASE_SECS).min(MAX_BACKOFF_SECS);
                }
            }
        }

        // ── Deep tick: full history + patterns (daily cooldown applies) ─────────
        if run_deep && (cooldowns.can_daily_synth(now) || date_rolled) {
            tracing::debug!("Proactive: deep tick");
            match analyzer.analyze_deep().await {
                Ok(raw) => {
                    let suggestions = {
                        let trust_ref = trust.as_ref().and_then(|t| t.lock().ok());
                        apply_policy(raw, TickType::Deep, trust_ref.as_deref())
                    };
                    let s = store.lock().await;
                    for suggestion in &suggestions {
                        if let Err(e) = s.store(suggestion) {
                            tracing::warn!("Failed to store proactive suggestion: {}", e);
                        }
                    }
                    if !suggestions.is_empty() {
                        tracing::info!("Proactive deep tick: {} suggestions", suggestions.len());
                    }
                    cooldowns.last_daily_synth_us = now;
                }
                Err(e) => tracing::debug!("Proactive deep tick skipped: {}", e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suggestion(t: SuggestionType, confidence: f32) -> ProactiveSuggestion {
        ProactiveSuggestion {
            id: uuid::Uuid::new_v4().to_string(),
            suggestion_type: t,
            title: "title".to_string(),
            body: "body".to_string(),
            confidence,
            disposition: SuggestionDisposition::InboxOnly,
            created_at: 1,
            metadata: serde_json::Value::Object(Default::default()),
        }
    }

    #[test]
    fn suggestion_type_as_str_covers_all_variants() {
        assert_eq!(SuggestionType::Followup.as_str(), "followup");
        assert_eq!(SuggestionType::MeetingPrep.as_str(), "meetingprep");
        assert_eq!(SuggestionType::WorkloadPattern.as_str(), "workloadpattern");
        assert_eq!(SuggestionType::Reminder.as_str(), "reminder");
        assert_eq!(SuggestionType::ContextSwitch.as_str(), "contextswitch");
        assert_eq!(SuggestionType::DailyDigest.as_str(), "dailydigest");
    }

    #[test]
    fn score_suggestion_high_confidence_pushes_on_fast_tick() {
        // DailyDigest interrupt cost 0.10 → effective 0.95 - 0.05 = 0.90 >= 0.72.
        let s = suggestion(SuggestionType::DailyDigest, 0.95);
        assert!(matches!(
            score_suggestion(&s, TickType::Fast, None),
            SuggestionDisposition::PushNow
        ));
    }

    #[test]
    fn score_suggestion_mid_confidence_is_inbox() {
        // Followup cost 0.25 → effective 0.6 - 0.125 = 0.475; below 0.72 push, above 0.30 inbox.
        let s = suggestion(SuggestionType::Followup, 0.6);
        assert!(matches!(
            score_suggestion(&s, TickType::Fast, None),
            SuggestionDisposition::InboxOnly
        ));
    }

    #[test]
    fn score_suggestion_low_confidence_drops() {
        let s = suggestion(SuggestionType::ContextSwitch, 0.2);
        assert!(matches!(
            score_suggestion(&s, TickType::Fast, None),
            SuggestionDisposition::Drop
        ));
    }

    #[test]
    fn score_suggestion_deep_tick_has_lower_push_threshold() {
        // Followup 0.6 → effective 0.475. Fast push threshold 0.72 → Inbox.
        // Deep push threshold 0.55 → still Inbox (0.475 < 0.55). Use a value between.
        let s = suggestion(SuggestionType::DailyDigest, 0.62);
        // effective = 0.62 - 0.05 = 0.57. Fast(0.72) → Inbox; Deep(0.55) → Push.
        assert!(matches!(
            score_suggestion(&s, TickType::Fast, None),
            SuggestionDisposition::InboxOnly
        ));
        assert!(matches!(
            score_suggestion(&s, TickType::Deep, None),
            SuggestionDisposition::PushNow
        ));
    }

    #[test]
    fn apply_policy_filters_dropped_and_sets_disposition() {
        let raw = vec![
            suggestion(SuggestionType::DailyDigest, 0.95), // push
            suggestion(SuggestionType::ContextSwitch, 0.1), // drop
        ];
        let kept = apply_policy(raw, TickType::Fast, None);
        assert_eq!(kept.len(), 1);
        assert!(matches!(kept[0].disposition, SuggestionDisposition::PushNow));
    }

    #[test]
    fn parse_suggestions_maps_types_and_defaults() {
        let content = "prefix [\
            {\"type\":\"meeting_prep\",\"title\":\"Prep\",\"body\":\"b\",\"confidence\":0.8},\
            {\"type\":\"unknown_kind\",\"title\":\"X\",\"body\":\"y\"}\
        ] suffix";
        let out = parse_suggestions(content);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0].suggestion_type, SuggestionType::MeetingPrep));
        assert!((out[0].confidence - 0.8).abs() < 1e-5);
        // Unknown type falls back to Followup, missing confidence defaults 0.5.
        assert!(matches!(out[1].suggestion_type, SuggestionType::Followup));
        assert!((out[1].confidence - 0.5).abs() < 1e-5);
    }

    #[test]
    fn parse_suggestions_skips_entries_missing_required_fields() {
        // Missing "body" → dropped.
        let content = "[{\"type\":\"followup\",\"title\":\"only title\"}]";
        assert!(parse_suggestions(content).is_empty());
    }

    #[test]
    fn parse_suggestions_empty_on_garbage() {
        assert!(parse_suggestions("no array here").is_empty());
    }

    #[test]
    fn extract_json_array_grabs_bracketed_span() {
        assert_eq!(extract_json_array("x [1,2] y").unwrap(), "[1,2]");
        assert_eq!(extract_json_array("none"), None);
    }

    #[test]
    fn parse_suggestion_type_reads_debug_strings() {
        assert!(matches!(parse_suggestion_type("MeetingPrep"), SuggestionType::MeetingPrep));
        assert!(matches!(parse_suggestion_type("WorkloadPattern"), SuggestionType::WorkloadPattern));
        assert!(matches!(parse_suggestion_type("Reminder"), SuggestionType::Reminder));
        assert!(matches!(parse_suggestion_type("ContextSwitch"), SuggestionType::ContextSwitch));
        assert!(matches!(parse_suggestion_type("DailyDigest"), SuggestionType::DailyDigest));
        assert!(matches!(parse_suggestion_type("Followup"), SuggestionType::Followup));
    }

    #[test]
    fn parse_disposition_reads_debug_strings() {
        assert!(matches!(parse_disposition("PushNow"), SuggestionDisposition::PushNow));
        assert!(matches!(parse_disposition("Drop"), SuggestionDisposition::Drop));
        assert!(matches!(parse_disposition("InboxOnly"), SuggestionDisposition::InboxOnly));
    }

    #[test]
    fn cooldown_guard_gates_by_interval() {
        let g = CooldownGuard::new();
        // At time 0 the cooldown has elapsed (0 - 0 >= threshold trivially, since last=0).
        assert!(g.can_episode_synth(EPISODE_COOLDOWN_SECS * 1_000_000));
        assert!(!g.can_episode_synth(1)); // 1us since epoch < 3min cooldown
        assert!(g.can_daily_synth(DAILY_COOLDOWN_SECS * 1_000_000));
        assert!(!g.can_daily_synth(1));
    }

    #[test]
    fn cooldown_guard_detects_date_rollover() {
        let mut g = CooldownGuard::new();
        // First call seeds the date and reports no rollover.
        assert!(!g.check_date_rollover("2026-07-23"));
        assert!(!g.check_date_rollover("2026-07-23"));
        // A new date reports rollover once.
        assert!(g.check_date_rollover("2026-07-24"));
        assert!(!g.check_date_rollover("2026-07-24"));
    }

    #[test]
    fn proactive_store_round_trips_suggestion() {
        let path = std::env::temp_dir().join(format!("shadow-proactive-{}", uuid::Uuid::new_v4()));
        let store = ProactiveStore::new(&path).unwrap();
        let mut s = suggestion(SuggestionType::Reminder, 0.7);
        s.disposition = SuggestionDisposition::PushNow;
        store.store(&s).unwrap();

        let got = store.get(&s.id).unwrap().expect("stored suggestion");
        assert_eq!(got.title, "title");
        assert!(matches!(got.suggestion_type, SuggestionType::Reminder));
        assert!(matches!(got.disposition, SuggestionDisposition::PushNow));

        let recent = store.list_recent(10).unwrap();
        assert_eq!(recent.len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn proactive_store_recent_titles_respects_window() {
        let path = std::env::temp_dir().join(format!("shadow-proactive-{}", uuid::Uuid::new_v4()));
        let store = ProactiveStore::new(&path).unwrap();
        let mut fresh = suggestion(SuggestionType::Reminder, 0.7);
        fresh.title = "fresh".to_string();
        fresh.created_at = wall_micros();
        store.store(&fresh).unwrap();

        let titles = store.recent_titles(3600);
        assert!(titles.contains(&"fresh".to_string()));

        // An old suggestion outside the window is excluded.
        let mut old = suggestion(SuggestionType::Reminder, 0.7);
        old.title = "ancient".to_string();
        old.created_at = 1; // ~epoch
        store.store(&old).unwrap();
        let recent = store.recent_titles(3600);
        assert!(recent.contains(&"fresh".to_string()));
        assert!(!recent.contains(&"ancient".to_string()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn proactive_store_get_missing_returns_none() {
        let path = std::env::temp_dir().join(format!("shadow-proactive-{}", uuid::Uuid::new_v4()));
        let store = ProactiveStore::new(&path).unwrap();
        assert!(store.get("nope").unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }
}
