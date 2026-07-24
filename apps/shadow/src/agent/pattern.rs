use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::llm::{orchestrator::LlmOrchestrator, LlmMessage, LlmRequest};
use crate::utils::{extract_json, wall_micros};

// ── Types ────────────────────────────────────────────────────────────────────

/// A single generalized step inside an AgentPattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStep {
    pub tool_name: String,
    pub purpose: String,
    pub key_arguments: Vec<String>,
    pub expected_outcome: String,
}

/// A generalized agent interaction pattern extracted from successful runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPattern {
    pub id: String,
    pub task_description: String,
    pub target_app: Option<String>,
    pub url_pattern: Option<String>,
    pub steps: Vec<PatternStep>,
    pub notes: String,
    pub success_count: u32,
    pub failure_count: u32,
    pub created_at: u64,
    pub last_used: u64,
}

// ── PatternStore ─────────────────────────────────────────────────────────────

/// Persists patterns as JSON files under `~/.shadow/data/patterns/`.
pub struct PatternStore {
    dir: PathBuf,
    cache: Option<Vec<AgentPattern>>,
}

impl PatternStore {
    pub fn new(dir: &Path) -> Self {
        std::fs::create_dir_all(dir).ok();
        Self {
            dir: dir.to_path_buf(),
            cache: None,
        }
    }

    /// Save a pattern to disk, overwriting if the ID already exists.
    pub fn save(&mut self, pattern: &AgentPattern) {
        let path = self.dir.join(format!("{}.json", pattern.id));
        if let Ok(json) = serde_json::to_string_pretty(pattern) {
            let _ = std::fs::write(path, json);
        }
        self.cache = None; // invalidate cache
    }

    /// Load all patterns from disk (lazy-cached).
    pub fn load_all(&mut self) -> &[AgentPattern] {
        if self.cache.is_none() {
            let mut patterns = Vec::new();
            if let Ok(dir) = std::fs::read_dir(&self.dir) {
                for entry in dir.flatten() {
                    if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                        if let Ok(text) = std::fs::read_to_string(entry.path()) {
                            if let Ok(p) = serde_json::from_str::<AgentPattern>(&text) {
                                // Skip archived patterns (failure_count > success_count*2 + 2)
                                if p.failure_count <= p.success_count * 2 + 2 {
                                    patterns.push(p);
                                }
                            }
                        }
                    }
                }
            }
            self.cache = Some(patterns);
        }
        self.cache.as_deref().unwrap_or(&[])
    }

    /// Find patterns relevant to the current context.
    /// Scores by keyword overlap, app match, and recency.
    pub fn find_relevant(
        &mut self,
        query: &str,
        app: &str,
        limit: usize,
    ) -> Vec<(AgentPattern, f32)> {
        let patterns = self.load_all().to_vec();
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();
        let app_lower = app.to_lowercase();
        let now = wall_micros();

        let mut scored: Vec<(AgentPattern, f32)> = patterns
            .into_iter()
            .filter_map(|p| {
                let score = score_pattern(&p, &query_words, &app_lower, now);
                if score > 0.0 {
                    Some((p, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    /// Increment success or failure count and persist.
    pub fn record_outcome(&mut self, id: &str, success: bool) {
        let path = self.dir.join(format!("{}.json", id));
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(mut p) = serde_json::from_str::<AgentPattern>(&text) {
                if success {
                    p.success_count += 1;
                } else {
                    p.failure_count += 1;
                }
                p.last_used = wall_micros();
                if let Ok(json) = serde_json::to_string_pretty(&p) {
                    let _ = std::fs::write(&path, json);
                }
                self.cache = None;
            }
        }
    }

    /// Format the top-N relevant patterns as a prompt injection block.
    pub fn format_for_prompt(&mut self, query: &str, app: &str) -> String {
        let relevant = self.find_relevant(query, app, 3);
        if relevant.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = relevant
            .iter()
            .map(|(p, _)| {
                let steps_summary: Vec<String> = p
                    .steps
                    .iter()
                    .map(|s| format!("{}: {}", s.tool_name, s.purpose))
                    .collect();
                format!(
                    "Pattern: {}\nApp: {}\nSteps: {}\nNotes: {}",
                    p.task_description,
                    p.target_app.as_deref().unwrap_or("any"),
                    steps_summary.join(" → "),
                    p.notes,
                )
            })
            .collect();
        format!("Relevant past patterns:\n{}", lines.join("\n---\n"))
    }
}

fn score_pattern(p: &AgentPattern, query_words: &[&str], app_lower: &str, now_us: u64) -> f32 {
    let mut score = 0.0f32;
    let desc_lower = p.task_description.to_lowercase();

    // Keyword overlap
    let matched = query_words
        .iter()
        .filter(|w| desc_lower.contains(**w))
        .count();
    if !query_words.is_empty() {
        score += 0.5 * matched as f32 / query_words.len() as f32;
    }

    // App match
    if let Some(app) = &p.target_app {
        if app.to_lowercase() == app_lower {
            score += 0.3;
        }
    }

    // Recency bonus: +0.1 if used within last 24 hours
    const ONE_DAY_US: u64 = 24 * 60 * 60 * 1_000_000;
    if p.last_used > 0 && now_us.saturating_sub(p.last_used) < ONE_DAY_US {
        score += 0.1;
    }

    // Success weighting
    if p.success_count > 0 {
        score *= 1.0 + 0.1 * (p.success_count as f32).min(5.0) / 5.0;
    }

    score
}

// ── PatternExtractor ─────────────────────────────────────────────────────────

/// Extracts a generalized `AgentPattern` from a completed agent run.
pub struct PatternExtractor;

impl PatternExtractor {
    /// Extract a pattern from the run description and tool sequence.
    /// Returns `None` if the run is ineligible (< 3 AX tool calls).
    pub async fn extract(
        run_description: &str,
        tools_used: &[String],
        orchestrator: &LlmOrchestrator,
    ) -> Option<AgentPattern> {
        const MIN_AX_CALLS: usize = 3;

        let ax_count = tools_used.iter().filter(|t| t.starts_with("ax_")).count();

        if ax_count < MIN_AX_CALLS {
            return None;
        }

        let tool_sequence = tools_used.join(", ");

        let prompt = format!(
            "Generalize this agent run into a reusable pattern.\n\
             Replace specific values (names, queries, paths) with {{{{PLACEHOLDER}}}} parameters.\n\
             Respond with JSON only:\n\
             {{\"task_description\":\"...\",\"target_app\":\"...\",\"steps\":[\
             {{\"tool_name\":\"...\",\"purpose\":\"...\",\"key_arguments\":[\"...\"],\
             \"expected_outcome\":\"...\"}}],\"notes\":\"...\"}}\n\n\
             Run description: {}\nTools used: {}",
            run_description, tool_sequence
        );

        let resp = orchestrator
            .generate(LlmRequest {
                messages: vec![LlmMessage::user(prompt)],
                temperature: 0.2,
                max_tokens: 512,
                ..Default::default()
            })
            .await
            .ok()?;

        let text = resp.content?;
        let json_str = extract_json(&text)?;
        let v: serde_json::Value = serde_json::from_str(&json_str).ok()?;

        let steps: Vec<PatternStep> = v["steps"]
            .as_array()?
            .iter()
            .filter_map(|s| {
                Some(PatternStep {
                    tool_name: s["tool_name"].as_str()?.to_string(),
                    purpose: s["purpose"].as_str().unwrap_or("").to_string(),
                    key_arguments: s["key_arguments"]
                        .as_array()
                        .unwrap_or(&vec![])
                        .iter()
                        .filter_map(|a| a.as_str().map(str::to_string))
                        .collect(),
                    expected_outcome: s["expected_outcome"].as_str().unwrap_or("").to_string(),
                })
            })
            .collect();

        if steps.is_empty() {
            return None;
        }

        let now = wall_micros();
        Some(AgentPattern {
            id: uuid::Uuid::new_v4().to_string(),
            task_description: v["task_description"].as_str()?.to_string(),
            target_app: v["target_app"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            url_pattern: None,
            steps,
            notes: v["notes"].as_str().unwrap_or("").to_string(),
            success_count: 1,
            failure_count: 0,
            created_at: now,
            last_used: now,
        })
    }

    /// Heuristic fallback: build a pattern directly from the tool sequence.
    pub fn extract_heuristic(run_description: &str, tools_used: &[String]) -> Option<AgentPattern> {
        if tools_used.len() < 3 {
            return None;
        }
        let steps = tools_used
            .iter()
            .map(|t| PatternStep {
                tool_name: t.clone(),
                purpose: format!("Step: {}", t),
                key_arguments: vec![],
                expected_outcome: String::new(),
            })
            .collect();
        let now = wall_micros();
        Some(AgentPattern {
            id: uuid::Uuid::new_v4().to_string(),
            task_description: run_description.to_string(),
            target_app: None,
            url_pattern: None,
            steps,
            notes: String::new(),
            success_count: 1,
            failure_count: 0,
            created_at: now,
            last_used: now,
        })
    }
}

// ── PatternMatcher ────────────────────────────────────────────────────────────

pub struct PatternMatcher;

impl PatternMatcher {
    /// Find relevant patterns and format them for prompt injection.
    pub fn find_and_format(query: &str, app: &str, store: &mut PatternStore) -> String {
        store.format_for_prompt(query, app)
    }

    /// Record run outcome against a pattern.
    pub fn record_outcome(id: &str, success: bool, store: &mut PatternStore) {
        store.record_outcome(id, success);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("shadow-pattern-{}", uuid::Uuid::new_v4()))
    }

    fn pattern(id: &str, desc: &str, app: Option<&str>) -> AgentPattern {
        AgentPattern {
            id: id.to_string(),
            task_description: desc.to_string(),
            target_app: app.map(str::to_string),
            url_pattern: None,
            steps: vec![PatternStep {
                tool_name: "ax_click".to_string(),
                purpose: "click send".to_string(),
                key_arguments: vec!["query".to_string()],
                expected_outcome: "sent".to_string(),
            }],
            notes: "n".to_string(),
            success_count: 0,
            failure_count: 0,
            created_at: 0,
            last_used: 0,
        }
    }

    #[test]
    fn save_then_load_all_round_trips() {
        let dir = temp_dir();
        let mut store = PatternStore::new(&dir);
        store.save(&pattern("p1", "send an email", Some("Mail")));
        let loaded = store.load_all();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, "p1");
        assert_eq!(loaded[0].target_app.as_deref(), Some("Mail"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_all_skips_archived_patterns() {
        let dir = temp_dir();
        let mut store = PatternStore::new(&dir);
        // Archived: failure_count (3) > success_count*2 + 2 (== 2).
        let mut archived = pattern("arch", "archived task", None);
        archived.failure_count = 3;
        store.save(&archived);
        // Kept: failure_count (2) <= success_count*2 + 2 (== 2).
        let mut kept = pattern("kept", "kept task", None);
        kept.failure_count = 2;
        store.save(&kept);

        let ids: Vec<&str> = store.load_all().iter().map(|p| p.id.as_str()).collect();
        assert!(ids.contains(&"kept"));
        assert!(!ids.contains(&"arch"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_relevant_excludes_zero_score_and_orders_by_score() {
        let dir = temp_dir();
        let mut store = PatternStore::new(&dir);
        store.save(&pattern("match", "send email report", Some("Mail")));
        store.save(&pattern("nomatch", "resize a photo", None));

        let results = store.find_relevant("send email", "Mail", 10);
        // Only the keyword-overlapping pattern is returned.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.id, "match");
        assert!(results[0].1 > 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_relevant_truncates_to_limit() {
        let dir = temp_dir();
        let mut store = PatternStore::new(&dir);
        for i in 0..5 {
            store.save(&pattern(&format!("p{i}"), "open the settings panel", None));
        }
        let results = store.find_relevant("open the settings", "", 2);
        assert_eq!(results.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn score_pattern_keyword_overlap_only() {
        // No app match, no recency, no success weighting → pure keyword fraction.
        let p = pattern("x", "send email", None);
        let query_words = ["send", "email", "report"];
        let score = score_pattern(&p, &query_words, "", 0);
        // 2 of 3 query words appear in "send email": 0.5 * 2/3.
        assert!((score - (0.5 * 2.0 / 3.0)).abs() < 1e-5, "score={score}");
    }

    #[test]
    fn score_pattern_adds_app_match_bonus() {
        let p = pattern("x", "send email", Some("Mail"));
        let words = ["send", "email"]; // full overlap → 0.5
        let with_app = score_pattern(&p, &words, "mail", 0);
        let without_app = score_pattern(&p, &words, "browser", 0);
        assert!((with_app - without_app - 0.3).abs() < 1e-5);
    }

    #[test]
    fn score_pattern_recency_bonus_within_a_day() {
        let mut p = pattern("x", "send email", None);
        let now = 10 * 24 * 60 * 60 * 1_000_000u64; // 10 days in micros
        p.last_used = now - 1_000_000; // 1 second ago
        let recent = score_pattern(&p, &["send", "email"], "", now);
        p.last_used = now - 2 * 24 * 60 * 60 * 1_000_000; // 2 days ago
        let stale = score_pattern(&p, &["send", "email"], "", now);
        assert!((recent - stale - 0.1).abs() < 1e-5);
    }

    #[test]
    fn record_outcome_increments_and_persists() {
        let dir = temp_dir();
        let mut store = PatternStore::new(&dir);
        store.save(&pattern("p1", "task", None));
        store.record_outcome("p1", true);
        store.record_outcome("p1", true);
        store.record_outcome("p1", false);

        let loaded = store.load_all();
        let p = loaded.iter().find(|p| p.id == "p1").expect("p1");
        assert_eq!(p.success_count, 2);
        assert_eq!(p.failure_count, 1);
        assert!(p.last_used > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_heuristic_needs_at_least_three_tools() {
        assert!(PatternExtractor::extract_heuristic("t", &["a".into(), "b".into()]).is_none());
        let p = PatternExtractor::extract_heuristic(
            "do the thing",
            &["ax_focus".into(), "ax_click".into(), "ax_type".into()],
        )
        .expect("pattern");
        assert_eq!(p.steps.len(), 3);
        assert_eq!(p.task_description, "do the thing");
        assert_eq!(p.success_count, 1);
        assert_eq!(p.failure_count, 0);
    }

    #[test]
    fn format_for_prompt_empty_when_no_matches() {
        let dir = temp_dir();
        let mut store = PatternStore::new(&dir);
        store.save(&pattern("p", "totally unrelated", None));
        assert_eq!(store.format_for_prompt("xyz nonsense query", ""), "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn format_for_prompt_includes_matched_description() {
        let dir = temp_dir();
        let mut store = PatternStore::new(&dir);
        store.save(&pattern("p", "compose a new message", Some("Mail")));
        let out = store.format_for_prompt("compose new message", "Mail");
        assert!(out.contains("Relevant past patterns"));
        assert!(out.contains("compose a new message"));
        assert!(out.contains("ax_click"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
