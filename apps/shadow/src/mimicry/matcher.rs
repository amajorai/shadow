use crate::mimicry::procedure::ProcedureStore;
use crate::mimicry::types::ProcedureTemplate;

/// Matches the current activity context against stored procedure templates.
pub struct ProcedureMatcher;

impl ProcedureMatcher {
    /// Find procedures that match the current context.
    ///
    /// Scores by: app name match (0.8), recent apps overlap (0.6),
    /// title/description keyword overlap (0.4).
    /// Returns up to 5 results sorted by score descending.
    pub fn match_context(
        app: &str,
        window_title: &str,
        recent_apps: &[String],
        store: &ProcedureStore,
    ) -> Vec<(ProcedureTemplate, f32)> {
        let procedures = match store.list() {
            Ok(p) => p,
            Err(_) => return vec![],
        };

        let app_lower = app.to_lowercase();
        let title_lower = window_title.to_lowercase();
        let title_words: Vec<&str> = title_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let mut scored: Vec<(ProcedureTemplate, f32)> = procedures
            .into_iter()
            .filter_map(|p| {
                let score = score_procedure(&p, &app_lower, &title_words, recent_apps);
                if score > 0.0 {
                    Some((p, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(5);
        scored
    }

    /// Format matched procedures as a hint block for LLM prompt injection.
    pub fn format_for_prompt(matches: &[(ProcedureTemplate, f32)]) -> String {
        if matches.is_empty() {
            return String::new();
        }
        let lines: Vec<String> = matches
            .iter()
            .map(|(p, score)| {
                format!(
                    "- '{}' (app={}, score={:.2}): {}",
                    p.name, p.app_name, score, p.description
                )
            })
            .collect();
        format!("Relevant known procedures:\n{}", lines.join("\n"))
    }
}

fn score_procedure(
    p: &ProcedureTemplate,
    app_lower: &str,
    title_words: &[&str],
    recent_apps: &[String],
) -> f32 {
    let mut score = 0.0f32;

    // App name exact match
    if !p.app_name.is_empty() && p.app_name.to_lowercase() == app_lower {
        score += 0.8;
    } else if recent_apps
        .iter()
        .any(|a| a.to_lowercase() == p.app_name.to_lowercase())
    {
        score += 0.6;
    }

    // Title keyword overlap with procedure name/description
    let proc_text = format!("{} {}", p.name.to_lowercase(), p.description.to_lowercase());
    let matching_words = title_words
        .iter()
        .filter(|w| proc_text.contains(*w))
        .count();
    if !title_words.is_empty() {
        score += 0.4 * (matching_words as f32 / title_words.len() as f32);
    }

    // Boost by historical success
    if p.success_count > 0 {
        score += 0.1 * (p.success_count as f32).min(5.0) / 5.0;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimicry::procedure::ProcedureStore;

    fn make(id: &str, name: &str, app: &str, desc: &str) -> ProcedureTemplate {
        ProcedureTemplate {
            id: id.to_string(),
            name: name.to_string(),
            app_name: app.to_string(),
            description: desc.to_string(),
            steps: vec![],
            preconditions: vec![],
            success_count: 0,
            failure_count: 0,
            last_used: 0,
            created_at: 0,
        }
    }

    fn temp_store() -> (ProcedureStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("shadow-match-{}", uuid::Uuid::new_v4()));
        let db = dir.join("procedures.db");
        (ProcedureStore::new(&db).expect("store"), dir)
    }

    #[test]
    fn score_procedure_exact_app_match() {
        let p = make("x", "Task", "Slack", "");
        let s = score_procedure(&p, "slack", &[], &[]);
        assert!((s - 0.8).abs() < 1e-5, "s={s}");
    }

    #[test]
    fn score_procedure_recent_apps_fallback() {
        let p = make("x", "Task", "Slack", "");
        let recent = vec!["Slack".to_string(), "Chrome".to_string()];
        // Current app differs, but Slack is in recent_apps → 0.6, not 0.8.
        let s = score_procedure(&p, "notepad", &[], &recent);
        assert!((s - 0.6).abs() < 1e-5, "s={s}");
    }

    #[test]
    fn score_procedure_title_keyword_overlap() {
        // No app match; title words (>3 chars) overlap procedure text.
        let p = make("x", "Compose", "Mail", "write email");
        let title_words = ["compose", "message"]; // only "compose" is in proc text
        let s = score_procedure(&p, "other", &title_words, &[]);
        // 0.4 * 1/2.
        assert!((s - 0.2).abs() < 1e-5, "s={s}");
    }

    #[test]
    fn score_procedure_empty_app_name_does_not_match_empty_app() {
        // Guard: a procedure with empty app_name must not score on an empty app arg.
        let p = make("x", "Task", "", "");
        let s = score_procedure(&p, "", &[], &[]);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn match_context_sorts_and_filters_zero_scores() {
        let (store, dir) = temp_store();
        store
            .save(&make("a", "Compose", "Mail", "write email"))
            .unwrap();
        store
            .save(&make("b", "Resize", "Photos", "crop image"))
            .unwrap();

        let matches = ProcedureMatcher::match_context("Mail", "compose window", &[], &store);
        // Only the Mail procedure scores (app match + maybe title); Photos scores 0.
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0.id, "a");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn match_context_caps_results_at_five() {
        let (store, dir) = temp_store();
        for i in 0..8 {
            store
                .save(&make(&format!("p{i}"), "Task", "Mail", "d"))
                .unwrap();
        }
        let matches = ProcedureMatcher::match_context("Mail", "", &[], &store);
        assert_eq!(matches.len(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn format_for_prompt_empty_when_no_matches() {
        assert_eq!(ProcedureMatcher::format_for_prompt(&[]), "");
    }

    #[test]
    fn format_for_prompt_renders_name_app_and_score() {
        let matches = vec![(make("x", "Compose", "Mail", "write email"), 0.8)];
        let out = ProcedureMatcher::format_for_prompt(&matches);
        assert!(out.contains("Relevant known procedures"));
        assert!(out.contains("Compose"));
        assert!(out.contains("app=Mail"));
        assert!(out.contains("0.80"));
    }
}
