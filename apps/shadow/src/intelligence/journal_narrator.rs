//! Optional LLM narration pass over the deterministic journal cards.
//!
//! `shadow_core::build_journal_snapshot` produces cards with heuristic titles
//! ("Cursor: main.rs") and templated summaries. This module upgrades that text
//! into a Dayflow-style personal work journal — memory-trigger titles,
//! natural-language summaries, reconstruction-grade detail, and nested
//! distractions — WITHOUT re-segmenting: every card keeps its id, start/end,
//! category, and app. The numeric aggregates (focus, stats) are therefore
//! stable and only the standup text is refreshed downstream.
//!
//! It routes through Shadow's configured `LlmOrchestrator`, which Core points at
//! the governed Gateway (`SHADOW_LLM_BASE_URL`). Any failure — no LLM, bad JSON,
//! timeout — falls back to the deterministic cards, so narration is a pure
//! enhancement and never a hard dependency (batteries-included, fail-open).

use std::sync::Arc;

use serde::Deserialize;
use shadow_core::journal::{CardDistraction, JournalCard};

use crate::llm::{orchestrator::LlmOrchestrator, LlmMessage, LlmRequest};

/// Above this many cards one pass would overrun `MAX_TOKENS` mid-JSON — the
/// reply truncates, `parse_narrated` fails, and we silently fall back to
/// deterministic cards. Keep the cap comfortably inside the token budget
/// (~200 output tokens/card) so a full response fits and actually lands.
const MAX_NARRATED_CARDS: usize = 18;
/// Output-token ceiling for one narration pass. Sized for MAX_NARRATED_CARDS.
const MAX_TOKENS: u32 = 4096;

const SYSTEM_PROMPT: &str = r#"You are writing someone's personal work journal from raw computer-activity logs. Each entry is one time block ("card") that already has a FIXED start/end time and category — you are NOT re-segmenting, splitting, or merging cards. Your only job is to rewrite each card's text so that when this person scans their timeline tomorrow, each card makes them go "oh right, that."

Write as if you ARE the person jotting notes about their own day — not an analyst, not a status report.

For each card produce:
- title: a memory trigger, specific enough that it could only describe THIS situation. Roughly 5-15 words. Name the actual thing worked on in plain language. Never generic like "Working in Chrome".
- summary: one or two sentences on what actually happened in this block.
- detailed_summary: a reconstruction-grade recap — the specific files, pages, tabs, searches, or topics touched, in order.
- distractions: brief (<5 min) UNRELATED detours inside this block (a quick social-media or messaging check while working). Empty array if none. Do NOT count related sub-tasks — googling an error while debugging is part of debugging, not a distraction.

Hard rules:
- Keep each card's "id" EXACTLY as given. Return exactly one object per input card, in the same order.
- Do not invent activity that the raw log text does not support. If a card is thin, keep its text modest and factual.
- Return ONLY a JSON array. No prose, no explanation, no markdown fences."#;

/// One card as the model returns it. Only the text fields; id ties it back.
#[derive(Debug, Deserialize)]
struct NarratedCard {
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    detailed_summary: String,
    #[serde(default)]
    distractions: Vec<NarratedDistraction>,
}

#[derive(Debug, Deserialize)]
struct NarratedDistraction {
    #[serde(default)]
    title: String,
    #[serde(default)]
    summary: String,
}

/// Rewrite `cards` in place through the LLM. Returns the cards untouched on any
/// failure. `cards` keeps its ids/ranges/categories; only text is replaced.
pub async fn narrate_cards(
    orchestrator: &Arc<LlmOrchestrator>,
    cards: Vec<JournalCard>,
) -> Vec<JournalCard> {
    if cards.is_empty() || cards.len() > MAX_NARRATED_CARDS {
        if cards.len() > MAX_NARRATED_CARDS {
            tracing::info!(
                count = cards.len(),
                "journal narration skipped: too many cards for one pass"
            );
        }
        return cards;
    }

    let user_prompt = build_user_prompt(&cards);
    let request = LlmRequest {
        messages: vec![
            LlmMessage::system(SYSTEM_PROMPT),
            LlmMessage::user(user_prompt),
        ],
        temperature: 0.4,
        max_tokens: MAX_TOKENS,
        ..Default::default()
    };

    let response = match orchestrator.generate(request).await {
        Ok(response) => response,
        Err(err) => {
            tracing::warn!(error = %err, "journal narration failed; using deterministic cards");
            return cards;
        }
    };

    let Some(content) = response.content else {
        return cards;
    };

    match parse_narrated(&content) {
        Some(narrated) => merge(cards, narrated),
        None => {
            tracing::warn!(
                "journal narration returned unparseable output; using deterministic cards"
            );
            cards
        }
    }
}

/// Compact JSON of the cards the model needs to see: id + context + the raw
/// window-title trail (already assembled into `detailed_summary`).
fn build_user_prompt(cards: &[JournalCard]) -> String {
    let items: Vec<serde_json::Value> = cards
        .iter()
        .map(|card| {
            let minutes = card.end_ts.saturating_sub(card.start_ts) / 60_000_000;
            serde_json::json!({
                "id": card.id,
                "app": card.primary_app,
                "category": card.category,
                "minutes": minutes,
                "raw": card.detailed_summary,
            })
        })
        .collect();
    let cards_json = serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string());

    format!(
        "Cards:\n{cards_json}\n\nReturn a JSON array with exactly one object per card, same order and same ids, shaped:\n[{{\"id\":\"\",\"title\":\"\",\"summary\":\"\",\"detailed_summary\":\"\",\"distractions\":[{{\"title\":\"\",\"summary\":\"\"}}]}}]"
    )
}

/// Extract the JSON array from a possibly-fenced/prose-wrapped model reply.
fn parse_narrated(content: &str) -> Option<Vec<NarratedCard>> {
    let start = content.find('[')?;
    let end = content.rfind(']')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<Vec<NarratedCard>>(&content[start..=end]).ok()
}

/// Overlay narrated text onto the deterministic cards, matched by id. Cards the
/// model omitted keep their deterministic text. Blank fields are ignored so a
/// lazy model can't wipe a good heuristic title.
fn merge(mut cards: Vec<JournalCard>, narrated: Vec<NarratedCard>) -> Vec<JournalCard> {
    for item in narrated {
        let Some(card) = cards.iter_mut().find(|card| card.id == item.id) else {
            continue;
        };
        if !item.title.trim().is_empty() {
            card.title = item.title.trim().to_string();
        }
        if !item.summary.trim().is_empty() {
            card.summary = item.summary.trim().to_string();
        }
        if !item.detailed_summary.trim().is_empty() {
            card.detailed_summary = item.detailed_summary.trim().to_string();
        }
        card.distractions = item
            .distractions
            .into_iter()
            .filter(|d| !d.title.trim().is_empty() || !d.summary.trim().is_empty())
            .map(|d| CardDistraction {
                title: d.title.trim().to_string(),
                summary: d.summary.trim().to_string(),
                // Advisory range: the model does not know exact sub-timestamps,
                // so nested distractions inherit the parent card's window.
                start_ts: card.start_ts,
                end_ts: card.end_ts,
            })
            .collect();
    }
    cards
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_card(id: &str) -> JournalCard {
        JournalCard {
            id: id.to_string(),
            start_ts: 0,
            end_ts: 60_000_000,
            title: "Cursor: main.rs".into(),
            summary: "Deep Work activity in Cursor.".into(),
            detailed_summary: "In Cursor: main.rs; journal.rs.".into(),
            category: "Deep Work".into(),
            primary_app: "Cursor".into(),
            event_count: 3,
            distraction: false,
            distractions: Vec::new(),
        }
    }

    #[test]
    fn merge_overlays_text_and_keeps_ranges() {
        let cards = vec![base_card("journal-1-0")];
        let narrated = parse_narrated(
            r#"here you go:
            [{"id":"journal-1-0","title":"Wiring the journal narrator into Shadow","summary":"Built the LLM pass.","detailed_summary":"Edited journal.rs and server.rs.","distractions":[{"title":"Quick X check","summary":"2 min on X"}]}]"#,
        )
        .expect("parse");
        let merged = merge(cards, narrated);

        assert_eq!(merged[0].title, "Wiring the journal narrator into Shadow");
        assert_eq!(merged[0].distractions.len(), 1);
        // Range + category preserved from the deterministic card.
        assert_eq!(merged[0].start_ts, 0);
        assert_eq!(merged[0].end_ts, 60_000_000);
        assert_eq!(merged[0].category, "Deep Work");
        assert_eq!(merged[0].distractions[0].end_ts, 60_000_000);
    }

    #[test]
    fn blank_fields_do_not_wipe_heuristic_text() {
        let cards = vec![base_card("a")];
        let narrated =
            parse_narrated(r#"[{"id":"a","title":"","summary":"","detailed_summary":""}]"#)
                .expect("parse");
        let merged = merge(cards, narrated);
        assert_eq!(merged[0].title, "Cursor: main.rs");
    }

    #[test]
    fn parse_rejects_non_array() {
        assert!(parse_narrated("no json here").is_none());
    }

    #[test]
    fn parse_rejects_reversed_brackets() {
        assert!(parse_narrated("] then [").is_none());
    }

    #[test]
    fn build_user_prompt_includes_ids_and_minutes() {
        let cards = vec![base_card("journal-1-0")];
        let prompt = build_user_prompt(&cards);
        assert!(prompt.contains("journal-1-0"));
        assert!(prompt.contains("\"minutes\": 1"));
        assert!(prompt.contains("Cursor"));
        assert!(prompt.contains("Return a JSON array"));
    }

    fn remote_orchestrator() -> Arc<LlmOrchestrator> {
        // Non-local config → no Ollama probe spawned; generate() is never called
        // on the early-return paths exercised below.
        Arc::new(LlmOrchestrator::new(&crate::config::LlmConfig {
            base_url: "http://example.invalid/v1".to_string(),
            model: "m".to_string(),
            api_key: "sk-test".to_string(),
        }))
    }

    #[tokio::test]
    async fn narrate_cards_empty_returns_empty_without_calling_llm() {
        let orch = remote_orchestrator();
        let out = narrate_cards(&orch, vec![]).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn narrate_cards_too_many_returns_unchanged() {
        let orch = remote_orchestrator();
        let cards: Vec<JournalCard> = (0..MAX_NARRATED_CARDS + 1)
            .map(|i| base_card(&format!("c{i}")))
            .collect();
        let n = cards.len();
        let out = narrate_cards(&orch, cards).await;
        assert_eq!(out.len(), n, "over-cap input is returned untouched");
        assert_eq!(out[0].title, "Cursor: main.rs");
    }
}
