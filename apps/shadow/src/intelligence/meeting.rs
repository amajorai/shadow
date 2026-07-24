use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::llm::{orchestrator::LlmOrchestrator, LlmMessage, LlmRequest};
use crate::utils::wall_micros;

/// A candidate meeting window detected from audio/activity overlap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingWindow {
    pub start_us: u64,
    pub end_us: u64,
    pub app_name: String,
    pub confidence: f32,
}

/// Full structured meeting summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSummary {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub key_points: Vec<String>,
    pub decisions: Vec<String>,
    pub action_items: Vec<ActionItem>,
    pub open_questions: Vec<String>,
    pub highlights: Vec<String>,
    pub participants: Vec<String>,
    pub start_us: u64,
    pub end_us: u64,
    pub app_name: String,
    pub created_at: u64,
    pub transcript_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub description: String,
    pub owner: Option<String>,
    pub due_date: Option<String>,
}

/// Detects meeting windows from timeline activity.
pub struct MeetingResolver;

impl MeetingResolver {
    pub fn find_meetings(&self, start_us: u64, end_us: u64) -> Result<Vec<MeetingWindow>> {
        let entries = shadow_core::query_time_range(start_us, end_us)?;

        let meeting_apps = [
            "zoom", "meet", "teams", "webex", "skype", "slack", "discord", "facetime", "telegram",
            "signal",
        ];

        let mut meetings = vec![];
        let mut current_meeting: Option<(u64, String)> = None;

        for entry in &entries {
            let app = entry.app_name.as_deref().unwrap_or("");
            let app_lower = app.to_lowercase();
            let is_meeting = meeting_apps.iter().any(|&m| app_lower.contains(m));

            if is_meeting {
                match &current_meeting {
                    None => {
                        current_meeting = Some((entry.ts, app.to_string()));
                    }
                    Some((start, cur_app)) => {
                        if app != cur_app {
                            // Different meeting app — close current
                            meetings.push(MeetingWindow {
                                start_us: *start,
                                end_us: entry.ts,
                                app_name: cur_app.clone(),
                                confidence: 0.85,
                            });
                            current_meeting = Some((entry.ts, app.to_string()));
                        }
                        // else: continue the same meeting
                    }
                }
            } else if let Some((start, cur_app)) = current_meeting.take() {
                // Meeting ended
                meetings.push(MeetingWindow {
                    start_us: start,
                    end_us: entry.ts,
                    app_name: cur_app,
                    confidence: 0.85,
                });
            }
        }

        // Close any open meeting
        if let Some((start, app)) = current_meeting {
            meetings.push(MeetingWindow {
                start_us: start,
                end_us: end_us,
                app_name: app,
                confidence: 0.85,
            });
        }

        Ok(meetings)
    }
}

/// Summarizes meetings using LLM + transcript.
pub struct MeetingSummarizer {
    orchestrator: Arc<LlmOrchestrator>,
}

impl MeetingSummarizer {
    pub fn new(orchestrator: Arc<LlmOrchestrator>) -> Self {
        Self { orchestrator }
    }

    pub async fn summarize(&self, window: &MeetingWindow) -> Result<MeetingSummary> {
        // Gather timeline context for the meeting window
        let entries = shadow_core::query_time_range(window.start_us, window.end_us)?;
        let context: Vec<String> = entries
            .iter()
            .map(|e| {
                format!(
                    "[{}] {}: {}",
                    e.ts / 1_000_000,
                    e.app_name.as_deref().unwrap_or(""),
                    e.window_title.as_deref().unwrap_or(""),
                )
            })
            .collect();

        let context_str = context.join("\n");
        let duration_min = (window.end_us - window.start_us) / 60_000_000;

        let prompt = format!(
            "You are summarizing a meeting that took place in {}.\n\
             Duration: ~{} minutes.\n\
             Activity log:\n{}\n\n\
             Generate a structured meeting summary. Respond with JSON matching this schema:\n\
             {{\n\
               \"title\": \"Brief meeting title\",\n\
               \"summary\": \"2-3 sentence overview\",\n\
               \"key_points\": [\"point1\", ...],\n\
               \"decisions\": [\"decision1\", ...],\n\
               \"action_items\": [{{\"description\": \"...\", \"owner\": null, \"due_date\": null}}, ...],\n\
               \"open_questions\": [\"question1\", ...],\n\
               \"highlights\": [\"highlight1\", ...],\n\
               \"participants\": []\n\
             }}",
            window.app_name,
            duration_min,
            context_str.chars().take(3000).collect::<String>()
        );

        let response = self
            .orchestrator
            .generate(LlmRequest {
                messages: vec![LlmMessage::user(prompt)],
                temperature: 0.3,
                max_tokens: 1024,
                ..Default::default()
            })
            .await?;

        let content = response.content.unwrap_or_default();
        let json_str = extract_json_object(&content).unwrap_or_else(|| "{}".to_string());
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_default();

        // Compute a transcript hash for dedup
        let hash = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(context_str.as_bytes());
            format!("{:x}", hasher.finalize())
        };

        let now = wall_micros();
        Ok(MeetingSummary {
            id: uuid::Uuid::new_v4().to_string(),
            title: parsed["title"].as_str().unwrap_or("Meeting").to_string(),
            summary: parsed["summary"].as_str().unwrap_or("").to_string(),
            key_points: json_array_of_strings(&parsed["key_points"]),
            decisions: json_array_of_strings(&parsed["decisions"]),
            action_items: parse_action_items(&parsed["action_items"]),
            open_questions: json_array_of_strings(&parsed["open_questions"]),
            highlights: json_array_of_strings(&parsed["highlights"]),
            participants: json_array_of_strings(&parsed["participants"]),
            start_us: window.start_us,
            end_us: window.end_us,
            app_name: window.app_name.clone(),
            created_at: now,
            transcript_hash: hash,
        })
    }
}

fn json_array_of_strings(val: &serde_json::Value) -> Vec<String> {
    val.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_action_items(val: &serde_json::Value) -> Vec<ActionItem> {
    val.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    Some(ActionItem {
                        description: v["description"].as_str()?.to_string(),
                        owner: v["owner"].as_str().map(|s| s.to_string()),
                        due_date: v["due_date"].as_str().map(|s| s.to_string()),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn extract_json_object(s: &str) -> Option<String> {
    let start = s.find('{')?;
    // Find the matching closing brace
    let bytes = s[start..].as_bytes();
    let mut depth = 0i32;
    let mut end = start;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    if end > start {
        Some(s[start..=end].to_string())
    } else {
        None
    }
}

/// SQLite-backed store for meeting summaries.
pub struct SummaryStore {
    conn: rusqlite::Connection,
}

impl SummaryStore {
    pub fn new(db_path: &std::path::Path) -> Result<Self> {
        let conn = rusqlite::Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meeting_summaries (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_points TEXT NOT NULL DEFAULT '[]',
                decisions TEXT NOT NULL DEFAULT '[]',
                action_items TEXT NOT NULL DEFAULT '[]',
                open_questions TEXT NOT NULL DEFAULT '[]',
                highlights TEXT NOT NULL DEFAULT '[]',
                participants TEXT NOT NULL DEFAULT '[]',
                start_us INTEGER NOT NULL,
                end_us INTEGER NOT NULL,
                app_name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                transcript_hash TEXT NOT NULL UNIQUE
            );",
        )?;
        Ok(Self { conn })
    }

    pub fn store(&self, summary: &MeetingSummary) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO meeting_summaries \
             (id, title, summary, key_points, decisions, action_items, open_questions, \
              highlights, participants, start_us, end_us, app_name, created_at, transcript_hash) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
            rusqlite::params![
                summary.id,
                summary.title,
                summary.summary,
                serde_json::to_string(&summary.key_points).unwrap_or_default(),
                serde_json::to_string(&summary.decisions).unwrap_or_default(),
                serde_json::to_string(&summary.action_items).unwrap_or_default(),
                serde_json::to_string(&summary.open_questions).unwrap_or_default(),
                serde_json::to_string(&summary.highlights).unwrap_or_default(),
                serde_json::to_string(&summary.participants).unwrap_or_default(),
                summary.start_us as i64,
                summary.end_us as i64,
                summary.app_name,
                summary.created_at as i64,
                summary.transcript_hash,
            ],
        )?;
        Ok(())
    }

    pub fn list(&self, limit: usize) -> Result<Vec<MeetingSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, summary, key_points, decisions, action_items, open_questions, \
             highlights, participants, start_us, end_us, app_name, created_at, transcript_hash \
             FROM meeting_summaries ORDER BY created_at DESC LIMIT ?1",
        )?;

        let rows = stmt.query_map([limit as i64], |row| {
            Ok(MeetingSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                summary: row.get(2)?,
                key_points: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                decisions: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                action_items: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                open_questions: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                highlights: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                participants: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                start_us: row.get::<_, i64>(9)? as u64,
                end_us: row.get::<_, i64>(10)? as u64,
                app_name: row.get(11)?,
                created_at: row.get::<_, i64>(12)? as u64,
                transcript_hash: row.get(13)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn get(&self, id: &str) -> Result<Option<MeetingSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, title, summary, key_points, decisions, action_items, open_questions, \
             highlights, participants, start_us, end_us, app_name, created_at, transcript_hash \
             FROM meeting_summaries WHERE id = ?1",
        )?;

        let mut rows = stmt.query_map([id], |row| {
            Ok(MeetingSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                summary: row.get(2)?,
                key_points: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                decisions: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                action_items: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
                open_questions: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                highlights: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                participants: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
                start_us: row.get::<_, i64>(9)? as u64,
                end_us: row.get::<_, i64>(10)? as u64,
                app_name: row.get(11)?,
                created_at: row.get::<_, i64>(12)? as u64,
                transcript_hash: row.get(13)?,
            })
        })?;

        Ok(rows.next().and_then(|r| r.ok()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("shadow-summary-{}", uuid::Uuid::new_v4()))
    }

    fn sample_summary(id: &str, hash: &str) -> MeetingSummary {
        MeetingSummary {
            id: id.to_string(),
            title: "Standup".to_string(),
            summary: "We synced.".to_string(),
            key_points: vec!["shipped X".to_string()],
            decisions: vec!["use Y".to_string()],
            action_items: vec![ActionItem {
                description: "follow up".to_string(),
                owner: Some("alice".to_string()),
                due_date: None,
            }],
            open_questions: vec![],
            highlights: vec!["good demo".to_string()],
            participants: vec!["alice".to_string(), "bob".to_string()],
            start_us: 100,
            end_us: 200,
            app_name: "Zoom".to_string(),
            created_at: 1,
            transcript_hash: hash.to_string(),
        }
    }

    #[test]
    fn json_array_of_strings_filters_non_strings() {
        let v = json!(["a", 1, "b", null, "c"]);
        assert_eq!(json_array_of_strings(&v), vec!["a", "b", "c"]);
        // Non-array yields empty.
        assert!(json_array_of_strings(&json!("nope")).is_empty());
    }

    #[test]
    fn parse_action_items_reads_fields_and_skips_incomplete() {
        let v = json!([
            {"description": "do thing", "owner": "bob", "due_date": "2026-08-01"},
            {"owner": "no description"}
        ]);
        let items = parse_action_items(&v);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].description, "do thing");
        assert_eq!(items[0].owner.as_deref(), Some("bob"));
        assert_eq!(items[0].due_date.as_deref(), Some("2026-08-01"));
    }

    #[test]
    fn parse_action_items_owner_and_due_optional() {
        let v = json!([{"description": "solo"}]);
        let items = parse_action_items(&v);
        assert_eq!(items.len(), 1);
        assert!(items[0].owner.is_none());
        assert!(items[0].due_date.is_none());
    }

    #[test]
    fn extract_json_object_balances_braces() {
        assert_eq!(
            extract_json_object("pre {\"a\":{\"b\":1}} post").unwrap(),
            "{\"a\":{\"b\":1}}"
        );
        assert_eq!(extract_json_object("no object"), None);
        assert_eq!(extract_json_object("{unterminated"), None);
    }

    #[test]
    fn summary_store_store_and_get_round_trips() {
        let path = temp_db();
        let store = SummaryStore::new(&path).unwrap();
        store.store(&sample_summary("m1", "hash1")).unwrap();

        let got = store.get("m1").unwrap().expect("stored summary");
        assert_eq!(got.title, "Standup");
        assert_eq!(got.key_points, vec!["shipped X".to_string()]);
        assert_eq!(got.action_items.len(), 1);
        assert_eq!(got.action_items[0].owner.as_deref(), Some("alice"));
        assert_eq!(got.participants.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn summary_store_get_missing_is_none() {
        let path = temp_db();
        let store = SummaryStore::new(&path).unwrap();
        assert!(store.get("absent").unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn summary_store_list_orders_by_created_at_desc() {
        let path = temp_db();
        let store = SummaryStore::new(&path).unwrap();
        let mut older = sample_summary("old", "h_old");
        older.created_at = 10;
        let mut newer = sample_summary("new", "h_new");
        newer.created_at = 20;
        store.store(&older).unwrap();
        store.store(&newer).unwrap();

        let list = store.list(10).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "new");
        assert_eq!(list[1].id, "old");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn summary_store_dedups_on_transcript_hash() {
        let path = temp_db();
        let store = SummaryStore::new(&path).unwrap();
        store.store(&sample_summary("first", "samehash")).unwrap();
        // Same transcript_hash (UNIQUE) → INSERT OR IGNORE keeps the first.
        store.store(&sample_summary("second", "samehash")).unwrap();
        let list = store.list(10).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "first");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn meeting_window_and_summary_serialize() {
        let w = MeetingWindow {
            start_us: 1,
            end_us: 2,
            app_name: "Meet".to_string(),
            confidence: 0.85,
        };
        let v = serde_json::to_value(&w).unwrap();
        assert!((v["confidence"].as_f64().unwrap() - 0.85).abs() < 1e-4);
        assert_eq!(v["app_name"], json!("Meet"));
        assert_eq!(v["start_us"], json!(1));
    }
}
