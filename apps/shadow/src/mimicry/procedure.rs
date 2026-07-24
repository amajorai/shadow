use anyhow::Result;
use rusqlite::Connection;

use super::types::ProcedureTemplate;
use crate::utils::wall_micros;

/// SQLite-backed procedure store.
pub struct ProcedureStore {
    conn: Connection,
}

impl ProcedureStore {
    pub fn new(db_path: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(db_path.parent().unwrap_or(std::path::Path::new(".")))?;
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                app_name TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                steps_json TEXT NOT NULL DEFAULT '[]',
                preconditions_json TEXT NOT NULL DEFAULT '[]',
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                last_used INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_procedures_app ON procedures(app_name);
            CREATE INDEX IF NOT EXISTS idx_procedures_success ON procedures(success_count DESC);",
        )?;
        Ok(Self { conn })
    }

    pub fn save(&self, proc: &ProcedureTemplate) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO procedures \
             (id, name, app_name, description, steps_json, preconditions_json, \
              success_count, failure_count, last_used, created_at) \
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            rusqlite::params![
                proc.id,
                proc.name,
                proc.app_name,
                proc.description,
                serde_json::to_string(&proc.steps).unwrap_or_default(),
                serde_json::to_string(&proc.preconditions).unwrap_or_default(),
                proc.success_count,
                proc.failure_count,
                proc.last_used as i64,
                proc.created_at as i64,
            ],
        )?;
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<ProcedureTemplate>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, app_name, description, steps_json, preconditions_json, \
             success_count, failure_count, last_used, created_at \
             FROM procedures ORDER BY success_count DESC LIMIT 100",
        )?;
        let rows = stmt
            .query_map([], row_to_procedure)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    pub fn get(&self, id: &str) -> Result<Option<ProcedureTemplate>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, app_name, description, steps_json, preconditions_json, \
             success_count, failure_count, last_used, created_at \
             FROM procedures WHERE id = ?1",
        )?;
        let result = stmt
            .query_map([id], row_to_procedure)?
            .next()
            .and_then(|r| r.ok());
        Ok(result)
    }

    pub fn find_by_name(&self, name: &str) -> Result<Option<ProcedureTemplate>> {
        let name_pat = format!("%{}%", name.to_lowercase());
        let mut stmt = self.conn.prepare(
            "SELECT id, name, app_name, description, steps_json, preconditions_json, \
             success_count, failure_count, last_used, created_at \
             FROM procedures WHERE lower(name) LIKE ?1 OR lower(description) LIKE ?1 \
             ORDER BY success_count DESC LIMIT 1",
        )?;
        let result = stmt
            .query_map([&name_pat], row_to_procedure)?
            .next()
            .and_then(|r| r.ok());
        Ok(result)
    }

    pub fn record_success(&self, id: &str) -> Result<()> {
        let now = wall_micros() as i64;
        self.conn.execute(
            "UPDATE procedures SET success_count = success_count + 1, last_used = ?1 WHERE id = ?2",
            rusqlite::params![now, id],
        )?;
        Ok(())
    }

    pub fn record_failure(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE procedures SET failure_count = failure_count + 1 WHERE id = ?1",
            [id],
        )?;
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM procedures WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Find procedures whose name or description matches any keyword in `description`.
    /// Returns up to `limit` results ordered by success count descending.
    pub fn find_similar(&self, description: &str, limit: usize) -> Result<Vec<ProcedureTemplate>> {
        let keywords: Vec<String> = description
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_lowercase())
            .collect();

        if keywords.is_empty() {
            return Ok(vec![]);
        }

        // Load all and filter in Rust; procedure stores are small (<1000 rows)
        let mut stmt = self.conn.prepare(
            "SELECT id, name, app_name, description, steps_json, preconditions_json, \
             success_count, failure_count, last_used, created_at \
             FROM procedures ORDER BY success_count DESC",
        )?;

        let mut results: Vec<ProcedureTemplate> = stmt
            .query_map([], row_to_procedure)?
            .filter_map(|r| r.ok())
            .filter(|p| {
                let name_lower = p.name.to_lowercase();
                let desc_lower = p.description.to_lowercase();
                keywords
                    .iter()
                    .any(|kw| name_lower.contains(kw.as_str()) || desc_lower.contains(kw.as_str()))
            })
            .take(limit)
            .collect();

        Ok(results)
    }
}

fn row_to_procedure(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProcedureTemplate> {
    Ok(ProcedureTemplate {
        id: row.get(0)?,
        name: row.get(1)?,
        app_name: row.get(2)?,
        description: row.get(3)?,
        steps: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
        preconditions: serde_json::from_str(&row.get::<_, String>(5)?).unwrap_or_default(),
        success_count: row.get::<_, i64>(6)? as u32,
        failure_count: row.get::<_, i64>(7)? as u32,
        last_used: row.get::<_, i64>(8)? as u64,
        created_at: row.get::<_, i64>(9)? as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimicry::types::{ProcedureStep, StepFailureAction};

    fn temp_store() -> (ProcedureStore, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("shadow-proc-{}", uuid::Uuid::new_v4()));
        let db = dir.join("procedures.db");
        (ProcedureStore::new(&db).expect("open store"), dir)
    }

    fn make(id: &str, name: &str, app: &str, desc: &str) -> ProcedureTemplate {
        ProcedureTemplate {
            id: id.to_string(),
            name: name.to_string(),
            app_name: app.to_string(),
            description: desc.to_string(),
            steps: vec![ProcedureStep {
                step_number: 1,
                description: "step".to_string(),
                tool_name: "ax_click".to_string(),
                tool_args: serde_json::json!({"query": "Send"}),
                verification: None,
                on_failure: StepFailureAction::Abort,
            }],
            preconditions: vec!["app running".to_string()],
            success_count: 0,
            failure_count: 0,
            last_used: 0,
            created_at: 1,
        }
    }

    #[test]
    fn save_and_get_round_trips_steps_and_preconditions() {
        let (store, dir) = temp_store();
        store
            .save(&make("p1", "Compose Email", "Mail", "write a new email"))
            .unwrap();
        let got = store.get("p1").unwrap().expect("p1");
        assert_eq!(got.name, "Compose Email");
        assert_eq!(got.steps.len(), 1);
        assert_eq!(got.steps[0].tool_name, "ax_click");
        assert_eq!(got.preconditions, vec!["app running".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_missing_returns_none() {
        let (store, dir) = temp_store();
        assert!(store.get("does-not-exist").unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_replaces_on_duplicate_id() {
        let (store, dir) = temp_store();
        store.save(&make("dup", "First", "App", "d")).unwrap();
        store.save(&make("dup", "Second", "App", "d")).unwrap();
        assert_eq!(store.list().unwrap().len(), 1);
        assert_eq!(store.get("dup").unwrap().unwrap().name, "Second");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_orders_by_success_count_desc() {
        let (store, dir) = temp_store();
        let mut low = make("low", "Low", "A", "d");
        low.success_count = 1;
        let mut high = make("high", "High", "A", "d");
        high.success_count = 9;
        store.save(&low).unwrap();
        store.save(&high).unwrap();
        let list = store.list().unwrap();
        assert_eq!(list[0].id, "high");
        assert_eq!(list[1].id, "low");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_success_and_failure_update_counts() {
        let (store, dir) = temp_store();
        store.save(&make("p", "P", "A", "d")).unwrap();
        store.record_success("p").unwrap();
        store.record_success("p").unwrap();
        store.record_failure("p").unwrap();
        let got = store.get("p").unwrap().unwrap();
        assert_eq!(got.success_count, 2);
        assert_eq!(got.failure_count, 1);
        assert!(got.last_used > 0, "record_success stamps last_used");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_removes_the_row() {
        let (store, dir) = temp_store();
        store.save(&make("p", "P", "A", "d")).unwrap();
        store.delete("p").unwrap();
        assert!(store.get("p").unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_by_name_is_case_insensitive_substring() {
        let (store, dir) = temp_store();
        store
            .save(&make("p", "Compose Email", "Mail", "start a draft"))
            .unwrap();
        assert!(store.find_by_name("compose").unwrap().is_some());
        assert!(store.find_by_name("EMAIL").unwrap().is_some());
        // Matches description too.
        assert!(store.find_by_name("draft").unwrap().is_some());
        assert!(store.find_by_name("nonexistent").unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_similar_matches_keywords_over_three_chars() {
        let (store, dir) = temp_store();
        store
            .save(&make(
                "p1",
                "Send Report",
                "Mail",
                "email the weekly report",
            ))
            .unwrap();
        store
            .save(&make("p2", "Resize Photo", "Photos", "crop and resize"))
            .unwrap();

        let hits = store.find_similar("send report", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "p1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_similar_ignores_short_words_and_returns_empty() {
        let (store, dir) = temp_store();
        store.save(&make("p1", "Send Report", "Mail", "d")).unwrap();
        // All words <= 3 chars → no keywords → empty result.
        let hits = store.find_similar("go to it", 10).unwrap();
        assert!(hits.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_similar_respects_limit() {
        let (store, dir) = temp_store();
        for i in 0..4 {
            store
                .save(&make(
                    &format!("p{i}"),
                    "Report Builder",
                    "App",
                    "generate report",
                ))
                .unwrap();
        }
        let hits = store.find_similar("report", 2).unwrap();
        assert_eq!(hits.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
