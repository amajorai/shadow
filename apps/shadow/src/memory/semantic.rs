use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::utils::wall_micros;

/// A single semantic memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub category: String,
    pub content: String,
    pub confidence: f32,
    pub source_episode_id: Option<String>,
    pub access_count: u32,
    pub last_accessed: u64,
    pub created_at: u64,
}

/// SQLite-backed semantic memory store.
pub struct SemanticMemoryStore {
    conn: Connection,
}

impl SemanticMemoryStore {
    pub fn new(db_path: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(db_path.parent().unwrap_or(std::path::Path::new(".")))?;
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source_episode_id TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_category ON memory_entries(category);
            CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_entries(created_at DESC);",
        )?;
        Ok(Self { conn })
    }

    /// Query memories by optional category and text substring.
    pub fn query(&self, category: Option<&str>, text: &str) -> Result<Vec<MemoryEntry>> {
        let now = wall_micros();
        let text_pat = format!("%{}%", text.to_lowercase());

        // Use a single query that fetches all matching text entries, then
        // filter by category in Rust to avoid stmt-in-match-arm lifetime issues.
        let mut stmt = self.conn.prepare(
            "SELECT id, category, content, confidence, source_episode_id, \
             access_count, last_accessed, created_at \
             FROM memory_entries WHERE lower(content) LIKE ?1 \
             ORDER BY confidence DESC, access_count DESC LIMIT 50",
        )?;
        let mut entries: Vec<MemoryEntry> = stmt
            .query_map(rusqlite::params![text_pat], row_to_entry)?
            .filter_map(|r| r.ok())
            .collect();

        if let Some(cat) = category {
            entries.retain(|e| e.category == cat);
        }
        entries.truncate(20);

        // Update access counts
        for entry in &entries {
            self.conn.execute(
                "UPDATE memory_entries SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
                rusqlite::params![now as i64, entry.id],
            ).ok();
        }

        Ok(entries)
    }

    /// Upsert a memory entry. If content+category matches an existing entry, update confidence.
    pub fn upsert(&self, entry: &MemoryEntry) -> Result<()> {
        // Check for duplicate by category + content similarity
        let existing: Option<String> = self
            .conn
            .query_row(
                "SELECT id FROM memory_entries WHERE category = ?1 AND content = ?2 LIMIT 1",
                rusqlite::params![entry.category, entry.content],
                |row| row.get(0),
            )
            .ok();

        if let Some(existing_id) = existing {
            // Update existing
            self.conn.execute(
                "UPDATE memory_entries SET confidence = MAX(confidence, ?1), access_count = access_count + 1, last_accessed = ?2 WHERE id = ?3",
                rusqlite::params![entry.confidence as f64, wall_micros() as i64, existing_id],
            )?;
        } else {
            self.conn.execute(
                "INSERT INTO memory_entries (id, category, content, confidence, source_episode_id, access_count, last_accessed, created_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    entry.id,
                    entry.category,
                    entry.content,
                    entry.confidence as f64,
                    entry.source_episode_id,
                    entry.access_count,
                    entry.last_accessed as i64,
                    entry.created_at as i64,
                ],
            )?;
        }
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM memory_entries WHERE id = ?1", [id])?;
        Ok(())
    }

    pub fn list_by_category(&self, category: &str) -> Result<Vec<MemoryEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, category, content, confidence, source_episode_id, \
             access_count, last_accessed, created_at \
             FROM memory_entries WHERE category = ?1 \
             ORDER BY confidence DESC LIMIT 50",
        )?;
        let entries = stmt
            .query_map([category], row_to_entry)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(entries)
    }
}

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEntry> {
    Ok(MemoryEntry {
        id: row.get(0)?,
        category: row.get(1)?,
        content: row.get(2)?,
        confidence: row.get::<_, f64>(3)? as f32,
        source_episode_id: row.get(4)?,
        access_count: row.get::<_, i64>(5)? as u32,
        last_accessed: row.get::<_, i64>(6)? as u64,
        created_at: row.get::<_, i64>(7)? as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("shadow-semantic-{}", uuid::Uuid::new_v4()))
    }

    fn entry(id: &str, category: &str, content: &str, confidence: f32) -> MemoryEntry {
        MemoryEntry {
            id: id.to_string(),
            category: category.to_string(),
            content: content.to_string(),
            confidence,
            source_episode_id: None,
            access_count: 0,
            last_accessed: 0,
            created_at: 1,
        }
    }

    #[test]
    fn upsert_then_query_by_text_substring() {
        let path = temp_db();
        let store = SemanticMemoryStore::new(&path).unwrap();
        store
            .upsert(&entry("1", "preference", "Prefers dark mode", 0.9))
            .unwrap();

        let hits = store.query(None, "dark").unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "1");
        // Query is case-insensitive.
        assert_eq!(store.query(None, "DARK MODE").unwrap().len(), 1);
        assert!(store.query(None, "nonexistent").unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn query_filters_by_category() {
        let path = temp_db();
        let store = SemanticMemoryStore::new(&path).unwrap();
        store
            .upsert(&entry("a", "habit", "drinks coffee", 0.8))
            .unwrap();
        store
            .upsert(&entry("b", "skill", "drinks tea recipe", 0.8))
            .unwrap();

        let habits = store.query(Some("habit"), "drinks").unwrap();
        assert_eq!(habits.len(), 1);
        assert_eq!(habits[0].category, "habit");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn upsert_dedups_by_category_and_content_keeping_max_confidence() {
        let path = temp_db();
        let store = SemanticMemoryStore::new(&path).unwrap();
        store
            .upsert(&entry("id1", "habit", "same fact", 0.5))
            .unwrap();
        // Same category+content, higher confidence, different id → updates in place.
        store
            .upsert(&entry("id2", "habit", "same fact", 0.9))
            .unwrap();

        let all = store.list_by_category("habit").unwrap();
        assert_eq!(
            all.len(),
            1,
            "duplicate content must not create a second row"
        );
        assert!((all[0].confidence - 0.9).abs() < 1e-5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn upsert_lower_confidence_does_not_lower_stored_value() {
        let path = temp_db();
        let store = SemanticMemoryStore::new(&path).unwrap();
        store.upsert(&entry("id1", "habit", "fact", 0.9)).unwrap();
        store.upsert(&entry("id2", "habit", "fact", 0.4)).unwrap();
        let all = store.list_by_category("habit").unwrap();
        assert!((all[0].confidence - 0.9).abs() < 1e-5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn query_increments_access_count() {
        let path = temp_db();
        let store = SemanticMemoryStore::new(&path).unwrap();
        store
            .upsert(&entry("1", "preference", "likes vim", 0.9))
            .unwrap();
        let _ = store.query(None, "vim").unwrap();
        let after = store.list_by_category("preference").unwrap();
        assert!(after[0].access_count >= 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn list_by_category_orders_by_confidence_desc() {
        let path = temp_db();
        let store = SemanticMemoryStore::new(&path).unwrap();
        store
            .upsert(&entry("lo", "project", "alpha task", 0.3))
            .unwrap();
        store
            .upsert(&entry("hi", "project", "beta task", 0.95))
            .unwrap();
        let list = store.list_by_category("project").unwrap();
        assert_eq!(list.len(), 2);
        assert!(list[0].confidence >= list[1].confidence);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_removes_entry() {
        let path = temp_db();
        let store = SemanticMemoryStore::new(&path).unwrap();
        store
            .upsert(&entry("1", "habit", "gone soon", 0.5))
            .unwrap();
        store.delete("1").unwrap();
        assert!(store.list_by_category("habit").unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
