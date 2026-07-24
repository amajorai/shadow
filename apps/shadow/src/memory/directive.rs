use anyhow::Result;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::utils::wall_micros;

/// A persistent behavioral directive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Directive {
    pub id: String,
    pub directive_type: String, // "reminder" | "habit" | "automation" | "watch"
    pub content: String,
    pub trigger_pattern: Option<String>,
    pub action: Option<String>,
    pub priority: u8,
    pub expires_at: Option<u64>,
    pub created_at: u64,
}

/// SQLite-backed directive store.
pub struct DirectiveMemoryStore {
    conn: Connection,
}

impl DirectiveMemoryStore {
    pub fn new(db_path: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(db_path.parent().unwrap_or(std::path::Path::new(".")))?;
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS directives (
                id TEXT PRIMARY KEY,
                directive_type TEXT NOT NULL,
                content TEXT NOT NULL,
                trigger_pattern TEXT,
                action TEXT,
                priority INTEGER NOT NULL DEFAULT 5,
                expires_at INTEGER,
                created_at INTEGER NOT NULL,
                completed_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_directives_type ON directives(directive_type);
            CREATE INDEX IF NOT EXISTS idx_directives_active ON directives(completed_at) WHERE completed_at IS NULL;",
        )?;
        Ok(Self { conn })
    }

    /// List active (non-expired, non-completed) directives.
    pub fn list_active(&self, type_filter: Option<&str>) -> Result<Vec<Directive>> {
        let now = wall_micros() as i64;
        // Single query for all active directives; filter by type in Rust if needed.
        let mut stmt = self.conn.prepare(
            "SELECT id, directive_type, content, trigger_pattern, action, priority, expires_at, created_at \
             FROM directives WHERE completed_at IS NULL \
             AND (expires_at IS NULL OR expires_at > ?1) \
             ORDER BY priority DESC, created_at DESC",
        )?;
        let mut directives: Vec<Directive> = stmt
            .query_map(rusqlite::params![now], row_to_directive)?
            .filter_map(|r| r.ok())
            .collect();
        if let Some(t) = type_filter {
            directives.retain(|d| d.directive_type == t);
        }
        Ok(directives)
    }

    pub fn create(&self, directive: &Directive) -> Result<()> {
        self.conn.execute(
            "INSERT INTO directives (id, directive_type, content, trigger_pattern, action, priority, expires_at, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                directive.id,
                directive.directive_type,
                directive.content,
                directive.trigger_pattern,
                directive.action,
                directive.priority as i64,
                directive.expires_at.map(|t| t as i64),
                directive.created_at as i64,
            ],
        )?;
        Ok(())
    }

    pub fn complete(&self, id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE directives SET completed_at = ?1 WHERE id = ?2",
            rusqlite::params![wall_micros() as i64, id],
        )?;
        Ok(())
    }

    /// Check which directives match the given context string.
    pub fn check_triggers(&self, context: &str) -> Result<Vec<Directive>> {
        let active = self.list_active(None)?;
        Ok(active
            .into_iter()
            .filter(|d| {
                d.trigger_pattern
                    .as_deref()
                    .map(|p| context.to_lowercase().contains(&p.to_lowercase()))
                    .unwrap_or(false)
            })
            .collect())
    }
}

fn row_to_directive(row: &rusqlite::Row<'_>) -> rusqlite::Result<Directive> {
    Ok(Directive {
        id: row.get(0)?,
        directive_type: row.get(1)?,
        content: row.get(2)?,
        trigger_pattern: row.get(3)?,
        action: row.get(4)?,
        priority: row.get::<_, i64>(5)? as u8,
        expires_at: row.get::<_, Option<i64>>(6)?.map(|t| t as u64),
        created_at: row.get::<_, i64>(7)? as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("shadow-directive-{}", uuid::Uuid::new_v4()))
    }

    fn directive(id: &str, dtype: &str, content: &str, priority: u8) -> Directive {
        Directive {
            id: id.to_string(),
            directive_type: dtype.to_string(),
            content: content.to_string(),
            trigger_pattern: None,
            action: None,
            priority,
            expires_at: None,
            created_at: 1,
        }
    }

    #[test]
    fn create_then_list_active_returns_it() {
        let path = temp_db();
        let store = DirectiveMemoryStore::new(&path).unwrap();
        store.create(&directive("d1", "reminder", "call back", 5)).unwrap();
        let active = store.list_active(None).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "d1");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn list_active_orders_by_priority_desc() {
        let path = temp_db();
        let store = DirectiveMemoryStore::new(&path).unwrap();
        store.create(&directive("lo", "reminder", "low", 1)).unwrap();
        store.create(&directive("hi", "reminder", "high", 9)).unwrap();
        let active = store.list_active(None).unwrap();
        assert_eq!(active[0].id, "hi");
        assert_eq!(active[1].id, "lo");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn list_active_filters_by_type() {
        let path = temp_db();
        let store = DirectiveMemoryStore::new(&path).unwrap();
        store.create(&directive("r", "reminder", "a", 5)).unwrap();
        store.create(&directive("h", "habit", "b", 5)).unwrap();
        let reminders = store.list_active(Some("reminder")).unwrap();
        assert_eq!(reminders.len(), 1);
        assert_eq!(reminders[0].directive_type, "reminder");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn completed_directives_are_excluded() {
        let path = temp_db();
        let store = DirectiveMemoryStore::new(&path).unwrap();
        store.create(&directive("d1", "reminder", "done", 5)).unwrap();
        store.complete("d1").unwrap();
        assert!(store.list_active(None).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn expired_directives_are_excluded() {
        let path = temp_db();
        let store = DirectiveMemoryStore::new(&path).unwrap();
        let mut d = directive("d1", "reminder", "stale", 5);
        d.expires_at = Some(1); // 1 microsecond since epoch → long past.
        store.create(&d).unwrap();
        assert!(store.list_active(None).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn check_triggers_matches_case_insensitive_substring() {
        let path = temp_db();
        let store = DirectiveMemoryStore::new(&path).unwrap();
        let mut d = directive("d1", "watch", "watch for invoices", 5);
        d.trigger_pattern = Some("Invoice".to_string());
        store.create(&d).unwrap();
        // Directive without a trigger must never match.
        store.create(&directive("d2", "reminder", "no trigger", 5)).unwrap();

        let matched = store.check_triggers("New INVOICE from vendor").unwrap();
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].id, "d1");
        assert!(store.check_triggers("unrelated context").unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
