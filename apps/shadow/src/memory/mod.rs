pub mod consolidator;
pub mod directive;
pub mod query_planner;
pub mod semantic;

pub use consolidator::SemanticConsolidator;
pub use directive::{Directive, DirectiveMemoryStore};
pub use query_planner::{MemoryQueryPlanner, MemoryResult, MemorySource, QueryPlan};
pub use semantic::{MemoryEntry, SemanticMemoryStore};

use anyhow::Result;
use std::sync::{Mutex, OnceLock};

/// Unified memory facade combining semantic and directive stores.
pub struct MemoryStore {
    pub semantic: SemanticMemoryStore,
    directive: DirectiveMemoryStore,
}

/// Global memory store — initialized once via init_memory().
/// Wrapped in Mutex because rusqlite::Connection is !Sync.
pub static MEMORY_STORE: OnceLock<Mutex<MemoryStore>> = OnceLock::new();

/// Initialize the global memory store.
pub fn init_memory(db_path: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(db_path.parent().unwrap_or(std::path::Path::new(".")))?;
    let semantic = SemanticMemoryStore::new(db_path)?;
    let directive = DirectiveMemoryStore::new(db_path)?;
    MEMORY_STORE
        .set(Mutex::new(MemoryStore {
            semantic,
            directive,
        }))
        .map_err(|_| anyhow::anyhow!("Memory store already initialized"))?;
    Ok(())
}

impl MemoryStore {
    /// Query semantic memory by optional category and text.
    pub fn query(&self, category: Option<&str>, text: &str) -> Result<Vec<MemoryEntry>> {
        self.semantic.query(category, text)
    }

    /// Store or update a semantic memory entry.
    pub fn upsert(&self, entry: &MemoryEntry) -> Result<()> {
        self.semantic.upsert(entry)
    }

    /// Delete a semantic memory entry by ID.
    pub fn delete_entry(&self, id: &str) -> Result<()> {
        self.semantic.delete(id)
    }

    /// Create a new directive.
    pub fn create_directive(&self, directive: &Directive) -> Result<()> {
        self.directive.create(directive)
    }

    /// List active directives, optionally filtered by type.
    pub fn list_active(&self, type_filter: Option<&str>) -> Result<Vec<Directive>> {
        self.directive.list_active(type_filter)
    }

    /// Mark a directive as completed.
    pub fn complete_directive(&self, id: &str) -> Result<()> {
        self.directive.complete(id)
    }

    /// Check which directives match the current context.
    pub fn check_triggers(&self, context: &str) -> Result<Vec<Directive>> {
        self.directive.check_triggers(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `MemoryStore` facade over two stores backed by the same temp db.
    /// (The struct's `directive` field is private, so `init_memory` is the public
    /// path; but we only need a facade instance here, constructed via the stores.)
    fn facade(path: &std::path::Path) -> MemoryStore {
        MemoryStore {
            semantic: SemanticMemoryStore::new(path).unwrap(),
            directive: DirectiveMemoryStore::new(path).unwrap(),
        }
    }

    fn temp_db() -> std::path::PathBuf {
        std::env::temp_dir().join(format!("shadow-memfacade-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn facade_semantic_upsert_query_delete() {
        let path = temp_db();
        let store = facade(&path);
        store
            .upsert(&MemoryEntry {
                id: "s1".to_string(),
                category: "preference".to_string(),
                content: "likes tea".to_string(),
                confidence: 0.8,
                source_episode_id: None,
                access_count: 0,
                last_accessed: 0,
                created_at: 1,
            })
            .unwrap();
        assert_eq!(store.query(Some("preference"), "tea").unwrap().len(), 1);
        store.delete_entry("s1").unwrap();
        assert!(store.query(None, "tea").unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn facade_directive_create_list_complete_and_triggers() {
        let path = temp_db();
        let store = facade(&path);
        store
            .create_directive(&Directive {
                id: "d1".to_string(),
                directive_type: "watch".to_string(),
                content: "watch invoices".to_string(),
                trigger_pattern: Some("invoice".to_string()),
                action: None,
                priority: 5,
                expires_at: None,
                created_at: 1,
            })
            .unwrap();
        assert_eq!(store.list_active(None).unwrap().len(), 1);
        assert_eq!(
            store.check_triggers("new INVOICE arrived").unwrap().len(),
            1
        );
        assert!(store.check_triggers("nothing relevant").unwrap().is_empty());

        store.complete_directive("d1").unwrap();
        assert!(store.list_active(None).unwrap().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn init_memory_sets_the_global_once() {
        let path = std::env::temp_dir()
            .join(format!("shadow-initmem-{}", uuid::Uuid::new_v4()))
            .join("memory.db");
        // First init succeeds and populates the global (unless a prior test in
        // this binary already claimed it — tolerate both).
        let first = init_memory(&path);
        if first.is_ok() {
            assert!(MEMORY_STORE.get().is_some());
            // A second init must fail — the OnceLock is already set.
            assert!(init_memory(&path).is_err());
        }
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
