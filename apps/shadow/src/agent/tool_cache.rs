use std::collections::HashMap;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// Single-session in-memory cache for agent tool results.
/// Keyed by `tool_name + sha256(args)` with per-tool TTLs.
pub struct ToolResultCache {
    entries: HashMap<String, CacheEntry>,
}

struct CacheEntry {
    value: serde_json::Value,
    expires_at: Instant,
}

impl ToolResultCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Look up a cached result, evicting stale entries in the process.
    pub fn get(&mut self, tool: &str, args: &serde_json::Value) -> Option<serde_json::Value> {
        self.evict_expired();
        let key = cache_key(tool, args);
        self.entries.get(&key).map(|e| e.value.clone())
    }

    /// Store a result. Silently skips tools that should never be cached
    /// (e.g. screenshots, live AX captures).
    pub fn set(&mut self, tool: &str, args: &serde_json::Value, value: serde_json::Value) {
        let ttl = ttl_for_tool(tool);
        if ttl.is_zero() {
            return; // never cache
        }
        let key = cache_key(tool, args);
        self.entries.insert(
            key,
            CacheEntry {
                value,
                expires_at: Instant::now() + ttl,
            },
        );
    }

    /// Remove all expired entries.
    pub fn evict_expired(&mut self) {
        let now = Instant::now();
        self.entries.retain(|_, e| e.expires_at > now);
    }

    /// Clear all entries (called at end of agent run).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for ToolResultCache {
    fn default() -> Self {
        Self::new()
    }
}

fn ttl_for_tool(tool: &str) -> Duration {
    match tool {
        // Never cache live visual outputs
        "capture_live_screenshot" | "ax_screenshot" => Duration::ZERO,
        // AX tree is volatile
        "ax_tree_query" | "ax_inspect" | "ax_element_at" => Duration::from_secs(15),
        // Memory/knowledge lookups are stable within a session
        "get_knowledge" | "get_directives" | "search_summaries" => Duration::from_secs(300),
        "search_hybrid" | "search_visual_memories" => Duration::from_secs(60),
        _ => Duration::from_secs(30),
    }
}

fn cache_key(tool: &str, args: &serde_json::Value) -> String {
    let args_str = serde_json::to_string(args).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(tool.as_bytes());
    hasher.update(b":");
    hasher.update(args_str.as_bytes());
    let hash = hasher.finalize();
    format!("{:x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn set_then_get_round_trips_for_cacheable_tool() {
        let mut cache = ToolResultCache::new();
        let args = json!({"category": "prefs"});
        cache.set("get_knowledge", &args, json!({"facts": [1, 2]}));
        let got = cache.get("get_knowledge", &args);
        assert_eq!(got, Some(json!({"facts": [1, 2]})));
    }

    #[test]
    fn get_returns_none_for_different_args() {
        let mut cache = ToolResultCache::new();
        cache.set("get_knowledge", &json!({"q": "a"}), json!("A"));
        assert_eq!(cache.get("get_knowledge", &json!({"q": "b"})), None);
    }

    #[test]
    fn never_cached_tools_are_not_stored() {
        let mut cache = ToolResultCache::new();
        let args = json!({});
        cache.set("capture_live_screenshot", &args, json!("img"));
        cache.set("ax_screenshot", &args, json!("img"));
        assert_eq!(cache.get("capture_live_screenshot", &args), None);
        assert_eq!(cache.get("ax_screenshot", &args), None);
    }

    #[test]
    fn clear_empties_the_cache() {
        let mut cache = ToolResultCache::new();
        cache.set("get_directives", &json!({}), json!([]));
        assert!(cache.get("get_directives", &json!({})).is_some());
        cache.clear();
        assert!(cache.get("get_directives", &json!({})).is_none());
    }

    #[test]
    fn evict_expired_removes_stale_entries() {
        let mut cache = ToolResultCache::new();
        // Manually insert an already-expired entry.
        let key = cache_key("get_knowledge", &json!({}));
        cache.entries.insert(
            key,
            CacheEntry {
                value: json!("stale"),
                expires_at: Instant::now() - Duration::from_secs(1),
            },
        );
        cache.evict_expired();
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn ttl_for_tool_matches_policy() {
        assert_eq!(ttl_for_tool("capture_live_screenshot"), Duration::ZERO);
        assert_eq!(ttl_for_tool("ax_screenshot"), Duration::ZERO);
        assert_eq!(ttl_for_tool("ax_tree_query"), Duration::from_secs(15));
        assert_eq!(ttl_for_tool("get_knowledge"), Duration::from_secs(300));
        assert_eq!(ttl_for_tool("search_hybrid"), Duration::from_secs(60));
        // Unknown tools fall back to the 30s default.
        assert_eq!(ttl_for_tool("some_new_tool"), Duration::from_secs(30));
    }

    #[test]
    fn cache_key_is_deterministic_and_order_sensitive_to_tool() {
        let args = json!({"a": 1});
        assert_eq!(cache_key("t", &args), cache_key("t", &args));
        assert_ne!(cache_key("t1", &args), cache_key("t2", &args));
    }

    #[test]
    fn default_is_equivalent_to_new() {
        let mut cache = ToolResultCache::default();
        assert!(cache.get("get_knowledge", &json!({})).is_none());
    }
}
