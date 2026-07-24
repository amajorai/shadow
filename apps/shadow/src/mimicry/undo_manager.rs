use crate::utils::wall_micros;

/// An entry pushed onto the undo stack before each executed step.
#[derive(Debug, Clone)]
pub struct UndoEntry {
    pub step_index: usize,
    /// Tool name that was about to execute (e.g. "ax_click").
    pub action_type: String,
    /// FNV hash of the AX tree state captured immediately before the step.
    pub ax_tree_hash: u64,
    pub timestamp_us: u64,
    /// App that was in focus, if known (for SwitchBack strategy).
    pub app_context: Option<String>,
    /// Scroll deltas for ReverseScroll strategy.
    pub scroll_dx: Option<i32>,
    pub scroll_dy: Option<i32>,
}

/// Strategy computed when undoing an entry.
#[derive(Debug, Clone)]
pub enum UndoStrategy {
    /// Send Ctrl+Z / Cmd+Z.
    UndoShortcut,
    /// Switch focus back to a previous app.
    SwitchBack(String),
    /// Scroll the opposite amount.
    ReverseScroll { dx: i32, dy: i32 },
    /// Cannot auto-undo; human intervention required.
    Manual(String),
}

/// LIFO stack of pre-step snapshots enabling step reversal.
pub struct ExecutionUndoManager {
    stack: Vec<UndoEntry>,
    max_size: usize,
}

impl ExecutionUndoManager {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            max_size: 50,
        }
    }

    /// Push an entry. Trims oldest entries when `max_size` is exceeded.
    pub fn push(&mut self, entry: UndoEntry) {
        if self.stack.len() >= self.max_size {
            self.stack.remove(0);
        }
        self.stack.push(entry);
    }

    /// Build and push a new entry in one call.
    pub fn push_step(
        &mut self,
        step_index: usize,
        action_type: impl Into<String>,
        ax_tree_hash: u64,
        app_context: Option<String>,
        scroll_dx: Option<i32>,
        scroll_dy: Option<i32>,
    ) {
        self.push(UndoEntry {
            step_index,
            action_type: action_type.into(),
            ax_tree_hash,
            timestamp_us: wall_micros(),
            app_context,
            scroll_dx,
            scroll_dy,
        });
    }

    /// Remove and return the most recent entry.
    pub fn pop(&mut self) -> Option<UndoEntry> {
        self.stack.pop()
    }

    /// Inspect the most recent entry without removing it.
    pub fn peek(&self) -> Option<&UndoEntry> {
        self.stack.last()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.stack.clear();
    }

    /// Length of the stack.
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Compute the reversal strategy for an entry without popping it.
    pub fn compute_reversal(entry: &UndoEntry) -> UndoStrategy {
        match entry.action_type.as_str() {
            "ax_click" | "ax_type" | "ax_hotkey" => UndoStrategy::UndoShortcut,
            "ax_scroll" => UndoStrategy::ReverseScroll {
                dx: -entry.scroll_dx.unwrap_or(0),
                dy: -entry.scroll_dy.unwrap_or(0),
            },
            "ax_focus_app" => {
                if let Some(app) = &entry.app_context {
                    UndoStrategy::SwitchBack(app.clone())
                } else {
                    UndoStrategy::Manual("Cannot restore focus: previous app unknown".to_string())
                }
            }
            other => UndoStrategy::Manual(format!("No auto-undo for action '{}'", other)),
        }
    }

    /// Pop the most recent entry and return its reversal strategy.
    pub fn pop_reversal(&mut self) -> Option<UndoStrategy> {
        self.stack.pop().map(|e| Self::compute_reversal(&e))
    }
}

impl Default for ExecutionUndoManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(action: &str) -> UndoEntry {
        UndoEntry {
            step_index: 0,
            action_type: action.to_string(),
            ax_tree_hash: 0,
            timestamp_us: 0,
            app_context: None,
            scroll_dx: None,
            scroll_dy: None,
        }
    }

    #[test]
    fn push_peek_pop_are_lifo() {
        let mut m = ExecutionUndoManager::new();
        assert!(m.is_empty());
        m.push(entry("ax_click"));
        m.push(entry("ax_type"));
        assert_eq!(m.len(), 2);
        assert_eq!(m.peek().unwrap().action_type, "ax_type");
        assert_eq!(m.pop().unwrap().action_type, "ax_type");
        assert_eq!(m.pop().unwrap().action_type, "ax_click");
        assert!(m.pop().is_none());
    }

    #[test]
    fn push_step_builds_entry() {
        let mut m = ExecutionUndoManager::new();
        m.push_step(3, "ax_scroll", 42, Some("Mail".to_string()), Some(5), Some(-5));
        let e = m.peek().unwrap();
        assert_eq!(e.step_index, 3);
        assert_eq!(e.action_type, "ax_scroll");
        assert_eq!(e.ax_tree_hash, 42);
        assert_eq!(e.app_context.as_deref(), Some("Mail"));
        assert_eq!(e.scroll_dx, Some(5));
    }

    #[test]
    fn push_trims_oldest_beyond_max_size() {
        let mut m = ExecutionUndoManager::new();
        for i in 0..60 {
            let mut e = entry("ax_click");
            e.step_index = i;
            m.push(e);
        }
        assert_eq!(m.len(), 50, "stack capped at max_size");
        // Oldest (step_index 0..9) trimmed; the bottom is now step 10.
        assert_eq!(m.pop().unwrap().step_index, 59);
    }

    #[test]
    fn clear_empties_stack() {
        let mut m = ExecutionUndoManager::new();
        m.push(entry("ax_click"));
        m.clear();
        assert!(m.is_empty());
    }

    #[test]
    fn compute_reversal_click_type_hotkey_use_undo_shortcut() {
        for action in ["ax_click", "ax_type", "ax_hotkey"] {
            assert!(matches!(
                ExecutionUndoManager::compute_reversal(&entry(action)),
                UndoStrategy::UndoShortcut
            ));
        }
    }

    #[test]
    fn compute_reversal_scroll_negates_deltas() {
        let mut e = entry("ax_scroll");
        e.scroll_dx = Some(7);
        e.scroll_dy = Some(-3);
        match ExecutionUndoManager::compute_reversal(&e) {
            UndoStrategy::ReverseScroll { dx, dy } => {
                assert_eq!(dx, -7);
                assert_eq!(dy, 3);
            }
            other => panic!("expected ReverseScroll, got {other:?}"),
        }
    }

    #[test]
    fn compute_reversal_focus_app_switches_back_or_manual() {
        let mut with_ctx = entry("ax_focus_app");
        with_ctx.app_context = Some("Slack".to_string());
        match ExecutionUndoManager::compute_reversal(&with_ctx) {
            UndoStrategy::SwitchBack(app) => assert_eq!(app, "Slack"),
            other => panic!("expected SwitchBack, got {other:?}"),
        }
        // No app context → Manual.
        assert!(matches!(
            ExecutionUndoManager::compute_reversal(&entry("ax_focus_app")),
            UndoStrategy::Manual(_)
        ));
    }

    #[test]
    fn compute_reversal_unknown_action_is_manual() {
        assert!(matches!(
            ExecutionUndoManager::compute_reversal(&entry("ax_teleport")),
            UndoStrategy::Manual(_)
        ));
    }

    #[test]
    fn pop_reversal_pops_and_computes() {
        let mut m = ExecutionUndoManager::new();
        m.push(entry("ax_click"));
        assert!(matches!(m.pop_reversal(), Some(UndoStrategy::UndoShortcut)));
        assert!(m.pop_reversal().is_none());
    }

    #[test]
    fn default_is_empty() {
        assert!(ExecutionUndoManager::default().is_empty());
    }
}
