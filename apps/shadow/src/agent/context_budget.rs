use crate::llm::LlmMessage;

/// Which tier of LLM model is being used.
#[derive(Debug, Clone, Copy)]
pub enum ModelTier {
    Cloud,
    LocalLarge,
    LocalSmall,
}

/// Specialized role for an agent sub-task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    /// Read and report screen state.
    Observer,
    /// Perform UI actions.
    Executor,
    /// Retrieve from memory stores.
    MemoryManager,
    /// Record and synthesize procedures.
    LearningEngine,
    /// Assess risk before execution.
    SafetyMonitor,
    /// Full context, no specialization.
    General,
}

/// Token budget allocations per tier.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub total: usize,
    pub system: usize,
    pub memory: usize,
    pub max_per_turn: usize,
}

impl Budget {
    fn for_tier(tier: ModelTier) -> Self {
        match tier {
            ModelTier::Cloud => Budget {
                total: 128_000,
                system: 2_000,
                memory: 4_000,
                max_per_turn: 32_000,
            },
            ModelTier::LocalLarge => Budget {
                total: 32_000,
                system: 1_500,
                memory: 2_000,
                max_per_turn: 8_000,
            },
            ModelTier::LocalSmall => Budget {
                total: 16_000,
                system: 1_000,
                memory: 1_000,
                max_per_turn: 4_000,
            },
        }
    }
}

pub struct ContextBudgetManager;

impl ContextBudgetManager {
    /// Trim `history` to fit within the `max_per_turn` budget and prepend
    /// the appropriate system prompt and memory context.
    ///
    /// Always preserves the system message and the most recent user message.
    pub fn build_context(
        role: AgentRole,
        tier: ModelTier,
        history: &[LlmMessage],
        memory_context: &str,
    ) -> Vec<LlmMessage> {
        let budget = Budget::for_tier(tier);
        let system_text = Self::system_prompt_for_role(role);
        // 4 chars/token; snap the cut DOWN to a char boundary — a raw byte slice
        // panics when a multi-byte char straddles the budget index.
        let memory_cut = floor_char_boundary(memory_context, budget.memory * 4);
        let memory_text = &memory_context[..memory_cut];

        let system_msg = if memory_text.is_empty() {
            LlmMessage::system(system_text)
        } else {
            LlmMessage::system(format!("{}\n\nContext:\n{}", system_text, memory_text))
        };

        // Trim history to fit max_per_turn
        let char_budget = budget.max_per_turn * 4; // conservative 4 chars/token
        let mut trimmed: Vec<LlmMessage> = Vec::new();
        let mut used = estimate_tokens(system_msg.content.as_str()) * 4;

        // Walk history newest-first, include until budget exhausted
        for (from_newest, msg) in history.iter().rev().enumerate() {
            let cost = estimate_tokens(msg.content.as_str()) * 4;
            if used + cost > char_budget {
                // The contract is "always preserves ... the most recent user
                // message": when even the NEWEST message alone busts the budget,
                // truncate it to the remaining room instead of dropping it and
                // sending the model a history-free prompt.
                if from_newest == 0 {
                    let remaining = char_budget.saturating_sub(used);
                    let text = msg.content.as_str();
                    let cut = floor_char_boundary(text, remaining);
                    if cut > 0 {
                        trimmed.push(LlmMessage {
                            role: msg.role.clone(),
                            content: crate::llm::MessageContent::text(&text[..cut]),
                        });
                    }
                }
                break;
            }
            used += cost;
            trimmed.push(msg.clone());
        }
        trimmed.reverse();

        let mut result = vec![system_msg];
        result.extend(trimmed);
        result
    }

    /// Focused system prompt for each agent role.
    pub fn system_prompt_for_role(role: AgentRole) -> &'static str {
        match role {
            AgentRole::Observer => {
                "You are an observer agent. Your sole task is to read and describe the current \
                 screen state using ax_tree_query and capture_live_screenshot. Report exactly \
                 what you see: UI elements, text, application state. Be factual and concise."
            }
            AgentRole::Executor => {
                "You are an executor agent. Your sole task is to perform UI actions precisely \
                 as instructed. Use ax_click, ax_type, ax_hotkey, ax_scroll, ax_focus_app, and \
                 ax_wait. After each action, verify the result. Abort if safety is uncertain."
            }
            AgentRole::MemoryManager => {
                "You are a memory manager agent. Retrieve information from memory stores using \
                 get_knowledge, search_hybrid, get_directives, and search_summaries. Synthesize \
                 the most relevant results for the given question."
            }
            AgentRole::LearningEngine => {
                "You are a learning engine agent. Your task is to observe user actions and \
                 synthesize them into reusable procedures. Identify parameterizable patterns \
                 and generalize specific values into placeholders."
            }
            AgentRole::SafetyMonitor => {
                "You are a safety monitor agent. Evaluate proposed actions for risk. Flag \
                 anything that could cause data loss, send messages, make purchases, or \
                 modify system settings. Provide a risk assessment: safe / needs_approval / blocked."
            }
            AgentRole::General => {
                "You are Shadow, a personal intelligence engine. You have continuous access \
                 to the user's screen, audio, and activity data. Help with questions about \
                 their work history, controlling their computer, managing memory, and \
                 automating tasks."
            }
        }
    }
}

/// Conservative token estimate: 4 characters per token.
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

/// Largest byte index `<= at` that sits on a UTF-8 char boundary of `text`.
/// (`str::floor_char_boundary` is still unstable.) Slicing at an arbitrary byte
/// index panics when a multi-byte char straddles it; every budget cut in this
/// module must go through here.
fn floor_char_boundary(text: &str, at: usize) -> usize {
    if at >= text.len() {
        return text.len();
    }
    let mut idx = at;
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_rounds_up_at_four_chars_per_token() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1); // (1+3)/4
        assert_eq!(estimate_tokens("abcd"), 1); // exactly 4
        assert_eq!(estimate_tokens("abcde"), 2); // (5+3)/4 = 2
        assert_eq!(estimate_tokens(&"x".repeat(400)), 100);
    }

    #[test]
    fn system_prompt_differs_per_role_and_is_nonempty() {
        let roles = [
            AgentRole::Observer,
            AgentRole::Executor,
            AgentRole::MemoryManager,
            AgentRole::LearningEngine,
            AgentRole::SafetyMonitor,
            AgentRole::General,
        ];
        let prompts: Vec<&str> = roles
            .iter()
            .map(|r| ContextBudgetManager::system_prompt_for_role(*r))
            .collect();
        for p in &prompts {
            assert!(!p.is_empty());
        }
        // Executor and Observer must not share the same prompt.
        assert_ne!(
            ContextBudgetManager::system_prompt_for_role(AgentRole::Observer),
            ContextBudgetManager::system_prompt_for_role(AgentRole::Executor)
        );
    }

    #[test]
    fn build_context_always_puts_system_message_first() {
        let history = [LlmMessage::user("hi"), LlmMessage::assistant("hello")];
        let out =
            ContextBudgetManager::build_context(AgentRole::General, ModelTier::Cloud, &history, "");
        assert_eq!(out[0].role, "system");
        // With empty memory, system message is just the role prompt (no "Context:").
        assert!(!out[0].content.as_str().contains("Context:"));
    }

    #[test]
    fn build_context_embeds_memory_into_system_message() {
        let out = ContextBudgetManager::build_context(
            AgentRole::General,
            ModelTier::Cloud,
            &[],
            "user prefers metric units",
        );
        assert_eq!(out.len(), 1); // just the system message, no history
        let sys = out[0].content.as_str();
        assert!(sys.contains("Context:"));
        assert!(sys.contains("user prefers metric units"));
    }

    #[test]
    fn build_context_preserves_small_history_in_order() {
        let history = [
            LlmMessage::user("first"),
            LlmMessage::assistant("second"),
            LlmMessage::user("third"),
        ];
        let out =
            ContextBudgetManager::build_context(AgentRole::General, ModelTier::Cloud, &history, "");
        // system + 3 history, original order preserved.
        assert_eq!(out.len(), 4);
        assert_eq!(out[1].content.as_str(), "first");
        assert_eq!(out[2].content.as_str(), "second");
        assert_eq!(out[3].content.as_str(), "third");
    }

    #[test]
    fn build_context_drops_oldest_when_over_budget() {
        // LocalSmall: max_per_turn=4000 → char_budget=16000. An oldest message of
        // 16000 chars won't fit once the newer small one is counted, so it's dropped.
        let huge = "a".repeat(16_000);
        let history = [LlmMessage::user(huge), LlmMessage::user("recent")];
        let out = ContextBudgetManager::build_context(
            AgentRole::Observer,
            ModelTier::LocalSmall,
            &history,
            "",
        );
        // Only the recent message survives alongside the system prompt.
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].content.as_str(), "recent");
    }

    #[test]
    fn build_context_truncates_a_single_oversized_recent_message() {
        // The doc contract: the most recent user message is always preserved. A
        // lone message exceeding the per-turn char budget is TRUNCATED to the
        // remaining room, never dropped (a history-free prompt would make the
        // model answer from nothing).
        let huge = "b".repeat(40_000);
        let history = [LlmMessage::user(huge)];
        let out = ContextBudgetManager::build_context(
            AgentRole::Observer,
            ModelTier::LocalSmall,
            &history,
            "",
        );
        assert_eq!(
            out.len(),
            2,
            "system message + the truncated recent message"
        );
        assert_eq!(out[1].role, "user");
        let kept = out[1].content.as_str();
        assert!(!kept.is_empty());
        assert!(kept.len() < 40_000, "must actually be truncated");
        assert!(kept.chars().all(|c| c == 'b'));
    }

    #[test]
    fn build_context_truncates_memory_to_tier_budget_ascii() {
        // Cloud memory budget = 4000 tokens → 16000 chars. A 20000-char ASCII memory
        // is sliced to 16000 bytes (safe: ASCII bytes are all char boundaries).
        let mem = "m".repeat(20_000);
        let out =
            ContextBudgetManager::build_context(AgentRole::General, ModelTier::Cloud, &[], &mem);
        let sys = out[0].content.as_str();
        // sys = system_prompt + "\n\nContext:\n" + (16000-byte memory slice).
        let prompt_len = ContextBudgetManager::system_prompt_for_role(AgentRole::General).len();
        assert_eq!(sys.len(), prompt_len + "\n\nContext:\n".len() + 16_000);
    }

    #[test]
    fn build_context_multibyte_memory_truncates_on_a_char_boundary() {
        // Regression: memory truncation used a raw byte slice at `budget.memory*4`
        // and PANICKED when a multi-byte char straddled the cut (LocalSmall cuts at
        // byte 4000; 3-byte '€' chars put a boundary violation there). The cut now
        // snaps down to a char boundary.
        let mem = "\u{20AC}".repeat(2_000); // 2000 * 3 bytes = 6000 bytes
        let out = ContextBudgetManager::build_context(
            AgentRole::General,
            ModelTier::LocalSmall,
            &[],
            &mem,
        );
        let sys = out[0].content.as_str();
        assert!(sys.contains("Context:"));
        // 4000 is not divisible by 3 → snapped to 3999 bytes = 1333 whole chars.
        let kept_euros = sys.chars().filter(|c| *c == '\u{20AC}').count();
        assert_eq!(kept_euros, 1333);
    }
}
