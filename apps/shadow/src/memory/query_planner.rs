use anyhow::Result;

use crate::intelligence::context::{EpisodeRecord, EpisodeStore};
use crate::llm::{orchestrator::LlmOrchestrator, LlmMessage, LlmRequest};
use crate::memory::directive::DirectiveMemoryStore;
use crate::memory::semantic::{MemoryEntry, SemanticMemoryStore};
use crate::mimicry::procedure::ProcedureStore;
use crate::mimicry::types::ProcedureTemplate;

/// Which memory sources to query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemorySource {
    SemanticKnowledge,
    Directives,
    Episodes,
    Procedures,
}

/// A plan for a multi-source memory query.
pub struct QueryPlan {
    pub sources: Vec<MemorySource>,
    pub query: String,
    pub max_chars: usize,
}

/// A normalized result from any memory source.
#[derive(Debug, Clone)]
pub struct MemoryResult {
    pub source: MemorySource,
    pub content: String,
    pub confidence: f32,
}

pub struct MemoryQueryPlanner;

impl MemoryQueryPlanner {
    /// Decide which sources to query for the given question.
    /// Uses the LLM when available; falls back to keyword heuristics.
    pub async fn plan(question: &str, orchestrator: &LlmOrchestrator) -> QueryPlan {
        // Try LLM planning with the local/fast provider
        if orchestrator.local().is_some() {
            if let Some(plan) = llm_plan(question, orchestrator).await {
                return plan;
            }
        }
        heuristic_plan(question)
    }

    /// Execute the plan against the given stores.
    pub fn execute(
        plan: &QueryPlan,
        semantic: Option<&SemanticMemoryStore>,
        directive: Option<&DirectiveMemoryStore>,
        episodes: Option<&EpisodeStore>,
        procedures: Option<&ProcedureStore>,
    ) -> Vec<MemoryResult> {
        let mut results = Vec::new();

        for source in &plan.sources {
            match source {
                MemorySource::SemanticKnowledge => {
                    if let Some(store) = semantic {
                        if let Ok(entries) = store.query(None, &plan.query) {
                            for e in entries.iter().take(5) {
                                results.push(MemoryResult {
                                    source: MemorySource::SemanticKnowledge,
                                    content: format!("[{}] {}", e.category, e.content),
                                    confidence: e.confidence,
                                });
                            }
                        }
                    }
                }
                MemorySource::Directives => {
                    if let Some(store) = directive {
                        if let Ok(dirs) = store.list_active(None) {
                            for d in dirs.iter().take(5) {
                                results.push(MemoryResult {
                                    source: MemorySource::Directives,
                                    content: format!("[{}] {}", d.directive_type, d.content),
                                    confidence: 1.0,
                                });
                            }
                        }
                    }
                }
                MemorySource::Episodes => {
                    if let Some(store) = episodes {
                        if let Ok(eps) = store.load_recent(5) {
                            for ep in eps {
                                results.push(MemoryResult {
                                    source: MemorySource::Episodes,
                                    content: format!("{}: {}", ep.app_name, ep.summary),
                                    confidence: 0.8,
                                });
                            }
                        }
                    }
                }
                MemorySource::Procedures => {
                    if let Some(store) = procedures {
                        if let Ok(procs) = store.find_similar(&plan.query, 5) {
                            for p in procs {
                                results.push(MemoryResult {
                                    source: MemorySource::Procedures,
                                    content: format!("[procedure] {}: {}", p.name, p.description),
                                    confidence: 0.9,
                                });
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Format results as a text block for agent prompt injection.
    /// Bounded by `max_chars`.
    pub fn format_for_context(results: &[MemoryResult], max_chars: usize) -> String {
        if results.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        for r in results {
            let line = format!("• {}\n", r.content);
            if out.len() + line.len() > max_chars {
                break;
            }
            out.push_str(&line);
        }
        out
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

async fn llm_plan(question: &str, orchestrator: &LlmOrchestrator) -> Option<QueryPlan> {
    let prompt = format!(
        "Which memory stores should be queried to answer this question?\n\
         Available stores: semantic_knowledge | directives | episodes | procedures\n\
         Respond with a comma-separated list of store names, nothing else.\n\
         Question: {}",
        question
    );
    let resp = orchestrator
        .generate(LlmRequest {
            messages: vec![LlmMessage::user(prompt)],
            temperature: 0.0,
            max_tokens: 20,
            ..Default::default()
        })
        .await
        .ok()?;

    let text = resp.content?;
    let sources: Vec<MemorySource> = text
        .split(',')
        .filter_map(|s| match s.trim().to_lowercase().as_str() {
            "semantic_knowledge" => Some(MemorySource::SemanticKnowledge),
            "directives" => Some(MemorySource::Directives),
            "episodes" => Some(MemorySource::Episodes),
            "procedures" => Some(MemorySource::Procedures),
            _ => None,
        })
        .collect();

    if sources.is_empty() {
        return None;
    }
    Some(QueryPlan {
        sources,
        query: question.to_string(),
        max_chars: 4000,
    })
}

fn heuristic_plan(question: &str) -> QueryPlan {
    let q = question.to_lowercase();
    let mut sources = Vec::new();

    if q.contains("remind") || q.contains("directive") || q.contains("rule") || q.contains("always")
    {
        sources.push(MemorySource::Directives);
    }
    if q.contains("procedure")
        || q.contains("replay")
        || q.contains("workflow")
        || q.contains("how to")
    {
        sources.push(MemorySource::Procedures);
    }
    if q.contains("remember")
        || q.contains("history")
        || q.contains("transcript")
        || q.contains("did i")
    {
        sources.push(MemorySource::Episodes);
        sources.push(MemorySource::SemanticKnowledge);
    }
    if sources.is_empty() {
        // Default: search all
        sources = vec![
            MemorySource::SemanticKnowledge,
            MemorySource::Directives,
            MemorySource::Episodes,
            MemorySource::Procedures,
        ];
    }

    QueryPlan {
        sources,
        query: question.to_string(),
        max_chars: 4000,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(source: MemorySource, content: &str) -> MemoryResult {
        MemoryResult {
            source,
            content: content.to_string(),
            confidence: 1.0,
        }
    }

    #[test]
    fn heuristic_plan_routes_directives() {
        let plan = heuristic_plan("remind me to call back");
        assert_eq!(plan.sources, vec![MemorySource::Directives]);
        assert_eq!(plan.query, "remind me to call back");
        assert_eq!(plan.max_chars, 4000);
    }

    #[test]
    fn heuristic_plan_routes_procedures() {
        let plan = heuristic_plan("how to export the report");
        assert_eq!(plan.sources, vec![MemorySource::Procedures]);
    }

    #[test]
    fn heuristic_plan_episodes_pushes_semantic_too() {
        let plan = heuristic_plan("what did i do in my history");
        assert_eq!(
            plan.sources,
            vec![MemorySource::Episodes, MemorySource::SemanticKnowledge]
        );
    }

    #[test]
    fn heuristic_plan_combines_multiple_intents_in_order() {
        // "remind" → Directives (pushed first), "how to" → Procedures (second).
        let plan = heuristic_plan("remind me how to reply");
        assert_eq!(
            plan.sources,
            vec![MemorySource::Directives, MemorySource::Procedures]
        );
    }

    #[test]
    fn heuristic_plan_defaults_to_all_sources() {
        let plan = heuristic_plan("banana pancakes");
        assert_eq!(
            plan.sources,
            vec![
                MemorySource::SemanticKnowledge,
                MemorySource::Directives,
                MemorySource::Episodes,
                MemorySource::Procedures,
            ]
        );
    }

    #[test]
    fn format_for_context_empty_when_no_results() {
        assert_eq!(MemoryQueryPlanner::format_for_context(&[], 1000), "");
    }

    #[test]
    fn format_for_context_bullets_each_result() {
        let results = [
            result(MemorySource::Directives, "always confirm"),
            result(MemorySource::Episodes, "opened Mail"),
        ];
        let out = MemoryQueryPlanner::format_for_context(&results, 1000);
        assert_eq!(out, "• always confirm\n• opened Mail\n");
    }

    #[test]
    fn format_for_context_stops_at_char_budget() {
        // Each line is "• " + content + "\n". First line ~ 12 chars; a tiny budget
        // admits only the first before the second would overflow.
        let results = [
            result(MemorySource::Directives, "one"),
            result(MemorySource::Directives, "two"),
        ];
        let out = MemoryQueryPlanner::format_for_context(&results, 8);
        assert_eq!(out, "• one\n");
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("shadow-qp-{tag}-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn execute_gathers_from_all_four_sources() {
        use crate::memory::directive::Directive;
        use crate::mimicry::types::{ProcedureStep, StepFailureAction};

        // Semantic store with one matching entry.
        let sem_path = temp_path("sem");
        let semantic = SemanticMemoryStore::new(&sem_path).unwrap();
        semantic
            .upsert(&MemoryEntry {
                id: "s1".to_string(),
                category: "preference".to_string(),
                content: "likes dark theme".to_string(),
                confidence: 0.9,
                source_episode_id: None,
                access_count: 0,
                last_accessed: 0,
                created_at: 1,
            })
            .unwrap();

        // Directive store with one active directive.
        let dir_path = temp_path("dir");
        let directive = DirectiveMemoryStore::new(&dir_path).unwrap();
        directive
            .create(&Directive {
                id: "d1".to_string(),
                directive_type: "reminder".to_string(),
                content: "call back".to_string(),
                trigger_pattern: None,
                action: None,
                priority: 5,
                expires_at: None,
                created_at: 1,
            })
            .unwrap();

        // Episode store with one episode.
        let ep_path = temp_path("ep");
        let episodes = EpisodeStore::new(&ep_path).unwrap();
        episodes
            .save(&EpisodeRecord {
                id: "e1".to_string(),
                start_us: 1,
                end_us: 2,
                app_name: "Mail".to_string(),
                window_title: "Inbox".to_string(),
                actions: vec![],
                summary: "read email".to_string(),
                bundle_id: None,
            })
            .unwrap();

        // Procedure store with one template.
        let proc_path = temp_path("proc");
        let procedures = ProcedureStore::new(&proc_path).unwrap();
        procedures
            .save(&ProcedureTemplate {
                id: "p1".to_string(),
                name: "Send report".to_string(),
                app_name: "Mail".to_string(),
                description: "compose and send".to_string(),
                steps: vec![ProcedureStep {
                    step_number: 1,
                    description: "click".to_string(),
                    tool_name: "ax_click".to_string(),
                    tool_args: serde_json::json!({}),
                    verification: None,
                    on_failure: StepFailureAction::Abort,
                }],
                preconditions: vec![],
                success_count: 1,
                failure_count: 0,
                last_used: 1,
                created_at: 1,
            })
            .unwrap();

        let plan = QueryPlan {
            sources: vec![
                MemorySource::SemanticKnowledge,
                MemorySource::Directives,
                MemorySource::Episodes,
                MemorySource::Procedures,
            ],
            query: "report".to_string(),
            max_chars: 4000,
        };

        let results = MemoryQueryPlanner::execute(
            &plan,
            Some(&semantic),
            Some(&directive),
            Some(&episodes),
            Some(&procedures),
        );

        // Directives (unconditional) + Episodes (load_recent) always yield;
        // semantic depends on text match ("report" won't match "dark theme"),
        // procedures depends on find_similar. At minimum directive + episode.
        assert!(results.iter().any(|r| r.source == MemorySource::Directives));
        assert!(results.iter().any(|r| r.source == MemorySource::Episodes));

        for p in [&sem_path, &dir_path, &ep_path, &proc_path] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn execute_with_no_stores_returns_empty() {
        let plan = QueryPlan {
            sources: vec![MemorySource::SemanticKnowledge, MemorySource::Procedures],
            query: "x".to_string(),
            max_chars: 100,
        };
        let results = MemoryQueryPlanner::execute(&plan, None, None, None, None);
        assert!(results.is_empty());
    }
}
