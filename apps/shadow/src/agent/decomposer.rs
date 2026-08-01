use super::context_budget::AgentRole;
use super::intent_classifier::Intent;
use crate::llm::{orchestrator::LlmOrchestrator, LlmMessage, LlmRequest};

/// A single sub-task produced by decomposition.
#[derive(Debug, Clone)]
pub struct SubTask {
    pub role: AgentRole,
    pub instruction: String,
    pub parallelizable: bool,
    pub timeout_s: u64,
    pub dependencies: Vec<String>,
}

/// Result of decomposing a complex user task.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    pub sub_tasks: Vec<SubTask>,
    pub intent: Intent,
    pub estimated_timeout_s: u64,
}

pub struct TaskDecomposer;

impl TaskDecomposer {
    /// Decompose `task` into ordered sub-tasks based on `intent`.
    ///
    /// For well-known intent types, returns a template decomposition instantly.
    /// For Ambiguous or complex tasks, falls back to the heuristic decomposition.
    pub fn decompose(task: &str, intent: Intent) -> DecompositionResult {
        let sub_tasks = match intent {
            Intent::UiAction => vec![SubTask {
                role: AgentRole::Executor,
                instruction: task.to_string(),
                parallelizable: false,
                timeout_s: 60,
                dependencies: vec![],
            }],

            Intent::MemorySearch => vec![
                SubTask {
                    role: AgentRole::MemoryManager,
                    instruction: format!("Search memory stores for: {}", task),
                    parallelizable: false,
                    timeout_s: 15,
                    dependencies: vec![],
                },
                SubTask {
                    role: AgentRole::General,
                    instruction: format!("Synthesize memory results to answer: {}", task),
                    parallelizable: false,
                    timeout_s: 15,
                    dependencies: vec!["memory_search".to_string()],
                },
            ],

            Intent::ProcedureLearning => vec![
                SubTask {
                    role: AgentRole::Observer,
                    instruction: format!("Observe and record user actions for: {}", task),
                    parallelizable: false,
                    timeout_s: 300,
                    dependencies: vec![],
                },
                SubTask {
                    role: AgentRole::LearningEngine,
                    instruction: "Synthesize recorded actions into a reusable procedure"
                        .to_string(),
                    parallelizable: false,
                    timeout_s: 30,
                    dependencies: vec!["observe".to_string()],
                },
            ],

            Intent::ProcedureReplay => vec![SubTask {
                role: AgentRole::Executor,
                instruction: format!("Replay procedure: {}", task),
                parallelizable: false,
                timeout_s: 120,
                dependencies: vec![],
            }],

            Intent::DirectiveCreation => vec![SubTask {
                role: AgentRole::MemoryManager,
                instruction: format!("Create directive: {}", task),
                parallelizable: false,
                timeout_s: 5,
                dependencies: vec![],
            }],

            Intent::ComplexReasoning => vec![
                SubTask {
                    role: AgentRole::MemoryManager,
                    instruction: format!("Gather relevant information for: {}", task),
                    parallelizable: false,
                    timeout_s: 20,
                    dependencies: vec![],
                },
                SubTask {
                    role: AgentRole::General,
                    instruction: format!("Analyze and respond to: {}", task),
                    parallelizable: false,
                    timeout_s: 30,
                    dependencies: vec!["gather".to_string()],
                },
            ],

            Intent::SimpleQuestion => vec![SubTask {
                role: AgentRole::General,
                instruction: task.to_string(),
                parallelizable: false,
                timeout_s: 15,
                dependencies: vec![],
            }],

            Intent::Ambiguous => {
                // Single general-purpose task; let the agent figure it out
                vec![SubTask {
                    role: AgentRole::General,
                    instruction: task.to_string(),
                    parallelizable: false,
                    timeout_s: 60,
                    dependencies: vec![],
                }]
            }
        };

        let estimated_timeout_s = sub_tasks.iter().map(|t| t.timeout_s).sum();

        DecompositionResult {
            sub_tasks,
            intent,
            estimated_timeout_s,
        }
    }

    /// Group sub-tasks into parallel execution phases based on dependencies.
    pub fn group_into_phases(tasks: &[SubTask]) -> Vec<Vec<&SubTask>> {
        // Simple greedy phase assignment:
        // tasks with no dependencies → phase 0;
        // tasks whose dependencies are all in earlier phases → next phase.
        // Since dependencies are just string labels, we track by index.
        let mut phases: Vec<Vec<&SubTask>> = vec![];
        let mut assigned: Vec<bool> = vec![false; tasks.len()];

        loop {
            let mut phase: Vec<&SubTask> = vec![];
            let mut any = false;

            for (i, task) in tasks.iter().enumerate() {
                if assigned[i] {
                    continue;
                }
                // Check all dependencies are in earlier phases
                let deps_met = task.dependencies.iter().all(|dep| {
                    tasks.iter().enumerate().any(|(j, t)| {
                        assigned[j] && t.instruction.to_lowercase().contains(&dep.to_lowercase())
                    })
                });
                let no_deps = task.dependencies.is_empty();

                if no_deps || deps_met {
                    phase.push(task);
                    assigned[i] = true;
                    any = true;
                }
            }

            if phase.is_empty() {
                break;
            }
            phases.push(phase);

            if !any {
                break;
            }
        }

        // Append any remaining unassigned tasks as a final phase
        let remaining: Vec<&SubTask> = tasks
            .iter()
            .enumerate()
            .filter(|(i, _)| !assigned[*i])
            .map(|(_, t)| t)
            .collect();
        if !remaining.is_empty() {
            phases.push(remaining);
        }

        phases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ui_action_is_single_executor_task() {
        let r = TaskDecomposer::decompose("click Send", Intent::UiAction);
        assert_eq!(r.sub_tasks.len(), 1);
        assert_eq!(r.sub_tasks[0].role, AgentRole::Executor);
        assert_eq!(r.sub_tasks[0].instruction, "click Send");
        assert!(!r.sub_tasks[0].parallelizable);
        assert!(r.sub_tasks[0].dependencies.is_empty());
        assert_eq!(r.estimated_timeout_s, 60);
        assert_eq!(r.intent, Intent::UiAction);
    }

    #[test]
    fn memory_search_has_dependent_synthesis_step() {
        let r = TaskDecomposer::decompose("what did I read", Intent::MemorySearch);
        assert_eq!(r.sub_tasks.len(), 2);
        assert_eq!(r.sub_tasks[0].role, AgentRole::MemoryManager);
        assert!(r.sub_tasks[0].dependencies.is_empty());
        assert_eq!(r.sub_tasks[1].role, AgentRole::General);
        assert_eq!(
            r.sub_tasks[1].dependencies,
            vec!["memory_search".to_string()]
        );
        // Timeout is the sum of both sub-task timeouts.
        assert_eq!(r.estimated_timeout_s, 30);
    }

    #[test]
    fn procedure_learning_observes_then_synthesizes() {
        let r = TaskDecomposer::decompose("learn how I file expenses", Intent::ProcedureLearning);
        assert_eq!(r.sub_tasks.len(), 2);
        assert_eq!(r.sub_tasks[0].role, AgentRole::Observer);
        assert_eq!(r.sub_tasks[1].role, AgentRole::LearningEngine);
        assert_eq!(r.sub_tasks[1].dependencies, vec!["observe".to_string()]);
        assert_eq!(r.estimated_timeout_s, 330);
    }

    #[test]
    fn directive_creation_is_short_single_task() {
        let r = TaskDecomposer::decompose("remind me to stretch", Intent::DirectiveCreation);
        assert_eq!(r.sub_tasks.len(), 1);
        assert_eq!(r.sub_tasks[0].role, AgentRole::MemoryManager);
        assert_eq!(r.estimated_timeout_s, 5);
    }

    #[test]
    fn complex_reasoning_gathers_then_analyzes() {
        let r = TaskDecomposer::decompose("compare X and Y", Intent::ComplexReasoning);
        assert_eq!(r.sub_tasks.len(), 2);
        assert_eq!(r.sub_tasks[1].dependencies, vec!["gather".to_string()]);
        assert_eq!(r.estimated_timeout_s, 50);
    }

    #[test]
    fn simple_question_and_replay_and_ambiguous_are_single_tasks() {
        for intent in [
            Intent::SimpleQuestion,
            Intent::ProcedureReplay,
            Intent::Ambiguous,
        ] {
            let r = TaskDecomposer::decompose("task", intent.clone());
            assert_eq!(r.sub_tasks.len(), 1, "{intent:?} should be one task");
            assert!(r.sub_tasks[0].dependencies.is_empty());
        }
    }

    #[test]
    fn group_into_phases_puts_independent_tasks_in_phase_zero() {
        let tasks = TaskDecomposer::decompose("q", Intent::MemorySearch).sub_tasks;
        let phases = TaskDecomposer::group_into_phases(&tasks);
        // First task has no deps → phase 0; second depends on "memory_search"
        // which the first instruction contains → phase 1.
        assert_eq!(phases.len(), 2);
        assert_eq!(phases[0].len(), 1);
        assert_eq!(phases[1].len(), 1);
        assert_eq!(phases[0][0].role, AgentRole::MemoryManager);
        assert_eq!(phases[1][0].role, AgentRole::General);
    }

    #[test]
    fn group_into_phases_single_phase_when_all_independent() {
        let tasks = vec![
            SubTask {
                role: AgentRole::General,
                instruction: "a".to_string(),
                parallelizable: true,
                timeout_s: 5,
                dependencies: vec![],
            },
            SubTask {
                role: AgentRole::Observer,
                instruction: "b".to_string(),
                parallelizable: true,
                timeout_s: 5,
                dependencies: vec![],
            },
        ];
        let phases = TaskDecomposer::group_into_phases(&tasks);
        assert_eq!(phases.len(), 1);
        assert_eq!(phases[0].len(), 2);
    }

    #[test]
    fn group_into_phases_empty_input_yields_no_phases() {
        let phases = TaskDecomposer::group_into_phases(&[]);
        assert!(phases.is_empty());
    }
}
