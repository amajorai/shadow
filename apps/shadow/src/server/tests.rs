//! Tests for the Shadow HTTP server, one file per area.

use super::*;

mod capture_policy;
mod context_capture;
mod handlers;
mod retention;
mod service_handlers;
mod weekly;

// Fixtures shared by more than one area.

/// The shared capture policy is process-global and the test
/// harness runs tests on parallel threads, so every test that mutates or
/// observes them must hold this lock or they race each other flakily.
/// `unwrap_or_else(PoisonError::into_inner)` keeps one panicking test from
/// cascading poison into the rest.
static CAPTURE_STATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn capture_state_guard() -> std::sync::MutexGuard<'static, ()> {
    CAPTURE_STATE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Build a minimal AppState with all optional fields set to None.
/// The handler must return a well-formed response even with no capture subsystems.
fn minimal_state() -> AppState {
    AppState {
        config: crate::config::Config::new(),
        computer_use: Arc::new(std::sync::RwLock::new(
            crate::config::ComputerUseConfig::default(),
        )),
        computer_use_lock: Arc::new(tokio::sync::Mutex::new(())),
        orchestrator: None,
        procedure_store: None,
        proactive_store: None,
        summary_store: None,
        mimicry: None,
        pattern_store: None,
        trust_tuner: None,
        delivery_manager: None,
        summary_queue: None,
        window_tracker: None,
        ax_tree: None,
    }
}
