//! Reads the focused window and its accessibility tree for one planning step.

use std::time::Duration;

use ghost_eyes::{ErrorCode, GhostError, SnapshotOptions, WalkLimits, WindowTarget};
use serde_json::json;

use super::effector::Effector;
use super::WindowState;
use crate::capture::WindowTracker;
use crate::server::AppState;

/// Computer-use observation limits: deeper than passive capture, with value reads.
pub(super) const OBSERVE_LIMITS: WalkLimits = WalkLimits {
    max_depth: 8,
    max_children: 100,
    max_nodes: 3000,
    budget: Duration::from_secs(3),
    read_values: true,
};

/// The window a snapshot or action targets: that process, or the focused window
/// when the tracker could not tell the pid.
pub(super) fn window_target(pid: i32) -> WindowTarget {
    if pid > 0 {
        WindowTarget::Pid { pid }
    } else {
        WindowTarget::Focused
    }
}

pub(super) async fn observe<E: Effector>(
    state: &AppState,
    fx: &E,
) -> Result<WindowState, GhostError> {
    let (app_name, title, pid) = if let Some(tracker) = &state.window_tracker {
        let tracker = tracker.lock().await;
        match WindowTracker::get_active_window(&*tracker).await {
            Some(window) => (window.app_name, window.title, window.pid),
            None => (String::new(), String::new(), 0),
        }
    } else {
        (String::new(), String::new(), 0)
    };
    if app_name.is_empty() && title.is_empty() {
        return Err(GhostError::new(
            ErrorCode::AppNotFound,
            "Shadow could not identify the focused application window",
        ));
    }
    if !crate::server::is_capture_allowed(&app_name) {
        return Err(GhostError::policy_denied(
            "the focused application is outside Shadow's capture allowlist",
        ));
    }
    observe_window(fx, app_name, title, pid).await
}

/// Snapshots the tracker's window by pid and refuses a snapshot of another process.
pub(super) async fn observe_window<E: Effector>(
    fx: &E,
    app_name: String,
    title: String,
    pid: i32,
) -> Result<WindowState, GhostError> {
    let snapshot = fx
        .snapshot(SnapshotOptions {
            target: window_target(pid),
            limits: OBSERVE_LIMITS,
            activate_renderer: true,
        })
        .await?;
    if pid > 0 && snapshot.pid != pid {
        return Err(GhostError::stale_ref(
            "the accessibility snapshot belongs to a different process",
        )
        .with_details(json!({
            "reason": "window_changed",
            "window_pid": pid,
            "snapshot_pid": snapshot.pid,
        })));
    }
    Ok(WindowState {
        app_name,
        title,
        pid: if pid > 0 { pid } else { snapshot.pid },
        complete: snapshot.complete,
        tree: snapshot.root,
    })
}

pub(super) fn page_changed(before: &WindowState, after: &WindowState) -> bool {
    before.title != after.title
        || serde_json::to_string(&before.tree).ok() != serde_json::to_string(&after.tree).ok()
}
