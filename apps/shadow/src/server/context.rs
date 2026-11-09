//! Current context, capture control, derived-history sweeps and accessibility text.

use super::*;

/// GET /context/current — snapshot of active window, selected text, and latest OCR frame.
///
/// Returns a well-formed empty payload when capture is paused or no data has been recorded yet;
/// never returns a 500 in steady state. When globally paused or the foreground app is not on the
/// allowlist, returns an empty payload with `paused: true` so clients can show the indicator.
pub(super) async fn current_context_handler(
    State(state): State<AppState>,
) -> Json<CurrentContextResponse> {
    use crate::capture::{AXTree, WindowTracker};

    let now_us = wall_micros();
    let globally_paused = is_capture_paused();

    // Early return when globally paused — do not query any capture subsystems.
    if globally_paused {
        return Json(CurrentContextResponse {
            timestamp_us: now_us,
            window_title: String::new(),
            app_name: String::new(),
            selected_text: String::new(),
            accessibility_text: String::new(),
            ocr_text: String::new(),
            ocr_timestamp_us: 0,
            capture_active: false,
            paused: true,
        });
    }

    // 1. Active window — live query via the platform window tracker.
    let (window_title, app_name) = if let Some(tracker) = &state.window_tracker {
        let t = tracker.lock().await;
        match WindowTracker::get_active_window(&*t).await {
            Some(win) => (win.title, win.app_name),
            None => (String::new(), String::new()),
        }
    } else {
        (String::new(), String::new())
    };

    // 1a. Check per-app allowlist: if the allowlist is non-empty and the foreground app is not
    //     listed, return the empty payload to suppress context for this app.
    if !app_name.is_empty() && !is_capture_allowed(&app_name) {
        return Json(CurrentContextResponse {
            timestamp_us: now_us,
            window_title: String::new(),
            app_name: String::new(),
            selected_text: String::new(),
            accessibility_text: String::new(),
            ocr_text: String::new(),
            ocr_timestamp_us: 0,
            capture_active: false,
            paused: false,
        });
    }

    // 2. Selected text — read from the AX tree's focused element value.
    //    We do a best-effort walk: if the focused tree has a non-empty value on the root
    //    or any direct child, treat it as the selection. Never fail hard.
    let accessibility_tree = if let Some(ax) = &state.ax_tree {
        let t = ax.lock().await;
        match AXTree::get_focused_tree(&*t).await {
            Ok(tree) => Some(tree),
            Err(_) => None,
        }
    } else {
        None
    };
    let selected_text = accessibility_tree
        .as_ref()
        .map(extract_selected_text)
        .unwrap_or_default();
    let accessibility_text = accessibility_tree
        .as_ref()
        .map(extract_accessibility_text)
        .unwrap_or_default();

    // 3. Latest OCR frame — read from the in-memory cache maintained by the
    //    screen-capture loop in capture_engine.rs. Zero overhead; no lock contention.
    let (ocr_text, ocr_timestamp_us) = crate::capture_engine::get_latest_ocr()
        .map(|(text, _app, ts)| (text, ts))
        .unwrap_or_else(|| (String::new(), 0));

    let capture_active = !window_title.is_empty() || !app_name.is_empty();

    Json(CurrentContextResponse {
        timestamp_us: now_us,
        window_title,
        app_name,
        selected_text,
        accessibility_text,
        ocr_text,
        ocr_timestamp_us,
        capture_active,
        paused: false,
    })
}

// ─── Capture control ───────────────────────────────────────────────────────────

/// GET /capture/control — read current pause state, allowlist, and frame toggle.
pub(super) async fn capture_control_get_handler() -> impl IntoResponse {
    match shadow_core::capture_control_snapshot() {
        Ok((policy, history_retention_days)) => Json(CaptureControlResponse {
            paused: policy.paused,
            app_allowlist: policy.app_allowlist,
            frames: policy.frames,
            history_retention_days,
        })
        .into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error":"could not read capture policy"})),
        )
            .into_response(),
    }
}

/// Omitted fields remain unchanged. Durable capture and retention state commits
/// atomically before publishing the accepted runtime state.
pub(super) async fn capture_control_post_handler(
    Json(req): Json<CaptureControlRequest>,
) -> impl IntoResponse {
    let retention_requested = req.history_retention_days.is_some();
    match shadow_core::update_capture_policy(req.paused, req.frames, req.app_allowlist, req.history_retention_days) {
        Ok((policy, history_retention_days)) => {
            // A shorter window takes effect now, not at the next hourly sweep.
            if retention_requested {
                tokio::spawn(run_history_sweep());
            }
            Json(CaptureControlResponse {
                paused: policy.paused, app_allowlist: policy.app_allowlist,
                frames: policy.frames, history_retention_days,
            }).into_response()
        }
        Err(shadow_core::ShadowError::CapturePolicyUninitialized) => (
            StatusCode::CONFLICT,
            Json(json!({"code":"capture_policy_uninitialized","error":"complete capture preferences are required before enabling capture"})),
        ).into_response(),
        Err(error) => {
            tracing::error!("could not persist Shadow capture policy: {error}");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error":"could not save capture preferences"}))).into_response()
        }
    }
}

/// Shadow's own stores derived from captured activity: 5-minute activity
/// episodes (app, window title, summary) and proactive suggestions built from
/// them. "Keep history" prunes them with shadow-core's timeline.
pub(crate) struct DerivedHistoryStores {
    pub episodes: Option<Arc<std::sync::Mutex<crate::intelligence::EpisodeStore>>>,
    pub proactive: Option<Arc<tokio::sync::Mutex<crate::intelligence::ProactiveStore>>>,
}

pub(super) static DERIVED_HISTORY: std::sync::OnceLock<DerivedHistoryStores> =
    std::sync::OnceLock::new();

/// Register the derived stores once at startup, before the first sweep.
pub(crate) fn register_derived_history(stores: DerivedHistoryStores) {
    let _ = DERIVED_HISTORY.set(stores);
}

/// Run one "Keep history" sweep: shadow-core's captured history, then the
/// derived stores, with the same cutoff. Used by the hourly loop and right
/// after `/capture/control` saves a retention change.
pub(crate) async fn run_history_sweep() {
    let outcome = tokio::task::spawn_blocking(shadow_core::run_history_retention_sweep).await;
    let cutoff_us = match &outcome {
        Ok(Ok(report)) => Some(report.cutoff_us),
        _ => None,
    };
    if let (Some(cutoff_us), Some(stores)) = (cutoff_us, DERIVED_HISTORY.get()) {
        prune_derived_history(stores, cutoff_us).await;
    }
    log_history_sweep(outcome);
}

/// Delete activity episodes and proactive suggestions from before `cutoff_us`.
/// A failure is logged and retried by the next sweep.
pub(crate) async fn prune_derived_history(stores: &DerivedHistoryStores, cutoff_us: u64) {
    if cutoff_us == 0 {
        return;
    }
    if let Some(episodes) = &stores.episodes {
        let pruned = match episodes.lock() {
            Ok(store) => store.delete_before(cutoff_us),
            Err(error) => Err(anyhow::anyhow!("episode store lock poisoned: {error}")),
        };
        match pruned {
            Ok(0) => {}
            Ok(n) => tracing::info!(episodes = n, "History retention: pruned activity episodes"),
            Err(error) => tracing::warn!("History retention will retry episodes: {error}"),
        }
    }
    if let Some(proactive) = &stores.proactive {
        match proactive.lock().await.delete_before(cutoff_us) {
            Ok(0) => {}
            Ok(n) => tracing::info!(suggestions = n, "History retention: pruned suggestions"),
            Err(error) => tracing::warn!("History retention will retry suggestions: {error}"),
        }
    }
}

/// Log one "Keep history" sweep: what it removed, and any part it must retry.
pub(crate) fn log_history_sweep(
    outcome: Result<
        Result<shadow_core::HistoryPruneReport, shadow_core::ShadowError>,
        tokio::task::JoinError,
    >,
) {
    match outcome {
        Ok(Ok(report)) => {
            if report.total_removed() > 0 {
                tracing::info!(
                    events = report.events_deleted,
                    search_docs = report.search_docs_deleted,
                    vectors = report.vector_entries_deleted,
                    keyframes = report.keyframes_deleted,
                    files = report.files_deleted,
                    bytes_freed = report.bytes_freed,
                    "History retention: pruned data older than the window"
                );
            }
            for error in &report.errors {
                tracing::warn!("History retention will retry: {error}");
            }
        }
        Ok(Err(error)) => tracing::warn!("History retention sweep failed: {error}"),
        Err(error) => tracing::warn!("History retention sweep did not finish: {error}"),
    }
}

/// Walk an AX tree node and return the first non-empty value string found,
/// preferring values on a focused/selected text field over generic containers.
pub fn extract_selected_text(node: &crate::capture::AXTreeNode) -> String {
    // Roles that are likely to carry typed/selected text content.
    let text_roles = [
        "text",
        "edit",
        "textfield",
        "combobox",
        "textarea",
        "document",
    ];
    let role_lower = node.role.to_lowercase();

    if text_roles.iter().any(|r| role_lower.contains(r)) {
        if let Some(val) = &node.value {
            if !val.is_empty() {
                return val.clone();
            }
        }
    }

    for child in &node.children {
        let found = extract_selected_text(child);
        if !found.is_empty() {
            return found;
        }
    }

    String::new()
}

pub(super) const MAX_ACCESSIBILITY_TEXT_BYTES: usize = 64 * 1024;

/// Flatten the focused accessibility tree into the text Appshots-like clients
/// can attach beside a screenshot. Titles and values are kept in tree order;
/// roles and identifiers stay out of the payload because they add machine
/// metadata rather than useful document text.
pub fn extract_accessibility_text(node: &crate::capture::AXTreeNode) -> String {
    fn append_line(output: &mut String, value: &str) -> bool {
        let value = value.trim();
        if value.is_empty() {
            return true;
        }
        let separator = if output.is_empty() { "" } else { "\n" };
        let available = MAX_ACCESSIBILITY_TEXT_BYTES
            .saturating_sub(output.len())
            .saturating_sub(separator.len());
        if available == 0 {
            return false;
        }
        let mut end = value.len().min(available);
        while end > 0 && !value.is_char_boundary(end) {
            end -= 1;
        }
        output.push_str(separator);
        output.push_str(&value[..end]);
        end == value.len()
    }

    fn walk(node: &crate::capture::AXTreeNode, output: &mut String) -> bool {
        let title = node.title.as_deref().unwrap_or_default();
        if !append_line(output, title) {
            return false;
        }
        let value = node.value.as_deref().unwrap_or_default();
        if value.trim() != title.trim() && !append_line(output, value) {
            return false;
        }
        for child in &node.children {
            if !walk(child, output) {
                return false;
            }
        }
        true
    }

    let mut output = String::new();
    let complete = walk(node, &mut output);
    if !complete {
        output.push_str("\n[Accessibility text truncated]");
    }
    output
}
