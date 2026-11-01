//! Re-resolves a planned target before input and computes its click point.

use ghost_eyes::accessibility::resolve::resolve_in_tree;
use ghost_eyes::{Bounds, GhostError};
use serde_json::json;

use super::{Candidate, WindowState};

/// Re-resolves `candidate` (chosen from `before`) in the fresh observation by its
/// identity, path first. Values are never compared, so a field whose text changed
/// is still the same target. Returns the element's current path.
pub(super) fn verify_target(
    before: &WindowState,
    current: &WindowState,
    candidate: &Candidate,
) -> Result<Vec<u32>, GhostError> {
    if before.pid != current.pid
        || before.app_name != current.app_name
        || before.title != current.title
    {
        return Err(GhostError::stale_ref(
            "the focused window changed while the planner was deciding; no action executed",
        )
        .with_details(json!({ "reason": "window_changed" })));
    }
    resolve_in_tree(&current.tree, &candidate.identity, &candidate.path)
        .map(|resolution| resolution.path)
}

/// The centre of `bounds`, refused when it falls outside the primary display.
pub(super) fn click_point(
    bounds: &Bounds,
    width: u32,
    height: u32,
) -> Result<(i32, i32), GhostError> {
    if width == 0 || height == 0 {
        return Err(GhostError::not_supported(
            "Shadow cannot determine the current display size",
        ));
    }
    let (x, y) = centre(bounds);
    let inside =
        x >= 0 && y >= 0 && i64::from(x) < i64::from(width) && i64::from(y) < i64::from(height);
    if !inside {
        return Err(GhostError::not_supported(
            "the accessibility target is outside the primary display",
        ));
    }
    Ok((x, y))
}

/// The centre of a rectangle.
pub(super) fn centre(bounds: &Bounds) -> (i32, i32) {
    let half_width = i32::try_from(bounds.width / 2).unwrap_or(i32::MAX);
    let half_height = i32::try_from(bounds.height / 2).unwrap_or(i32::MAX);
    (
        bounds.x.saturating_add(half_width),
        bounds.y.saturating_add(half_height),
    )
}
