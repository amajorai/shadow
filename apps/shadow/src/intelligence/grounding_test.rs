//! Tests for coordinate parsing and accessibility grounding.

use super::*;

#[test]
fn parses_bracket_format() {
    assert_eq!(parse_coordinates("point [0.5, 0.25]"), Some((0.5, 0.25)));
}

#[test]
fn parses_paren_format() {
    assert_eq!(parse_coordinates("(12, 34)"), Some((12.0, 34.0)));
}

#[test]
fn parses_click_function_format() {
    assert_eq!(
        parse_coordinates("call click(x=100, y=200) now"),
        Some((100.0, 200.0))
    );
}

#[test]
fn returns_none_when_no_coordinate_present() {
    assert_eq!(parse_coordinates("no coordinates here"), None);
}

fn snapshot_with(children: Vec<ghost_eyes::AXTreeNode>) -> ghost_eyes::AXSnapshot {
    let root = ghost_eyes::AXTreeNode {
        role: "AXWindow".to_string(),
        title: Some("Login".to_string()),
        children,
        ..ghost_eyes::AXTreeNode::default()
    };
    ghost_eyes::AXSnapshot {
        root,
        pid: 42,
        app_name: Some("App".to_string()),
        window_title: Some("Login".to_string()),
        complete: true,
        stats: ghost_eyes::WalkStats::default(),
        renderer: ghost_eyes::RendererA11y::NotApplicable,
    }
}

fn leaf(role: &str, title: &str, x: i32) -> ghost_eyes::AXTreeNode {
    ghost_eyes::AXTreeNode {
        role: role.to_string(),
        title: Some(title.to_string()),
        bounds: Some(crate::capture::Bounds {
            x,
            y: 0,
            width: 20,
            height: 10,
        }),
        ..ghost_eyes::AXTreeNode::default()
    }
}

#[test]
fn exact_ax_hit_carries_a_pinned_locator_and_centre() {
    let snap = snapshot_with(vec![
        leaf("AXStaticText", "Hi", 0),
        leaf("AXButton", "Sign in", 100),
    ]);
    let hit = try_ax_exact(&snap, "sign in", (1000, 1000)).expect("hit");
    let target = hit.ax_target.expect("ax target");
    assert_eq!(target.locator.path, vec![1]);
    assert_eq!(target.locator.pid, 42);
    assert_eq!(
        target.locator.window,
        ghost_eyes::WindowTarget::PidTitle {
            pid: 42,
            title: "Login".to_string()
        }
    );
    assert_eq!(target.center, (110, 5));
    assert!(serde_json::to_string(&GroundingResult {
        ax_target: None,
        ..hit
    })
    .is_ok());
}

#[test]
fn fuzzy_ax_match_never_reads_a_secure_value() {
    let mut secret = leaf("AXSecureTextField", "Password", 0);
    secret.secure = true;
    secret.value = Some("hunter2".to_string());
    let snap = snapshot_with(vec![secret]);
    assert!(try_ax_fuzzy(&snap, "hunter2", (1000, 1000)).is_none());
    assert!(try_ax_exact(&snap, "hunter2", (1000, 1000)).is_none());
}

#[test]
fn bracket_takes_precedence_over_paren() {
    // Both present: the bracket pattern is tried first.
    assert_eq!(parse_coordinates("[1, 2] and (3, 4)"), Some((1.0, 2.0)));
}
