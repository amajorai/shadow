//! Candidate classification, planner-state redaction and target re-resolution.

use ghost_eyes::{AXTreeNode, Bounds, ErrorCode};
use serde_json::Value;

use super::super::candidates::{build_action_space, candidates};
use super::super::verify::verify_target;
use super::super::{WindowState, MAX_CANDIDATES};

fn bounds(x: i32, y: i32) -> Option<Bounds> {
    Some(Bounds {
        x,
        y,
        width: 80,
        height: 20,
    })
}

fn node(role: &str, title: Option<&str>, y: i32) -> AXTreeNode {
    AXTreeNode {
        role: role.to_owned(),
        title: title.map(str::to_owned),
        bounds: bounds(10, y),
        ..AXTreeNode::default()
    }
}

fn keyed(role: &str, role_key: &str, title: &str, y: i32) -> AXTreeNode {
    AXTreeNode {
        role_key: Some(role_key.to_owned()),
        ..node(role, Some(title), y)
    }
}

fn window(children: Vec<AXTreeNode>) -> WindowState {
    WindowState {
        app_name: "Editor".to_owned(),
        title: "Document".to_owned(),
        pid: 42,
        tree: AXTreeNode {
            role: "AXWindow".to_owned(),
            title: Some("Document".to_owned()),
            bounds: Some(Bounds {
                x: 0,
                y: 0,
                width: 800,
                height: 600,
            }),
            children,
            ..AXTreeNode::default()
        },
        complete: true,
    }
}

/// (role key, editable) per candidate, in walker order.
fn classes(children: Vec<AXTreeNode>) -> Vec<(String, bool)> {
    let (found, cut) = candidates(&window(children).tree);
    assert!(!cut);
    found.into_iter().map(|c| (c.role, c.editable)).collect()
}

fn owned(pairs: &[(&str, bool)]) -> Vec<(String, bool)> {
    pairs.iter().map(|(r, e)| ((*r).to_owned(), *e)).collect()
}

#[test]
fn classifies_the_macos_vocabulary() {
    let found = classes(vec![
        node("AXButton", Some("OK"), 10),
        node("AXTextField", Some("Name"), 40),
        node("AXSecureTextField", Some("Password"), 70),
        node("AXStaticText", Some("Hello"), 100),
        node("AXCheckBox", Some("Remember"), 130),
        node("AXLink", Some("Help"), 160),
    ]);
    assert_eq!(
        found,
        owned(&[
            ("button", false),
            ("textfield", true),
            ("textfield", true),
            ("checkbox", false),
            ("link", false),
        ])
    );
}

#[test]
fn classifies_the_uia_vocabulary_by_role_key_not_localized_role() {
    let found = classes(vec![
        keyed("Schaltfläche", "button", "OK", 10),
        keyed("Bearbeiten", "textfield", "Name", 40),
        keyed("Kombinationsfeld", "combobox", "Land", 70),
        keyed("Text", "statictext", "Hinweis", 100),
        keyed("Listenelement", "listitem", "Eins", 130),
    ]);
    assert_eq!(
        found,
        owned(&[
            ("button", false),
            ("textfield", true),
            ("combobox", true),
            ("listitem", false),
        ])
    );
}

#[test]
fn classifies_the_atspi_vocabulary() {
    let found = classes(vec![
        node("push button", Some("OK"), 10),
        node("password text", Some("Password"), 40),
        node("check box", Some("Remember"), 70),
        node("label", Some("Hello"), 100),
        node("menu item", Some("Open"), 130),
    ]);
    assert_eq!(
        found,
        owned(&[
            ("button", false),
            ("textfield", true),
            ("checkbox", false),
            ("menuitem", false),
        ])
    );
}

#[test]
fn secure_values_never_reach_the_planner_state_and_labels_never_use_values() {
    let secret = "hunter2-do-not-leak";
    let mut password = node("AXSecureTextField", Some("Password"), 40);
    // Defence in depth: even an unredacted secure value must not leak.
    password.value = Some(secret.to_owned());
    password.secure = true;
    let mut untitled = node("AXTextField", None, 70);
    untitled.value = Some("draft-value-text".to_owned());
    let observed = window(vec![password, untitled]);
    let (found, _) = candidates(&observed.tree);
    assert!(found[0].secure);
    assert_eq!(found[0].value, "");
    assert_eq!(found[1].label, "textfield");
    let (_, space) = build_action_space(&observed, &found, false, "log in", &[]);
    let wire = space.to_string();
    assert!(!wire.contains(secret));
    let elements = space["state"]["elements"].as_array().unwrap();
    assert_eq!(elements[0]["secure"], Value::Bool(true));
    assert_eq!(elements[0]["value"], "");
    assert_eq!(space["state"]["truncated"], Value::Bool(false));
}

#[test]
fn reports_truncation_when_the_candidate_cap_cuts_the_list() {
    let buttons = (0..MAX_CANDIDATES + 5)
        .map(|i| node("AXButton", Some(&format!("B{i}")), 10))
        .collect();
    let (found, cut) = candidates(&window(buttons).tree);
    assert_eq!(found.len(), MAX_CANDIDATES);
    assert!(cut);
}

#[test]
fn a_changed_value_does_not_make_the_target_stale() {
    let mut field = node("AXTextField", Some("Search"), 40);
    field.value = Some("old".to_owned());
    let before = window(vec![node("AXButton", Some("Go"), 10), field.clone()]);
    let (found, _) = candidates(&before.tree);
    field.value = Some("completely different".to_owned());
    let current = window(vec![node("AXButton", Some("Go"), 10), field]);
    assert_eq!(
        verify_target(&before, &current, &found[1]).unwrap(),
        vec![1]
    );
}

#[test]
fn two_buttons_with_the_same_name_are_ambiguous() {
    let before = window(vec![node("AXButton", Some("Save"), 10)]);
    let (found, _) = candidates(&before.tree);
    // The path now points at a different element and two "Save" buttons exist.
    let current = window(vec![
        node("AXStaticText", Some("Banner"), 0),
        node("AXButton", Some("Save"), 200),
        node("AXButton", Some("Save"), 300),
    ]);
    let error = verify_target(&before, &current, &found[0]).unwrap_err();
    assert_eq!(error.code, ErrorCode::AmbiguousTarget);
}

#[test]
fn a_different_window_process_is_stale() {
    let before = window(vec![node("AXButton", Some("Save"), 10)]);
    let (found, _) = candidates(&before.tree);
    let mut current = before.clone();
    current.pid = 43;
    let error = verify_target(&before, &current, &found[0]).unwrap_err();
    assert_eq!(error.code, ErrorCode::StaleRef);
}
