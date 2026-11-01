//! Tests for the `ghost_eyes` capture adapter.

use super::*;

fn eyes_node(role: &str, title: Option<&str>, value: Option<&str>) -> ghost_eyes::AXTreeNode {
    ghost_eyes::AXTreeNode {
        role: role.to_string(),
        title: title.map(str::to_string),
        value: value.map(str::to_string),
        ..ghost_eyes::AXTreeNode::default()
    }
}

#[test]
fn conversion_keeps_the_six_fields_and_children_in_order() {
    let mut root = eyes_node("AXWindow", Some("Editor"), None);
    root.identifier = Some("main".to_string());
    root.bounds = Some(Bounds {
        x: 10,
        y: 20,
        width: 300,
        height: 200,
    });
    root.children = vec![
        eyes_node("AXButton", Some("Save"), None),
        eyes_node("AXTextField", Some("Name"), Some("Ada")),
    ];

    let node = AXTreeNode::from(&root);
    assert_eq!(node.role, "AXWindow");
    assert_eq!(node.title.as_deref(), Some("Editor"));
    assert_eq!(node.identifier.as_deref(), Some("main"));
    assert_eq!(
        node.bounds,
        Some(Bounds {
            x: 10,
            y: 20,
            width: 300,
            height: 200
        })
    );
    assert_eq!(node.children.len(), 2);
    assert_eq!(node.children[0].title.as_deref(), Some("Save"));
    assert_eq!(node.children[1].value.as_deref(), Some("Ada"));
}

#[test]
fn converted_json_keeps_the_persisted_six_field_shape() {
    let mut root = eyes_node("button", Some("OK"), None);
    root.secure = true;
    root.focused = true;
    root.role_key = Some("button".to_string());
    let json = serde_json::to_value(AXTreeNode::from(&root)).unwrap();
    let mut keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["bounds", "children", "identifier", "role", "title", "value"]
    );
}

#[test]
fn a_secure_value_never_survives_conversion() {
    // An unredacted secure node (defence in depth: eyes redacts at the source).
    let mut secret = eyes_node("AXSecureTextField", Some("Password"), Some("hunter2"));
    secret.secure = true;
    let mut root = eyes_node("AXWindow", Some("Login"), None);
    root.children = vec![secret];

    let node = AXTreeNode::from(&root);
    assert_eq!(node.children[0].value, None);
    assert_eq!(node.children[0].title.as_deref(), Some("Password"));
    let json = serde_json::to_string(&node).unwrap();
    assert!(!json.contains("hunter2"));
}

#[test]
fn find_never_matches_a_secure_value() {
    let mut secret = eyes_node("AXSecureTextField", Some("Password"), Some("hunter2"));
    secret.secure = true;
    let mut root = eyes_node("AXWindow", Some("Login"), None);
    root.children = vec![secret];
    assert!(find_in_tree(&root, "hunter2", FIND_OPTIONS).is_none());
    assert!(find_in_tree(&root, "password", FIND_OPTIONS).is_some());
}

#[test]
fn passive_capture_never_activates_the_renderer_or_reads_extra_values() {
    let opts = passive_options();
    assert_eq!(opts.target, WindowTarget::Focused);
    assert!(!opts.activate_renderer);
    assert!(!opts.limits.read_values);
}

#[test]
fn passive_limits_match_the_historical_per_os_caps() {
    let limits = passive_options().limits;
    // The caps Shadow's own walkers used before the adapter: macOS depth 4 with
    // 40 children, Windows depth 4 with 100, Linux AT-SPI depth 6 with 100.
    let expected: (u32, usize) = if cfg!(target_os = "macos") {
        (4, 40)
    } else if cfg!(target_os = "windows") {
        (4, 100)
    } else {
        (6, 100)
    };
    assert_eq!((limits.max_depth, limits.max_children), expected);
    assert_eq!(limits.max_nodes, 1500);
    assert_eq!(limits.budget, std::time::Duration::from_millis(1500));
}
