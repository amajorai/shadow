//! The semantic-first action rules, driven through a fake `Effector`.

use std::sync::Mutex;

use async_trait::async_trait;
use ghost_eyes::{
    AXSnapshot, AXTreeNode, ActionOptions, ActionOutcome, Bounds, Delivery, ElementAction,
    ElementIdentity, ElementLocator, ElementState, ErrorCode, GhostError, MatchBy, RendererA11y,
    ResolvedElement, SnapshotOptions, Verification, VerificationScope, WindowTarget,
};
use serde_json::json;

use super::super::effector::Effector;
use super::super::input::{click, scroll, type_into};

const TEXT: &str = "secret typed words";

#[derive(Default)]
struct Fake {
    physical: bool,
    /// Errors returned by `perform`, in order; an empty queue means success.
    perform_errors: Mutex<Vec<GhostError>>,
    performed: Mutex<Vec<&'static str>>,
    clicks: Mutex<u32>,
    typed: Mutex<u32>,
    scrolls: Mutex<Vec<(i32, i32)>>,
    type_fails: bool,
    read_back: Option<String>,
}

fn outcome(action: &'static str) -> ActionOutcome {
    ActionOutcome {
        action,
        method: "AXPress",
        delivery: Delivery::DeliveredVerified,
        verification: Verification {
            scope: VerificationScope::ElementState,
            reason: None,
        },
        attempts: Vec::new(),
        element: ResolvedElement {
            node: AXTreeNode::default(),
            path: vec![0],
            matched_by: MatchBy::Path,
            pid: 42,
            state: ElementState::default(),
        },
        before: None,
        after: None,
        renderer: RendererA11y::NotApplicable,
    }
}

#[async_trait]
impl Effector for Fake {
    fn physical(&self) -> bool {
        self.physical
    }
    async fn snapshot(&self, _opts: SnapshotOptions) -> Result<AXSnapshot, GhostError> {
        Err(GhostError::internal("not used"))
    }
    async fn perform(
        &self,
        _locator: &ElementLocator,
        action: &ElementAction,
        _opts: &ActionOptions,
    ) -> Result<ActionOutcome, GhostError> {
        self.performed.lock().unwrap().push(action.name());
        let mut errors = self.perform_errors.lock().unwrap();
        if errors.is_empty() {
            Ok(outcome(action.name()))
        } else {
            Err(errors.remove(0))
        }
    }
    async fn read_value(&self, _locator: &ElementLocator) -> Option<String> {
        self.read_back.clone()
    }
    fn display_size(&self) -> (u32, u32) {
        (1920, 1080)
    }
    fn click(&self, _x: i32, _y: i32) -> anyhow::Result<()> {
        *self.clicks.lock().unwrap() += 1;
        Ok(())
    }
    fn type_text(&self, text: &str) -> anyhow::Result<()> {
        *self.typed.lock().unwrap() += 1;
        if self.type_fails {
            anyhow::bail!("could not type '{text}'");
        }
        Ok(())
    }
    fn scroll(&self, x: i32, y: i32, _direction: &str, _amount: i32) -> anyhow::Result<()> {
        self.scrolls.lock().unwrap().push((x, y));
        Ok(())
    }
}

fn locator() -> ElementLocator {
    ElementLocator {
        pid: 42,
        window: WindowTarget::Pid { pid: 42 },
        path: vec![0],
        identity: ElementIdentity::default(),
    }
}

fn unsupported() -> GhostError {
    GhostError::not_supported("no semantic action")
}

fn write_ignored() -> GhostError {
    GhostError::new(ErrorCode::ActionFailed, "the value did not change")
        .with_delivery(Delivery::DeliveredUnverified)
        .with_details(json!({ "cause": "write_ignored" }))
}

fn fake(physical: bool, errors: Vec<GhostError>) -> Fake {
    Fake {
        physical,
        perform_errors: Mutex::new(errors),
        ..Fake::default()
    }
}

#[tokio::test]
async fn click_uses_the_semantic_press_without_physical_input() {
    let fx = fake(true, Vec::new());
    let report = click(&fx, &locator(), Some((5, 5))).await.unwrap();
    assert_eq!(report.method, "AXPress");
    assert_eq!(report.verified, Some(true));
    assert_eq!(*fx.clicks.lock().unwrap(), 0);
}

#[tokio::test]
async fn click_falls_back_to_physical_input_only_on_a_desktop_session() {
    let fx = fake(true, vec![unsupported()]);
    let report = click(&fx, &locator(), Some((5, 5))).await.unwrap();
    assert_eq!(report.method, "hid.click");
    assert_eq!(*fx.clicks.lock().unwrap(), 1);

    let headless = fake(false, vec![unsupported()]);
    let error = click(&headless, &locator(), Some((5, 5)))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ActionNotSupported);
    assert!(error.suggestion.is_some());
    assert_eq!(*headless.clicks.lock().unwrap(), 0);
}

#[tokio::test]
async fn type_text_does_not_retype_after_a_write_ignored_failure() {
    let fx = fake(true, vec![write_ignored()]);
    let error = type_into(&fx, &locator(), TEXT, None, Some((5, 5)))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ActionFailed);
    assert_eq!(*fx.typed.lock().unwrap(), 0);
    assert_eq!(*fx.clicks.lock().unwrap(), 0);
    assert_eq!(*fx.performed.lock().unwrap(), vec!["set_value"]);
}

#[tokio::test]
async fn type_text_sets_the_value_semantically_first() {
    let fx = fake(true, Vec::new());
    let report = type_into(&fx, &locator(), TEXT, Some("old"), None)
        .await
        .unwrap();
    assert_eq!(report.verified, Some(true));
    assert_eq!(*fx.typed.lock().unwrap(), 0);
}

#[tokio::test]
async fn type_text_falls_back_to_focus_and_verified_typing() {
    let mut fx = fake(true, vec![unsupported()]);
    fx.read_back = Some(TEXT.to_owned());
    let report = type_into(&fx, &locator(), TEXT, Some("old"), Some((5, 5)))
        .await
        .unwrap();
    assert_eq!(report.verified, Some(true));
    assert!(report.method.ends_with("hid.type"));
    assert_eq!(*fx.performed.lock().unwrap(), vec!["set_value", "focus"]);
    assert_eq!(*fx.typed.lock().unwrap(), 1);
}

#[tokio::test]
async fn a_mismatched_read_back_fails_without_echoing_the_text() {
    let mut fx = fake(true, vec![unsupported()]);
    fx.read_back = Some("something else".to_owned());
    let error = type_into(&fx, &locator(), TEXT, None, Some((5, 5)))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ActionFailed);
    assert_eq!(error.delivery, Delivery::DeliveredUnverified);
    assert!(!error.to_json().to_string().contains(TEXT));
}

#[tokio::test]
async fn a_typing_error_never_carries_the_text() {
    let mut fx = fake(true, vec![unsupported()]);
    fx.type_fails = true;
    let error = type_into(&fx, &locator(), TEXT, None, Some((5, 5)))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ActionFailed);
    assert_eq!(error.delivery, Delivery::Unknown);
    assert!(!error.to_string().contains(TEXT));
    assert!(error.message.contains(&TEXT.chars().count().to_string()));
}

#[tokio::test]
async fn an_unreadable_secure_field_is_unverified_not_failed() {
    let fx = fake(true, vec![unsupported()]);
    let report = type_into(&fx, &locator(), TEXT, None, Some((5, 5)))
        .await
        .unwrap();
    assert_eq!(report.verified, Some(false));
}

#[test]
fn scroll_targets_the_window_centre_and_is_refused_headless() {
    let window = Bounds {
        x: 100,
        y: 50,
        width: 400,
        height: 300,
    };
    let fx = fake(true, Vec::new());
    scroll(&fx, Some(&window), false).unwrap();
    assert_eq!(*fx.scrolls.lock().unwrap(), vec![(300, 200)]);

    let headless = fake(false, Vec::new());
    let error = scroll(&headless, Some(&window), true).unwrap_err();
    assert_eq!(error.code, ErrorCode::PolicyDenied);
    assert!(headless.scrolls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn type_text_does_not_click_when_focus_finds_a_stale_target() {
    let stale = GhostError::stale_ref("the element is no longer present");
    let fx = fake(true, vec![unsupported(), stale]);
    let error = type_into(&fx, &locator(), TEXT, None, Some((5, 5)))
        .await
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::StaleRef);
    assert_eq!(*fx.clicks.lock().unwrap(), 0);
    assert_eq!(*fx.typed.lock().unwrap(), 0);
    assert_eq!(*fx.performed.lock().unwrap(), vec!["set_value", "focus"]);
}

#[tokio::test]
async fn type_text_clicks_to_focus_only_after_an_unsupported_focus() {
    let fx = fake(true, vec![unsupported(), unsupported()]);
    let report = type_into(&fx, &locator(), TEXT, None, Some((5, 5)))
        .await
        .unwrap();
    assert_eq!(report.method, "hid.click+hid.type");
    assert_eq!(*fx.clicks.lock().unwrap(), 1);
    assert_eq!(*fx.typed.lock().unwrap(), 1);
}
