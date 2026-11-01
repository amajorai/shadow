//! One action on a verified target: the semantic accessibility action first,
//! physical input only as a gated fallback.
//!
//! Fallback rule (shared with Ghost): physical input runs only when the semantic
//! action failed with ACTION_NOT_SUPPORTED or PLATFORM_NOT_SUPPORTED and was not
//! delivered, and only when `Effector::physical` (a desktop session) is true.
//! Any other semantic failure, including an ignored write, ends the step without
//! retrying, so nothing is ever pressed or typed twice.

use std::time::Duration;

use ghost_eyes::accessibility::verify::verify_typed;
use ghost_eyes::{
    ActionOptions, ActionOutcome, Bounds, Delivery, ElementAction, ElementLocator, ErrorCode,
    GhostError, VerificationScope,
};
use serde_json::json;

use super::effector::Effector;
use super::verify::centre;

const SCROLL_AMOUNT: i32 = 3;
const FOCUS_SETTLE: Duration = Duration::from_millis(120);
const TYPE_SETTLE: Duration = Duration::from_millis(150);
const HEADLESS_HINT: &str =
    "physical input is unavailable in a headless session; this control needs a desktop session";

/// What a successful action did. `outcome` never contains typed text or values.
#[derive(Debug, Clone)]
pub(super) struct StepReport {
    pub method: String,
    pub verified: Option<bool>,
    pub outcome: &'static str,
}

fn semantic_options() -> ActionOptions {
    ActionOptions {
        activate_renderer: true,
        ..ActionOptions::default()
    }
}

/// Whether a semantic failure permits the physical fallback.
pub(super) fn permits_fallback(error: &GhostError) -> bool {
    matches!(
        error.code,
        ErrorCode::ActionNotSupported | ErrorCode::PlatformNotSupported
    ) && error.delivery == Delivery::NotDelivered
}

fn semantic_verified(outcome: &ActionOutcome) -> bool {
    outcome.delivery == Delivery::DeliveredVerified
        || outcome.verification.scope == VerificationScope::AlreadyInState
}

/// A semantic failure that may not fall back: explain a headless refusal.
fn no_fallback(error: GhostError, physical: bool) -> GhostError {
    if permits_fallback(&error) && !physical && error.suggestion.is_none() {
        error.with_suggestion(HEADLESS_HINT)
    } else {
        error
    }
}

fn click_failed() -> GhostError {
    GhostError::new(ErrorCode::ActionFailed, "the physical click failed")
}

/// CLICK: `perform(Press)`, then a physical click at `point` when permitted.
pub(super) async fn click<E: Effector>(
    fx: &E,
    locator: &ElementLocator,
    point: Option<(i32, i32)>,
) -> Result<StepReport, GhostError> {
    match fx
        .perform(locator, &ElementAction::Press, &semantic_options())
        .await
    {
        Ok(outcome) => Ok(StepReport {
            method: outcome.method.to_owned(),
            verified: Some(semantic_verified(&outcome)),
            outcome: "clicked",
        }),
        Err(error) if permits_fallback(&error) && fx.physical() => {
            let (x, y) = point.ok_or(error)?;
            fx.click(x, y).map_err(|_| click_failed())?;
            Ok(StepReport {
                method: "hid.click".to_owned(),
                verified: Some(false),
                outcome: "clicked with physical input",
            })
        }
        Err(error) => Err(no_fallback(error, fx.physical())),
    }
}

/// TYPE_TEXT with replace semantics: `perform(SetValue)`, then (when permitted)
/// focus + physical typing that clears the field first, verified by read-back.
pub(super) async fn type_into<E: Effector>(
    fx: &E,
    locator: &ElementLocator,
    text: &str,
    before: Option<&str>,
    point: Option<(i32, i32)>,
) -> Result<StepReport, GhostError> {
    let set_value = ElementAction::SetValue(text.to_owned());
    match fx.perform(locator, &set_value, &semantic_options()).await {
        Ok(outcome) => Ok(StepReport {
            method: outcome.method.to_owned(),
            verified: Some(semantic_verified(&outcome)),
            outcome: "set the field value",
        }),
        Err(error) if permits_fallback(&error) && fx.physical() => {
            type_physically(fx, locator, text, before, point).await
        }
        Err(error) => Err(no_fallback(error, fx.physical())),
    }
}

async fn type_physically<E: Effector>(
    fx: &E,
    locator: &ElementLocator,
    text: &str,
    before: Option<&str>,
    point: Option<(i32, i32)>,
) -> Result<StepReport, GhostError> {
    let focus_method = match fx
        .perform(locator, &ElementAction::Focus, &semantic_options())
        .await
    {
        Ok(outcome) => outcome.method,
        // Only an unsupported focus may fall back to a click: a stale, ambiguous or
        // denied target must never be clicked at its old centre.
        Err(error) if permits_fallback(&error) => {
            let (x, y) = point.ok_or(error)?;
            fx.click(x, y).map_err(|_| click_failed())?;
            tokio::time::sleep(FOCUS_SETTLE).await;
            "hid.click"
        }
        Err(error) => return Err(error),
    };
    let count = text.chars().count();
    // The backend's error can echo the characters it could not type: replace it.
    fx.type_text(text).map_err(|_| {
        GhostError::new(
            ErrorCode::ActionFailed,
            format!("typing failed ({count} characters)"),
        )
        .with_delivery(Delivery::Unknown)
    })?;
    tokio::time::sleep(TYPE_SETTLE).await;
    let after = fx.read_value(locator).await;
    let verified = match verify_typed(before, text, true, after.as_deref()) {
        Some(false) => {
            return Err(GhostError::new(
                ErrorCode::ActionFailed,
                format!("the field did not hold the {count} typed characters afterwards"),
            )
            .with_delivery(Delivery::DeliveredUnverified)
            .with_details(json!({ "kind": "post_action_verification" }))
            .with_suggestion("observe the field again before retrying"))
        }
        Some(true) => true,
        // Secure field or nothing to read back: unverified, not failed.
        None => false,
    };
    Ok(StepReport {
        method: format!("{focus_method}+hid.type"),
        verified: Some(verified),
        outcome: "focused the field and entered text",
    })
}

/// SCROLL: physical only, at the centre of the window (not the display).
pub(super) fn scroll<E: Effector>(
    fx: &E,
    window: Option<&Bounds>,
    up: bool,
) -> Result<StepReport, GhostError> {
    if !fx.physical() {
        return Err(GhostError::policy_denied(
            "scrolling needs physical input, which is unavailable in a headless session",
        ));
    }
    let bounds = window
        .filter(|b| b.width > 0 && b.height > 0)
        .ok_or_else(|| GhostError::not_supported("the focused window reports no bounds"))?;
    let (x, y) = centre(bounds);
    let direction = if up { "up" } else { "down" };
    fx.scroll(x, y, direction, SCROLL_AMOUNT)
        .map_err(|_| GhostError::new(ErrorCode::ActionFailed, "scrolling the window failed"))?;
    Ok(StepReport {
        method: "hid.scroll".to_owned(),
        verified: None,
        outcome: "scrolled",
    })
}
