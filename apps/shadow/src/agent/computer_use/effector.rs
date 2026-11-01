//! The side-effecting seam of computer use: accessibility snapshots, semantic
//! actions and physical input. `Desktop` is the real implementation; tests
//! inject a fake so the fallback rules run without a display.

use std::time::Duration;

use async_trait::async_trait;
use ghost_eyes::{
    AXSnapshot, AXTree as EyesAXTree, ActionOptions, ActionOutcome, ElementAction, ElementLocator,
    GhostError, PlatformAXTree, SnapshotOptions,
};

/// How long a post-typing read-back may take.
const READ_BACK_DEADLINE: Duration = Duration::from_secs(2);

#[async_trait]
pub(super) trait Effector: Send + Sync {
    /// Whether physical (HID) input may run: a desktop session, never headless.
    fn physical(&self) -> bool;
    async fn snapshot(&self, opts: SnapshotOptions) -> Result<AXSnapshot, GhostError>;
    /// Semantic action only; never synthesizes physical input.
    async fn perform(
        &self,
        locator: &ElementLocator,
        action: &ElementAction,
        opts: &ActionOptions,
    ) -> Result<ActionOutcome, GhostError>;
    /// The element's current value; `None` when unknown or secure.
    async fn read_value(&self, locator: &ElementLocator) -> Option<String>;
    fn display_size(&self) -> (u32, u32);
    fn click(&self, x: i32, y: i32) -> anyhow::Result<()>;
    /// Replaces the focused field's text. Errors may echo characters, so callers
    /// must never surface them.
    fn type_text(&self, text: &str) -> anyhow::Result<()>;
    fn scroll(&self, x: i32, y: i32, direction: &str, amount: i32) -> anyhow::Result<()>;
}

/// The real desktop: `ghost_eyes` for the tree and actions, `ghost_hands` for input.
pub(super) struct Desktop {
    ax: PlatformAXTree,
    physical: bool,
}

impl Desktop {
    pub(super) fn new() -> anyhow::Result<Self> {
        Ok(Self {
            ax: PlatformAXTree::new()?,
            physical: ghost_hands::mode::hid_permitted(true),
        })
    }
}

#[async_trait]
impl Effector for Desktop {
    fn physical(&self) -> bool {
        self.physical
    }

    async fn snapshot(&self, opts: SnapshotOptions) -> Result<AXSnapshot, GhostError> {
        EyesAXTree::snapshot(&self.ax, opts).await
    }

    async fn perform(
        &self,
        locator: &ElementLocator,
        action: &ElementAction,
        opts: &ActionOptions,
    ) -> Result<ActionOutcome, GhostError> {
        EyesAXTree::perform(&self.ax, locator, action, opts).await
    }

    async fn read_value(&self, locator: &ElementLocator) -> Option<String> {
        let resolved = EyesAXTree::resolve(&self.ax, locator, READ_BACK_DEADLINE)
            .await
            .ok()?;
        if resolved.node.secure || resolved.state.secure {
            return None;
        }
        resolved.state.value.or(resolved.node.value)
    }

    fn display_size(&self) -> (u32, u32) {
        crate::capture::screen::get_primary_display_size()
    }

    fn click(&self, x: i32, y: i32) -> anyhow::Result<()> {
        ghost_hands::mouse_click(x, y, ghost_hands::MouseButton::Left, 1)
    }

    fn type_text(&self, text: &str) -> anyhow::Result<()> {
        ghost_hands::type_text(text, true)
    }

    fn scroll(&self, x: i32, y: i32, direction: &str, amount: i32) -> anyhow::Result<()> {
        ghost_hands::scroll(x, y, direction, amount)
    }
}
