//! Shadow's accessibility capture: a thin adapter over `ghost_eyes`.
//!
//! Shadow keeps its own six-field `AXTreeNode` (persisted as `ax_snapshot` JSON and
//! built as literals elsewhere), while every platform walk, limit and secure-field
//! redaction comes from `ghost_eyes`. Passive capture uses `WalkLimits::passive()`,
//! which keeps the historical per-OS depth and sibling caps, never activates the
//! Chromium/Electron renderer tree, and skips the extra Linux value reads.

use anyhow::Result;
use async_trait::async_trait;
use ghost_eyes::accessibility::find::{find_in_tree, FindOptions};
use ghost_eyes::accessibility::AXTree as EyesTree;
use ghost_eyes::{SnapshotOptions, WalkLimits, WindowTarget};
use serde::{Deserialize, Serialize};

pub use ghost_eyes::accessibility::Bounds;

/// Accessibility tree node (Shadow's persisted six-field shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AXTreeNode {
    pub role: String,
    pub title: Option<String>,
    pub value: Option<String>,
    pub identifier: Option<String>,
    pub bounds: Option<Bounds>,
    pub children: Vec<AXTreeNode>,
}

impl From<&ghost_eyes::AXTreeNode> for AXTreeNode {
    /// The eyes walker already redacts secure values; the conversion drops a
    /// secure node's value again so an unredacted node can never leak through.
    fn from(node: &ghost_eyes::AXTreeNode) -> Self {
        Self {
            role: node.role.clone(),
            title: node.title.clone(),
            value: if node.secure {
                None
            } else {
                node.value.clone()
            },
            identifier: node.identifier.clone(),
            bounds: node.bounds.clone(),
            children: node.children.iter().map(Self::from).collect(),
        }
    }
}

/// Accessibility tree trait.
#[async_trait]
pub trait AXTree: Send + Sync {
    async fn get_focused_tree(&self) -> Result<AXTreeNode>;
    async fn find_element(&self, description: &str) -> Option<AXTreeNode>;
}

/// Matching used by `find_element`: role, title, identifier and (non-secure) value.
pub(crate) const FIND_OPTIONS: FindOptions = FindOptions {
    match_identifier: true,
    match_value: true,
};

/// Snapshot options for passive capture: focused window, passive limits, no
/// renderer activation.
pub(crate) fn passive_options() -> SnapshotOptions {
    SnapshotOptions {
        target: WindowTarget::Focused,
        limits: WalkLimits::passive(),
        activate_renderer: false,
    }
}

/// The platform accessibility tree, backed by `ghost_eyes`.
pub struct PlatformAXTree {
    inner: ghost_eyes::PlatformAXTree,
}

impl PlatformAXTree {
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: ghost_eyes::PlatformAXTree::new()?,
        })
    }

    /// The underlying `ghost_eyes` backend, for snapshots, live resolution and
    /// semantic actions.
    pub fn eyes(&self) -> &ghost_eyes::PlatformAXTree {
        &self.inner
    }

    /// A passive snapshot of the focused window in the eyes node shape.
    async fn passive_tree(&self) -> Result<ghost_eyes::AXTreeNode> {
        let snapshot = self.inner.snapshot(passive_options()).await?;
        Ok(snapshot.root)
    }
}

#[async_trait]
impl AXTree for PlatformAXTree {
    async fn get_focused_tree(&self) -> Result<AXTreeNode> {
        let root = self.passive_tree().await?;
        Ok(AXTreeNode::from(&root))
    }

    async fn find_element(&self, description: &str) -> Option<AXTreeNode> {
        if description.trim().is_empty() {
            return None;
        }
        let root = self.passive_tree().await.ok()?;
        find_in_tree(&root, description, FIND_OPTIONS).map(|node| AXTreeNode::from(&node))
    }
}

#[cfg(test)]
#[path = "accessibility_test.rs"]
mod tests;
