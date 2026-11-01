use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingResult {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub strategy: GroundingStrategy,
    /// Set for accessibility hits: the element to act on semantically. Not
    /// serialized (it is a live-resolution handle, not a result field).
    #[serde(skip)]
    pub ax_target: Option<AxTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroundingStrategy {
    AxExact,
    AxFuzzy,
    LocalVlm,
    CloudVision,
}

// ─── ShowUI-2B ONNX model (requires `ort` feature) ───────────────────────────

#[cfg(feature = "ort")]
mod showui {
    use super::*;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::Tensor;
    use std::path::Path;
    use std::sync::Mutex;

    #[inline]
    fn oe(e: impl std::fmt::Display) -> anyhow::Error {
        anyhow::anyhow!("{}", e)
    }

    // ShowUI-2B expected input resolution
    const SHOWUI_W: u32 = 1280;
    const SHOWUI_H: u32 = 828;
    const SHOWUI_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    const SHOWUI_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub struct ShowUIModel {
        session: Mutex<Session>,
    }

    impl ShowUIModel {
        pub fn load(model_path: &Path) -> Result<Self> {
            tracing::info!("Loading ShowUI-2B model from {:?}", model_path);
            let session = Session::builder()
                .map_err(oe)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(oe)?
                .with_intra_threads(4)
                .map_err(oe)?
                .commit_from_file(model_path)
                .map_err(|e| anyhow::anyhow!("Failed to load ShowUI-2B ONNX model: {}", e))?;
            tracing::info!("ShowUI-2B loaded");
            Ok(Self {
                session: Mutex::new(session),
            })
        }

        /// Run grounding: screenshot (BGRA) + instruction → normalised (x, y).
        pub async fn ground(
            &self,
            screenshot: &[u8],
            width: u32,
            height: u32,
            instruction: &str,
        ) -> Result<GroundingResult> {
            let img_pixels = bgra_to_pixels(screenshot, width, height)?;
            let img_tensor = Tensor::<f32>::from_array((
                [1usize, 3, SHOWUI_H as usize, SHOWUI_W as usize],
                img_pixels,
            ))
            .map_err(|e| anyhow::anyhow!("ShowUI image tensor: {}", e))?;

            // Encode instruction as UTF-8 bytes padded to 256 tokens
            let text_tokens: Vec<i64> = instruction
                .bytes()
                .take(255)
                .map(|b| b as i64)
                .chain(std::iter::repeat(0))
                .take(256)
                .collect();
            let text_tensor = Tensor::<i64>::from_array(([1usize, 256], text_tokens))
                .map_err(|e| anyhow::anyhow!("ShowUI text tensor: {}", e))?;

            let mut guard = self
                .session
                .lock()
                .map_err(|_| anyhow::anyhow!("ShowUI session mutex poisoned"))?;
            let outputs = guard
                .run(ort::inputs![
                    "image"       => &img_tensor,
                    "instruction" => &text_tensor
                ])
                .map_err(|e| anyhow::anyhow!("ShowUI-2B inference failed: {}", e))?;

            // Model outputs a [1, 2] float tensor: [norm_x, norm_y]
            let (_, flat) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
                anyhow::anyhow!("ShowUI-2B: failed to extract coordinate tensor: {}", e)
            })?;
            if flat.len() < 2 {
                anyhow::bail!("ShowUI-2B returned unexpected output shape");
            }

            Ok(GroundingResult {
                x: flat[0].clamp(0.0, 1.0),
                y: flat[1].clamp(0.0, 1.0),
                confidence: 0.75,
                strategy: GroundingStrategy::LocalVlm,
                ax_target: None,
            })
        }
    }

    /// Convert BGRA raw pixels → CHW f32 Vec normalised with ImageNet stats,
    /// resized to ShowUI resolution.
    fn bgra_to_pixels(bgra: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        let img = image::RgbaImage::from_raw(width, height, bgra.to_vec())
            .ok_or_else(|| anyhow::anyhow!("BGRA→RgbaImage failed"))?;
        let rgb = image::DynamicImage::ImageRgba8(img)
            .resize_exact(SHOWUI_W, SHOWUI_H, image::imageops::FilterType::Lanczos3)
            .to_rgb8();

        let w = SHOWUI_W as usize;
        let h = SHOWUI_H as usize;
        let mut out = vec![0.0f32; 3 * h * w];
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let yi = y as usize;
            let xi = x as usize;
            for c in 0..3usize {
                out[c * h * w + yi * w + xi] =
                    (pixel[c] as f32 / 255.0 - SHOWUI_MEAN[c]) / SHOWUI_STD[c];
            }
        }
        Ok(out)
    }

    pub use ShowUIModel as ShowUI;
}

// ─── AX-based grounding helpers ───────────────────────────────────────────────

/// A grounding hit on a live accessibility element: a locator the caller can hand
/// to a semantic action, plus the element's screen centre for an input fallback.
#[derive(Debug, Clone)]
pub struct AxTarget {
    pub locator: ghost_eyes::ElementLocator,
    pub center: (i32, i32),
}

/// One snapshot of the focused window for grounding (no renderer activation).
async fn grounding_snapshot() -> Option<ghost_eyes::AXSnapshot> {
    use ghost_eyes::accessibility::AXTree as _;
    let ax = crate::capture::accessibility::PlatformAXTree::new().ok()?;
    let opts = ghost_eyes::SnapshotOptions {
        target: ghost_eyes::WindowTarget::Focused,
        limits: ghost_eyes::WalkLimits::platform_default(),
        activate_renderer: false,
    };
    ax.eyes().snapshot(opts).await.ok()
}

/// Locator for `node` at `path` (relative to the snapshot root). A known pid pins
/// the process and window title, so the action refuses a window that changed.
fn locator_for(
    snapshot: &ghost_eyes::AXSnapshot,
    path: Vec<u32>,
    node: &ghost_eyes::AXTreeNode,
) -> ghost_eyes::ElementLocator {
    use ghost_eyes::WindowTarget;
    let window = match (snapshot.pid, snapshot.window_title.as_deref()) {
        (pid, Some(title)) if pid > 0 && !title.is_empty() => WindowTarget::PidTitle {
            pid,
            title: title.to_string(),
        },
        (pid, _) if pid > 0 => WindowTarget::Pid { pid },
        _ => WindowTarget::Focused,
    };
    ghost_eyes::ElementLocator {
        pid: snapshot.pid,
        window,
        path,
        identity: ghost_eyes::ElementIdentity::from_node(node),
    }
}

fn center_of(bounds: &crate::capture::Bounds) -> (i32, i32) {
    (
        bounds.x + (bounds.width / 2) as i32,
        bounds.y + (bounds.height / 2) as i32,
    )
}

fn ax_result(
    snapshot: &ghost_eyes::AXSnapshot,
    path: Vec<u32>,
    node: &ghost_eyes::AXTreeNode,
    screen: (u32, u32),
    confidence: f32,
    strategy: GroundingStrategy,
) -> Option<GroundingResult> {
    let center = center_of(node.bounds.as_ref()?);
    let (screen_w, screen_h) = (screen.0.max(1) as f32, screen.1.max(1) as f32);
    Some(GroundingResult {
        x: (center.0 as f32 / screen_w).clamp(0.0, 1.0),
        y: (center.1 as f32 / screen_h).clamp(0.0, 1.0),
        confidence,
        strategy,
        ax_target: Some(AxTarget {
            locator: locator_for(snapshot, path, node),
            center,
        }),
    })
}

/// Try to find an element in the accessibility tree that matches `query` exactly
/// (case-insensitive substring of role, title, identifier or non-secure value).
fn try_ax_exact(
    snapshot: &ghost_eyes::AXSnapshot,
    query: &str,
    screen: (u32, u32),
) -> Option<GroundingResult> {
    use ghost_eyes::accessibility::find::find_all;
    let opts = crate::capture::accessibility::FIND_OPTIONS;
    let (path, node) = find_all(&snapshot.root, query, opts, 1)
        .into_iter()
        .next()?;
    ax_result(
        snapshot,
        path,
        node,
        screen,
        0.95,
        GroundingStrategy::AxExact,
    )
}

/// Fuzzy AX match: walk the full tree and pick the element whose role/title/value
/// has the highest token overlap with the query. Secure values are never read
/// (they are withheld at the source).
fn try_ax_fuzzy(
    snapshot: &ghost_eyes::AXSnapshot,
    query: &str,
    screen: (u32, u32),
) -> Option<GroundingResult> {
    use ghost_eyes::AXTreeNode;

    struct Best<'a> {
        score: f32,
        path: Vec<u32>,
        node: Option<&'a AXTreeNode>,
    }

    fn walk<'a>(node: &'a AXTreeNode, tokens: &[&str], path: &mut Vec<u32>, best: &mut Best<'a>) {
        let value = if node.secure {
            ""
        } else {
            node.value.as_deref().unwrap_or("")
        };
        let text = format!(
            "{} {} {}",
            node.role,
            node.title.as_deref().unwrap_or(""),
            value
        )
        .to_lowercase();
        let matched = tokens.iter().filter(|&&t| text.contains(t)).count();
        let score = matched as f32 / tokens.len().max(1) as f32;
        if score > best.score && node.bounds.is_some() {
            best.score = score;
            best.path = path.clone();
            best.node = Some(node);
        }
        for (i, child) in node.children.iter().enumerate() {
            path.push(u32::try_from(i).unwrap_or(u32::MAX));
            walk(child, tokens, path, best);
            path.pop();
        }
    }

    let query_lower = query.to_lowercase();
    let tokens: Vec<&str> = query_lower.split_whitespace().collect();
    if tokens.is_empty() {
        return None;
    }
    let mut best = Best {
        score: 0.0,
        path: Vec::new(),
        node: None,
    };
    walk(&snapshot.root, &tokens, &mut Vec::new(), &mut best);

    // Only accept if at least half the query tokens matched
    if best.score < 0.5 {
        return None;
    }
    ax_result(
        snapshot,
        best.path,
        best.node?,
        screen,
        0.5 + best.score * 0.4,
        GroundingStrategy::AxFuzzy,
    )
}

// ─── Grounding oracle ─────────────────────────────────────────────────────────

/// Multi-strategy grounding oracle.
/// Cascade: AX exact → AX fuzzy → ShowUI-2B (local VLM) → error
pub struct GroundingOracle {
    #[cfg(feature = "ort")]
    showui: Option<showui::ShowUI>,
    vlm_threshold: f32,
}

impl GroundingOracle {
    pub fn new() -> Result<Self> {
        Ok(Self {
            #[cfg(feature = "ort")]
            showui: None,
            vlm_threshold: 0.30,
        })
    }

    /// Load ShowUI-2B model (optional; if not called, the local VLM stage is skipped).
    #[cfg(feature = "ort")]
    pub async fn load_showui(&mut self, model_path: &std::path::Path) -> Result<()> {
        self.showui = Some(showui::ShowUI::load(model_path)?);
        tracing::info!("Grounding oracle: ShowUI-2B ready");
        Ok(())
    }

    /// Ground an instruction against the current screen.
    ///
    /// `screenshot` must be BGRA raw pixel data of dimensions `width` × `height`.
    /// Pass `&[]` to skip ShowUI and rely only on the AX cascade.
    pub async fn ground(
        &self,
        instruction: &str,
        screenshot: &[u8],
        width: u32,
        height: u32,
    ) -> Result<GroundingResult> {
        if let Some(snapshot) = grounding_snapshot().await {
            // 1. AX exact match — near-zero latency, highest confidence
            if let Some(r) = try_ax_exact(&snapshot, instruction, (width, height)) {
                tracing::debug!("Grounding: AX exact hit for {:?}", instruction);
                return Ok(r);
            }

            // 2. AX fuzzy match — still free, covers partial descriptions
            if let Some(r) = try_ax_fuzzy(&snapshot, instruction, (width, height)) {
                tracing::debug!(
                    "Grounding: AX fuzzy hit for {:?} (conf={:.2})",
                    instruction,
                    r.confidence
                );
                return Ok(r);
            }
        }

        // 3. ShowUI-2B local VLM (~300 ms)
        #[cfg(feature = "ort")]
        if let Some(ref model) = self.showui {
            if !screenshot.is_empty() {
                match model.ground(screenshot, width, height, instruction).await {
                    Ok(r) if r.confidence >= self.vlm_threshold => {
                        tracing::debug!("Grounding: ShowUI-2B hit for {:?}", instruction);
                        return Ok(r);
                    }
                    Ok(r) => tracing::debug!(
                        "Grounding: ShowUI-2B below threshold ({:.2}) for {:?}",
                        r.confidence,
                        instruction
                    ),
                    Err(e) => tracing::warn!("ShowUI-2B error: {}", e),
                }
            }
        }

        anyhow::bail!(
            "Grounding failed: no strategy found a match for {:?}",
            instruction
        )
    }
}

impl Default for GroundingOracle {
    fn default() -> Self {
        Self::new().expect("Failed to create GroundingOracle")
    }
}

/// Parse normalised coordinates from LLM / VLM text output.
/// Handles [x, y], (x, y), click(x=N, y=N) formats.
pub fn parse_coordinates(output: &str) -> Option<(f32, f32)> {
    let patterns: &[&str] = &[
        r"\[(\d+\.?\d*),\s*(\d+\.?\d*)\]",
        r"\((\d+\.?\d*),\s*(\d+\.?\d*)\)",
        r"click\(x=(\d+\.?\d*),\s*y=(\d+\.?\d*)\)",
    ];
    for pat in patterns {
        if let Some(caps) = regex::Regex::new(pat).ok()?.captures(output) {
            let x: f32 = caps.get(1)?.as_str().parse().ok()?;
            let y: f32 = caps.get(2)?.as_str().parse().ok()?;
            return Some((x, y));
        }
    }
    None
}

#[cfg(test)]
#[path = "grounding_test.rs"]
mod tests;
