//! Budgeted keyframe extractor for ingested clips (Ryu Clips phase 2).
//!
//! This is the *additive* half of the frame pipeline: the on-demand
//! [`crate::capture::clip::extract_frame`] cache tier stays for recorded clips,
//! while this module builds the whole keyframe *set* for an ingested video (a
//! watched URL or a local file) up front — scene-detected, deduplicated, capped
//! per detail mode, and written as `at-<atMs>.jpg` into the same `frames_dir`
//! the frame endpoint reads. That way the desktop composer samples real scene
//! frames with zero on-demand ffmpeg.
//!
//! Pure `ffmpeg` + `image` logic — no `clip.rs` privates. The caller
//! ([`crate::capture::clip::ingest`]) runs this under `spawn_blocking` since
//! every step shells out to the `ffmpeg` CLI (the same pattern as
//! `video::FrameExtractor`, NOT the optional `video` cargo feature).
//!
//! Placement (CLAUDE.md §1): deciding *what is captured/sampled on this device*
//! is the sensor half → Shadow. Nothing here enforces policy.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

// ─── Swappable defaults (nothing hardcoded — every knob is an env override) ──────

/// Scene-change score above which ffmpeg emits a keyframe (`gt(scene,THRESH)`).
const DEFAULT_SCENE_THRESHOLD: f64 = 0.3;
/// Output frame width; frames are downscaled to this via `scale=W:-1`. Distinct
/// from the capture-time `MAX_FRAME_WIDTH` (1280) — clip keyframes are for an
/// agent to read, so 512px keeps the token cost down.
const DEFAULT_FRAME_WIDTH: u32 = 512;
/// Grayscale-thumbnail mean-abs-delta (0–255) at or below which a frame is
/// dropped as a near-duplicate of the previous KEPT frame.
const DEFAULT_DEDUP_DELTA: f64 = 2.0;
/// Target frame count for the duration-aware `efficient` mode (fps = N / secs).
const DEFAULT_EFFICIENT_FRAMES: u32 = 50;
/// Max frames the `balanced` mode keeps; more are subsampled evenly.
const DEFAULT_BALANCED_CAP: usize = 100;

/// A video longer than this counts as "long" for the subsample `scan_warning`.
const LONG_VIDEO_MS: u64 = 5 * 60 * 1000;
/// Edge of the grayscale thumbnail used for the dedup delta.
const THUMB_EDGE: u32 = 32;

// ─── Wire types ─────────────────────────────────────────────────────────────────

/// How much visual detail to extract. `Transcript` skips frames entirely (the
/// agent gets only the transcript); `Efficient` samples at a duration-aware fps
/// with no scene detection; `Balanced` scene-detects and caps; `TokenBurner`
/// scene-detects uncapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum DetailMode {
    Transcript,
    Efficient,
    #[default]
    Balanced,
    TokenBurner,
}

/// One extracted keyframe: the moment it depicts and where its JPEG landed.
#[derive(Debug, Clone)]
pub struct Keyframe {
    pub at_ms: u64,
    pub path: PathBuf,
}

/// The result of an extraction pass: the kept frames plus an optional
/// human-facing note (e.g. a long video was subsampled to the cap).
#[derive(Debug, Clone, Default)]
pub struct KeyframeSet {
    pub frames: Vec<Keyframe>,
    pub scan_warning: Option<String>,
}

// ─── Env knobs ──────────────────────────────────────────────────────────────────

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(default)
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn frame_width() -> u32 {
    env_u32("RYU_CLIP_FRAME_WIDTH", DEFAULT_FRAME_WIDTH).max(16)
}

fn secs_str(ms: u64) -> String {
    format!("{:.3}", ms as f64 / 1000.0)
}

// ─── Entry point ────────────────────────────────────────────────────────────────

/// Build the keyframe set for `mp4` into `out_dir` (the clip's `frames/`), writing
/// each kept frame as `at-<atMs>.jpg`. `start_ms`/`end_ms` trim the analysed span;
/// `duration_ms` is the probed clip length (drives the `efficient` fps budget).
pub fn extract_keyframes(
    mp4: &Path,
    out_dir: &Path,
    mode: DetailMode,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
    duration_ms: u64,
) -> anyhow::Result<KeyframeSet> {
    if matches!(mode, DetailMode::Transcript) {
        return Ok(KeyframeSet::default());
    }
    std::fs::create_dir_all(out_dir)?;
    let width = frame_width();

    match mode {
        DetailMode::Transcript => Ok(KeyframeSet::default()),
        DetailMode::Efficient => {
            let frames = extract_efficient(mp4, out_dir, start_ms, end_ms, duration_ms, width)?;
            Ok(KeyframeSet {
                frames,
                scan_warning: None,
            })
        }
        DetailMode::Balanced | DetailMode::TokenBurner => {
            let mut set = extract_scene(mp4, out_dir, mode, start_ms, end_ms, duration_ms, width)?;
            // Safety net: a short/low-motion clip may trip no scene changes at all.
            // Fall back to the duration-aware sampler so ingest never returns an
            // empty keyframe set (recommendedMoments must be non-empty).
            if set.frames.is_empty() {
                set.frames =
                    extract_efficient(mp4, out_dir, start_ms, end_ms, duration_ms, width)?;
                if !set.frames.is_empty() {
                    set.scan_warning = Some(
                        "no scene changes crossed the threshold; sampled at a fixed interval"
                            .to_string(),
                    );
                }
            }
            Ok(set)
        }
    }
}

// ─── Efficient: duration-aware fixed-fps sampling ────────────────────────────────

fn extract_efficient(
    mp4: &Path,
    out_dir: &Path,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
    duration_ms: u64,
    width: u32,
) -> anyhow::Result<Vec<Keyframe>> {
    let eff_ms = effective_span_ms(start_ms, end_ms, duration_ms);
    let dur_secs = (eff_ms as f64 / 1000.0).max(0.001);
    let target = env_u32("RYU_CLIP_EFFICIENT_FRAMES", DEFAULT_EFFICIENT_FRAMES).max(1);
    // fps = target frames / duration, clamped so a very short clip can't ask for a
    // firehose and a very long one still samples occasionally.
    let fps = (target as f64 / dur_secs).clamp(0.05, 30.0);

    let vf = format!("fps={fps:.4},scale={width}:-1");
    run_ffmpeg_extract(mp4, out_dir, &vf, start_ms, end_ms)?;

    // Frames are evenly spaced at `fps`; the Nth (0-based) output frame sits at
    // start + N/fps seconds.
    let raw = collect_raw_frames(out_dir);
    let base = start_ms.unwrap_or(0);
    let mut frames = Vec::with_capacity(raw.len());
    for (idx, raw_path) in raw.into_iter().enumerate() {
        let at_ms = base + ((idx as f64) * 1000.0 / fps).round() as u64;
        if let Some(kf) = rename_kept(out_dir, &raw_path, at_ms) {
            frames.push(kf);
        }
    }
    cleanup_raw(out_dir);
    Ok(frames)
}

// ─── Balanced / TokenBurner: scene detection + dedup + cap ───────────────────────

fn extract_scene(
    mp4: &Path,
    out_dir: &Path,
    mode: DetailMode,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
    duration_ms: u64,
    width: u32,
) -> anyhow::Result<KeyframeSet> {
    let threshold = env_f64("RYU_CLIP_SCENE_THRESHOLD", DEFAULT_SCENE_THRESHOLD);
    let meta_name = "scene-meta.txt";

    // `select` emits a frame whenever the scene score exceeds the threshold;
    // `metadata=print` writes the per-frame pts_time we recover atMs from. The
    // metadata sink uses a *relative* filename so we can side-step ffmpeg
    // filter-graph path escaping (colons/backslashes on Windows) by running with
    // `current_dir = out_dir`.
    let vf = format!(
        "select='gt(scene,{threshold})',metadata=print:file={meta_name},scale={width}:-1"
    );
    run_ffmpeg_extract(mp4, out_dir, &vf, start_ms, end_ms)?;

    let pts_times = parse_pts_times(&out_dir.join(meta_name));
    let raw = collect_raw_frames(out_dir);
    let base = start_ms.unwrap_or(0);

    // Zip each raw frame with its pts (both in chronological order). If ffmpeg's
    // metadata and file counts disagree, the shorter of the two bounds us.
    let paired: Vec<(u64, PathBuf)> = raw
        .into_iter()
        .zip(pts_times.into_iter())
        .map(|(path, pts)| (base + (pts * 1000.0).round() as u64, path))
        .collect();

    // Grayscale-delta dedup against the previous KEPT frame.
    let dedup_delta = env_f64("RYU_CLIP_DEDUP_DELTA", DEFAULT_DEDUP_DELTA);
    let mut kept: Vec<(u64, PathBuf)> = Vec::new();
    let mut prev_thumb: Option<Vec<u8>> = None;
    for (at_ms, path) in paired {
        let thumb = gray_thumb(&path);
        let is_dup = match (&prev_thumb, &thumb) {
            (Some(prev), Some(cur)) => mean_abs_delta(prev, cur) <= dedup_delta,
            _ => false,
        };
        if is_dup {
            let _ = std::fs::remove_file(&path);
            continue;
        }
        if thumb.is_some() {
            prev_thumb = thumb;
        }
        kept.push((at_ms, path));
    }

    // Cap `balanced`; `tokenBurner` is uncapped.
    let mut scan_warning = None;
    if matches!(mode, DetailMode::Balanced) {
        let cap = env_usize("RYU_CLIP_BALANCED_CAP", DEFAULT_BALANCED_CAP);
        if kept.len() > cap {
            let total = kept.len();
            // Delete the frames we're about to drop so only the kept subset stays.
            let subset = subsample_even(kept, cap);
            kept = subset;
            if duration_ms >= LONG_VIDEO_MS {
                scan_warning = Some(format!(
                    "sampled {} of {total} scene frames across a {}s video",
                    kept.len(),
                    duration_ms / 1000
                ));
            }
        }
    }

    // Rename the survivors to at-<atMs>.jpg (the frame-endpoint cache path).
    let mut frames = Vec::with_capacity(kept.len());
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for (at_ms, path) in kept {
        if !seen.insert(at_ms) {
            // Two frames rounded to the same ms — keep the first, drop the second.
            let _ = std::fs::remove_file(&path);
            continue;
        }
        if let Some(kf) = rename_kept(out_dir, &path, at_ms) {
            frames.push(kf);
        }
    }
    frames.sort_by_key(|f| f.at_ms);

    // Drop any leftover raw frames (subsampled-away, unpaired) + the metadata file.
    cleanup_raw(out_dir);
    let _ = std::fs::remove_file(out_dir.join(meta_name));

    Ok(KeyframeSet {
        frames,
        scan_warning,
    })
}

// ─── ffmpeg + IO helpers ─────────────────────────────────────────────────────────

/// Run one ffmpeg extraction pass writing `kf-%05d.jpg` sequentially into
/// `out_dir` (its working directory). `-ss` (input seek, resets pts to 0) is
/// placed before `-i`; `-to` (relative to the reset timeline) after it.
fn run_ffmpeg_extract(
    mp4: &Path,
    out_dir: &Path,
    vf: &str,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
) -> anyhow::Result<()> {
    let mp4_abs = std::fs::canonicalize(mp4).unwrap_or_else(|_| mp4.to_path_buf());

    let mut cmd = Command::new("ffmpeg");
    cmd.current_dir(out_dir);
    cmd.arg("-y");
    if let Some(s) = start_ms {
        cmd.args(["-ss", &secs_str(s)]);
    }
    cmd.arg("-i").arg(&mp4_abs);
    // `-to` after `-i` is relative to the (reset) timeline, i.e. end - start.
    if let Some(to) = trimmed_end_ms(start_ms, end_ms) {
        cmd.args(["-to", &secs_str(to)]);
    }
    cmd.args(["-vf", vf, "-vsync", "vfr", "-q:v", "2", "kf-%05d.jpg"]);

    let output = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("ffmpeg not found: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg keyframe extraction failed: {stderr}");
    }
    Ok(())
}

/// The end of the trimmed span on the post-seek (pts-reset) timeline.
fn trimmed_end_ms(start_ms: Option<u64>, end_ms: Option<u64>) -> Option<u64> {
    let end = end_ms?;
    Some(end.saturating_sub(start_ms.unwrap_or(0)).max(1))
}

/// The analysed span length, honouring an optional start/end trim.
fn effective_span_ms(start_ms: Option<u64>, end_ms: Option<u64>, duration_ms: u64) -> u64 {
    match (start_ms, end_ms) {
        (Some(s), Some(e)) if e > s => e - s,
        (Some(s), None) => duration_ms.saturating_sub(s).max(1),
        (None, Some(e)) => e.max(1),
        _ => duration_ms.max(1),
    }
}

/// The raw `kf-*.jpg` frames ffmpeg wrote, sorted chronologically by their index.
fn collect_raw_frames(out_dir: &Path) -> Vec<PathBuf> {
    let mut frames: Vec<(u64, PathBuf)> = std::fs::read_dir(out_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let stem = path.file_stem()?.to_str()?;
            let idx = stem.strip_prefix("kf-")?.parse::<u64>().ok()?;
            Some((idx, path))
        })
        .collect();
    frames.sort_by_key(|(idx, _)| *idx);
    frames.into_iter().map(|(_, p)| p).collect()
}

/// Rename a raw frame to the `at-<atMs>.jpg` cache path the frame endpoint reads.
fn rename_kept(out_dir: &Path, raw_path: &Path, at_ms: u64) -> Option<Keyframe> {
    let dest = out_dir.join(format!("at-{at_ms}.jpg"));
    match std::fs::rename(raw_path, &dest) {
        Ok(()) => Some(Keyframe { at_ms, path: dest }),
        Err(e) => {
            tracing::trace!("clip frames: rename {} failed: {e}", raw_path.display());
            None
        }
    }
}

/// Delete any leftover `kf-*.jpg` (dropped duplicates / subsampled-away frames).
fn cleanup_raw(out_dir: &Path) {
    for raw in collect_raw_frames(out_dir) {
        let _ = std::fs::remove_file(raw);
    }
}

/// Parse `pts_time:<secs>` values (one per emitted frame, in order) from an
/// ffmpeg `metadata=print` file.
fn parse_pts_times(meta_path: &Path) -> Vec<f64> {
    let content = std::fs::read_to_string(meta_path).unwrap_or_default();
    let mut times = Vec::new();
    for line in content.lines() {
        if let Some(idx) = line.find("pts_time:") {
            let rest = &line[idx + "pts_time:".len()..];
            if let Some(tok) = rest.split_whitespace().next() {
                if let Ok(t) = tok.parse::<f64>() {
                    times.push(t.max(0.0));
                }
            }
        }
    }
    times
}

/// Decode a frame and reduce it to a small grayscale thumbnail (raw luma bytes)
/// for the dedup delta. Mirrors the `image`-crate usage in `save_frame_jpeg`.
fn gray_thumb(path: &Path) -> Option<Vec<u8>> {
    let img = image::open(path).ok()?;
    let small = img
        .resize_exact(THUMB_EDGE, THUMB_EDGE, image::imageops::FilterType::Triangle)
        .to_luma8();
    Some(small.into_raw())
}

/// Mean absolute per-pixel difference (0–255) between two equal-length luma
/// thumbnails. Returns `f64::MAX` for a shape mismatch so unrelated frames are
/// never treated as duplicates.
fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::MAX;
    }
    let sum: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

/// Evenly subsample `items` down to at most `cap`, always keeping the endpoints.
fn subsample_even<T>(items: Vec<T>, cap: usize) -> Vec<T> {
    let n = items.len();
    if n <= cap || cap == 0 {
        return items;
    }
    let mut keep = vec![false; n];
    if cap == 1 {
        keep[0] = true;
    } else {
        let step = (n - 1) as f64 / (cap - 1) as f64;
        for i in 0..cap {
            let idx = ((i as f64) * step).round() as usize;
            keep[idx.min(n - 1)] = true;
        }
    }
    // Free the frames we drop so only the kept subset survives on disk.
    items
        .into_iter()
        .enumerate()
        .filter_map(|(i, item)| if keep[i] { Some(item) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_mode_defaults_to_balanced() {
        assert_eq!(DetailMode::default(), DetailMode::Balanced);
    }

    #[test]
    fn detail_mode_round_trips_camel_case() {
        assert_eq!(
            serde_json::to_string(&DetailMode::TokenBurner).unwrap(),
            "\"tokenBurner\""
        );
        let back: DetailMode = serde_json::from_str("\"efficient\"").unwrap();
        assert_eq!(back, DetailMode::Efficient);
    }

    #[test]
    fn mean_abs_delta_flags_identical_frames() {
        let a = vec![10u8; 64];
        let b = vec![10u8; 64];
        assert_eq!(mean_abs_delta(&a, &b), 0.0);
        let c = vec![200u8; 64];
        assert!(mean_abs_delta(&a, &c) > 2.0);
    }

    #[test]
    fn subsample_keeps_endpoints_and_cap() {
        let items: Vec<u64> = (0..10).collect();
        let out = subsample_even(items, 4);
        assert!(out.len() <= 4);
        assert_eq!(*out.first().unwrap(), 0);
        assert_eq!(*out.last().unwrap(), 9);
    }

    #[test]
    fn parse_pts_times_reads_frame_lines() {
        let meta = "frame:0    pts:0        pts_time:0\nlavfi.scene_score=0.4\nframe:1    pts:9000     pts_time:3.5\n";
        let dir = std::env::temp_dir().join(format!("ryu-pts-{}", uuid::Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.txt");
        std::fs::write(&path, meta).unwrap();
        let times = parse_pts_times(&path);
        assert_eq!(times, vec![0.0, 3.5]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
