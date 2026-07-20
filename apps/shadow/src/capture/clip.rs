//! Ryu Clips — device-local screen + audio recording for the agent-native
//! Loom/Jam feature.
//!
//! A "clip" is a one-click screen+audio recording bundled into an
//! **agent-readable** directory the desktop can attach into chat. Unlike the
//! passive timeline capture (`capture_engine.rs`) this is an explicit,
//! user-initiated session with a clear start/stop, its own frame sequence, and a
//! manifest (`agent-context.json`) that ties browser diagnostics (console /
//! network errors streamed in from the extension) to moments in the video.
//!
//! Lifecycle mirrors [`crate::capture::meeting`]: a process-global recorder so
//! the `/clips/*` HTTP handlers reach it without threading fields through
//! `AppState`. Screen frames are captured on a dedicated `std::thread` (its own
//! current-thread tokio runtime drives the async screenshot API); optional mic +
//! system-loopback audio ride cpal input streams. On stop we shell out to the
//! ffmpeg CLI (the same pattern as `video::FrameExtractor`, NOT the optional
//! `video` cargo feature) to mux the JPEG sequence (+ WAV) into `clip.mp4`, POST
//! the audio to Core's whisper endpoint for a transcript, and rewrite the
//! manifest.
//!
//! Placement (CLAUDE.md §1): this is the *sensor* half — it decides *what is
//! captured on this device*, so it lives in Shadow. Core owns *what runs* (the
//! clip session lifecycle it drives via the `/api/clips/*` proxy) and *what is
//! shared* (diagnostics redaction/DLP is a Gateway concern on egress). Shadow
//! only records and bundles.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;
use serde::{Deserialize, Serialize};

use crate::capture::mixer;

/// Default screen-capture rate (frames/sec). A clip is a low-motion demo, so 2
/// fps keeps the JPEG sequence + muxed MP4 small while staying watchable.
/// Overridable via `SHADOW_CLIP_FPS` — a swappable default, never hardcoded.
const DEFAULT_CLIP_FPS: u32 = 2;

/// Max JPEG width; wider frames are downscaled to keep the bundle small (mirrors
/// `video::VideoEncoder::save_keyframe`).
const MAX_FRAME_WIDTH: u32 = 1280;

/// Core's loopback port. The transcript POST target is validated against this so
/// a rogue `RYU_CORE_URL` can't exfiltrate audio off-box (SSRF guard, mirrors
/// `server::resolve_meeting_ingest_url`).
const CORE_PORT: u16 = 7980;

// ─── Wire types (camelCase, matching agent-context.json) ────────────────────────

/// Options for starting a clip, posted by Core (`POST /clips/start`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipStartOpts {
    #[serde(default = "default_true")]
    pub screen: bool,
    #[serde(default)]
    pub mic: bool,
    #[serde(default)]
    pub system_audio: bool,
    #[serde(default)]
    pub display_id: Option<u32>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub tab: Option<ClipTab>,
    /// Where to capture from. When present it supersedes the legacy `display_id`:
    /// `screen` grabs the primary display, `display` a chosen monitor, `window` a
    /// chosen window (best-effort — degrades to that window's display).
    #[serde(default)]
    pub target: Option<ClipTarget>,
}

fn default_true() -> bool {
    true
}

/// The capture source for a clip (wire shape
/// `{ "kind": "screen"|"display"|"window", "displayId"?, "windowId"? }`).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipTarget {
    pub kind: ClipTargetKind,
    #[serde(default)]
    pub display_id: Option<u32>,
    #[serde(default)]
    pub window_id: Option<u64>,
}

/// What a [`ClipTarget`] points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ClipTargetKind {
    /// The primary display, fullscreen (the legacy default).
    Screen,
    /// A specific monitor selected by `displayId`.
    Display,
    /// A specific window selected by `windowId` (best-effort).
    Window,
}

/// Options for ingesting an existing video (a watched URL or a local file) into
/// the SAME agent-context bundle a recorded clip produces. Posted by Core
/// (`POST /clips/ingest`) after it has resolved `video_path` to a local file
/// (yt-dlp for URLs; a validated path for local files) and, best-effort, pulled
/// `captions`. All fields default so a minimal `{ "videoPath": "..." }` works.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClipIngestOpts {
    /// Absolute path to the local source video Shadow should normalize + sample.
    pub video_path: String,
    pub title: Option<String>,
    /// Captions text (already parsed from a .vtt by Core). When non-empty this is
    /// used as the transcript and Whisper is skipped ("captions-first").
    pub captions: Option<String>,
    /// Timed caption cues (parsed from the same .vtt by Core). When present these
    /// populate the transcript segments on the captions-first path.
    pub caption_segments: Vec<TranscriptSegment>,
    pub detail: crate::capture::frames::DetailMode,
    /// Trim the analysed span to `[start, end)` ms (both optional).
    pub start: Option<u64>,
    pub end: Option<u64>,
    /// STT engine to use when there are no captions (threaded to Core as
    /// `?engine=`). `None` uses Core's default.
    pub stt_engine: Option<String>,
}

/// The browser tab a clip was recorded against (tagged by the extension).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipTab {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// What the clip captured.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipCapture {
    pub screen: bool,
    pub mic: bool,
    pub system_audio: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tab: Option<ClipTab>,
}

/// A moment worth jumping to, recomputed from diagnostics on each ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecommendedMoment {
    pub at_ms: u64,
    pub reason: String,
}

/// The clip manifest — `agent-context.json`. This is the whole point: an
/// agent-readable summary tying the video, transcript, and diagnostics together.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipContext {
    pub id: String,
    pub title: String,
    pub duration_ms: u64,
    pub created_at: String,
    pub t0_epoch_ms: i64,
    pub capture: ClipCapture,
    pub video: String,
    pub transcript_path: String,
    pub diagnostics_path: String,
    pub frames_endpoint: String,
    #[serde(default)]
    pub recommended_moments: Vec<RecommendedMoment>,
    /// A human-facing note from the keyframe extractor (e.g. a long ingested
    /// video was subsampled to the per-mode cap). Absent for recorded clips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan_warning: Option<String>,
}

/// A one-line clip for the picker (`GET /clips`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipSummary {
    pub id: String,
    pub title: String,
    pub duration_ms: u64,
    pub created_at: String,
}

/// One transcript segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// `agent-transcript.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptDoc {
    pub text: String,
    #[serde(default)]
    pub segments: Vec<TranscriptSegment>,
}

/// One diagnostic event streamed in from the extension (console / exception /
/// network). `t` is ms since `t0_epoch_ms` and may be negative.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticEvent {
    pub t: i64,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// `diagnostics.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticsDoc {
    #[serde(default)]
    pub events: Vec<DiagnosticEvent>,
}

// ─── One source's rolling mono buffer ───────────────────────────────────────────

/// A single cpal source (mic or loopback): its downmixed-mono samples and the
/// rate they were captured at. Drained once at stop.
struct SourceBuffer {
    samples: Mutex<Vec<f32>>,
    rate: u32,
}

impl SourceBuffer {
    fn new(rate: u32) -> Arc<Self> {
        Arc::new(Self {
            samples: Mutex::new(Vec::new()),
            rate,
        })
    }

    fn drain(&self) -> (Vec<f32>, u32) {
        let mut guard = self.samples.lock().unwrap_or_else(|e| e.into_inner());
        (std::mem::take(&mut *guard), self.rate)
    }
}

// ─── The active recorder ────────────────────────────────────────────────────────

/// An in-progress clip: the running flag the screen thread watches, its join
/// handle, the live cpal streams (kept alive so capture continues), and the
/// audio buffers drained on stop.
pub struct ClipRecorder {
    clip_id: String,
    t0_epoch_ms: i64,
    dir: PathBuf,
    fps: u32,
    running: Arc<AtomicBool>,
    /// True while the clip is paused: capture is suspended and the paused span is
    /// excluded from the final duration. Watched by the screen thread.
    paused: Arc<AtomicBool>,
    /// Sum of completed paused spans (ms), excluded from the clip duration.
    paused_accum_ms: Arc<AtomicU64>,
    /// Epoch-ms the current pause began (0 when not paused).
    pause_started_ms: Arc<AtomicI64>,
    screen_thread: Option<std::thread::JoinHandle<()>>,
    mic_stream: Option<Stream>,
    sys_stream: Option<Stream>,
    mic_buf: Option<Arc<SourceBuffer>>,
    sys_buf: Option<Arc<SourceBuffer>>,
    frame_count: Arc<AtomicUsize>,
    opts: ClipStartOpts,
}

// cpal `Stream` holds a non-Send callback; we only build it on start and drop it
// on stop, always from the handler thread, never sending it across threads — the
// same justification as `MeetingRecorder`.
unsafe impl Send for ClipRecorder {}
unsafe impl Sync for ClipRecorder {}

static CLIP: OnceLock<Mutex<Option<ClipRecorder>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<ClipRecorder>> {
    CLIP.get_or_init(|| Mutex::new(None))
}

// ─── Paths ──────────────────────────────────────────────────────────────────────

/// The Shadow data root: `SHADOW_DATA_DIR` if set, else `~/.shadow` (mirrors
/// `crates/shadow-core/src/config.rs`). Everything a clip writes lives under
/// `<root>/media/clips/<id>/`.
pub fn data_root() -> PathBuf {
    if let Ok(dir) = std::env::var("SHADOW_DATA_DIR") {
        let trimmed = dir.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".shadow")
}

fn clips_root() -> PathBuf {
    data_root().join("media").join("clips")
}

fn clip_dir(id: &str) -> PathBuf {
    clips_root().join(id)
}

fn context_path(id: &str) -> PathBuf {
    clip_dir(id).join("agent-context.json")
}

fn diagnostics_path(id: &str) -> PathBuf {
    clip_dir(id).join("diagnostics.json")
}

fn transcript_path(id: &str) -> PathBuf {
    clip_dir(id).join("agent-transcript.json")
}

fn frames_dir(id: &str) -> PathBuf {
    clip_dir(id).join("frames")
}

/// Path to the muxed clip video.
pub fn clip_file_path(id: &str) -> PathBuf {
    clip_dir(id).join("clip.mp4")
}

// ─── FPS ────────────────────────────────────────────────────────────────────────

fn clip_fps() -> u32 {
    std::env::var("SHADOW_CLIP_FPS")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_CLIP_FPS)
        .clamp(1, 30)
}

// ─── Lifecycle ──────────────────────────────────────────────────────────────────

/// Whether a clip is currently recording.
pub fn is_recording() -> bool {
    slot().lock().map(|g| g.is_some()).unwrap_or(false)
}

/// The id of the clip currently recording, if any.
pub fn current_clip_id() -> Option<String> {
    slot()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|r| r.clip_id.clone()))
}

/// Start a new clip. Tears down any prior recording first (like the meeting
/// recorder). Returns the initial manifest (durationMs 0).
pub fn start(opts: ClipStartOpts) -> anyhow::Result<ClipContext> {
    // Discard any prior in-progress clip without muxing it — start replaces.
    if let Ok(mut guard) = slot().lock() {
        if let Some(mut prev) = guard.take() {
            prev.running.store(false, Ordering::Relaxed);
            if let Some(handle) = prev.screen_thread.take() {
                let _ = handle.join();
            }
        }
    }

    let clip_id = format!("clip_{}", uuid::Uuid::new_v4().simple());
    let t0_epoch_ms = chrono::Utc::now().timestamp_millis();
    let dir = clip_dir(&clip_id);
    std::fs::create_dir_all(frames_dir(&clip_id))?;

    let fps = clip_fps();

    // Resolve the capture source. `target` (screen/display/window) supersedes the
    // legacy `display_id`; a window target degrades to its display with a note.
    let (resolved_display, target_note) = resolve_capture_display(&opts);

    // Seed the bundle so the files always exist even if the recorder crashes.
    write_doc(&diagnostics_path(&clip_id), &DiagnosticsDoc::default())?;
    write_doc(&transcript_path(&clip_id), &TranscriptDoc::default())?;

    let capture = ClipCapture {
        screen: opts.screen,
        mic: opts.mic,
        system_audio: opts.system_audio,
        tab: opts.tab.clone(),
    };
    let context = ClipContext {
        id: clip_id.clone(),
        title: opts
            .title
            .clone()
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| "Untitled clip".to_string()),
        duration_ms: 0,
        created_at: chrono::Utc::now().to_rfc3339(),
        t0_epoch_ms,
        capture,
        video: "clip.mp4".to_string(),
        transcript_path: "agent-transcript.json".to_string(),
        diagnostics_path: "diagnostics.json".to_string(),
        frames_endpoint: format!("/clips/{clip_id}/frame"),
        recommended_moments: Vec::new(),
        scan_warning: target_note.clone(),
    };
    write_doc(&context_path(&clip_id), &context)?;

    let running = Arc::new(AtomicBool::new(true));
    let paused = Arc::new(AtomicBool::new(false));
    let paused_accum_ms = Arc::new(AtomicU64::new(0));
    let pause_started_ms = Arc::new(AtomicI64::new(0));
    let frame_count = Arc::new(AtomicUsize::new(0));

    // ── Screen thread ────────────────────────────────────────────────────────
    let screen_thread = if opts.screen {
        let running_t = Arc::clone(&running);
        let paused_t = Arc::clone(&paused);
        let frame_count_t = Arc::clone(&frame_count);
        let out_dir = frames_dir(&clip_id);
        let display_id = resolved_display;
        let frame_interval = std::time::Duration::from_millis((1000 / fps.max(1)) as u64);
        let handle = std::thread::spawn(move || {
            // The screenshot API is async; drive it from a private current-thread
            // runtime so this OS thread stays fully self-contained.
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::warn!("clip: could not build screen-capture runtime: {e}");
                    return;
                }
            };
            while running_t.load(Ordering::Relaxed) {
                let loop_start = std::time::Instant::now();

                // Honour the same consent gates the passive keyframe path uses:
                // never write a frame while paused, when frames are disabled, or
                // for a non-allowlisted app (best-effort app name from OCR cache).
                let app = crate::capture_engine::get_latest_ocr()
                    .map(|(_, app, _)| app)
                    .unwrap_or_default();
                let allowed = crate::server::is_frame_capture_enabled()
                    && !crate::server::is_capture_paused()
                    && !paused_t.load(Ordering::Relaxed)
                    && crate::server::is_capture_allowed(&app);

                if allowed {
                    match rt.block_on(crate::capture::screen::quick_screenshot(display_id)) {
                        Ok(frame) => {
                            let seq = frame_count_t.load(Ordering::Relaxed) + 1;
                            let path = out_dir.join(format!("seq-{seq:06}.jpg"));
                            if let Err(e) = save_frame_jpeg(&frame, &path) {
                                tracing::trace!("clip: frame encode failed: {e}");
                            } else {
                                frame_count_t.store(seq, Ordering::Relaxed);
                            }
                        }
                        Err(e) => {
                            // DXGI WAIT_TIMEOUT (0x887A0027) is a normal no-new-frame
                            // skip at low fps, not a real failure.
                            let msg = e.to_string();
                            if !msg.contains("0x887A0027") && !msg.contains("AcquireNextFrame") {
                                tracing::trace!("clip: screenshot failed: {e}");
                            }
                        }
                    }
                }

                let elapsed = loop_start.elapsed();
                if let Some(rem) = frame_interval.checked_sub(elapsed) {
                    std::thread::sleep(rem);
                }
            }
        });
        Some(handle)
    } else {
        None
    };

    // ── Audio (best-effort) ──────────────────────────────────────────────────
    let host = cpal::default_host();
    let (mic_stream, mic_buf) = if opts.mic {
        match build_mic_stream(&host) {
            Ok((stream, buf)) => (Some(stream), Some(buf)),
            Err(e) => {
                tracing::warn!("clip: microphone capture unavailable ({e}); recording without mic");
                (None, None)
            }
        }
    } else {
        (None, None)
    };
    let (sys_stream, sys_buf) = if opts.system_audio {
        match build_loopback_stream(&host) {
            Ok((stream, buf)) => (Some(stream), Some(buf)),
            Err(e) => {
                tracing::warn!("clip: system-audio loopback unavailable ({e}); recording without it");
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    let recorder = ClipRecorder {
        clip_id,
        t0_epoch_ms,
        dir,
        fps,
        running,
        paused,
        paused_accum_ms,
        pause_started_ms,
        screen_thread,
        mic_stream,
        sys_stream,
        mic_buf,
        sys_buf,
        frame_count,
        opts,
    };
    if let Ok(mut guard) = slot().lock() {
        *guard = Some(recorder);
    }
    Ok(context)
}

/// Resolve the display to capture (and an optional note) from a clip's options.
/// `target` supersedes the legacy `display_id`. A `window` target is best-effort:
/// per-window capture is not implemented, so it degrades to that window's display
/// and returns a note the manifest surfaces as `scanWarning`.
fn resolve_capture_display(opts: &ClipStartOpts) -> (u32, Option<String>) {
    let Some(target) = &opts.target else {
        return (opts.display_id.unwrap_or(0), None);
    };
    match target.kind {
        ClipTargetKind::Screen => (0, None),
        ClipTargetKind::Display => (
            target.display_id.or(opts.display_id).unwrap_or(0),
            None,
        ),
        ClipTargetKind::Window => match target.window_id {
            Some(window_id) => {
                let display = window_display_id(window_id).unwrap_or(0);
                (
                    display,
                    Some(format!(
                        "per-window capture is unavailable; recorded display {display} (the display containing the selected window)"
                    )),
                )
            }
            None => (
                opts.display_id.unwrap_or(0),
                Some("window target missing windowId; recorded the primary display".to_string()),
            ),
        },
    }
}

/// The display index a window sits on (Windows only; `None` elsewhere or when the
/// window can't be located).
#[cfg(windows)]
fn window_display_id(window_id: u64) -> Option<u32> {
    crate::capture::screen::display_id_for_window(window_id)
}

#[cfg(not(windows))]
fn window_display_id(_window_id: u64) -> Option<u32> {
    None
}

/// Pause the in-progress clip: suspend frame + audio capture and open a paused
/// span excluded from the final duration (`t0` is unchanged). Idempotent. Returns
/// the clip id, or `None` when nothing is recording.
pub fn pause() -> Option<String> {
    let mut guard = slot().lock().ok()?;
    let rec = guard.as_mut()?;
    if !rec.paused.swap(true, Ordering::SeqCst) {
        rec.pause_started_ms
            .store(chrono::Utc::now().timestamp_millis(), Ordering::SeqCst);
        // Halt audio capture for the paused span (cpal keeps the stream alive).
        if let Some(s) = &rec.mic_stream {
            let _ = s.pause();
        }
        if let Some(s) = &rec.sys_stream {
            let _ = s.pause();
        }
    }
    Some(rec.clip_id.clone())
}

/// Resume a paused clip: close the paused span (adding it to `paused_accum_ms`)
/// and restart capture. Idempotent. Returns the clip id, or `None` when nothing
/// is recording.
pub fn resume() -> Option<String> {
    let mut guard = slot().lock().ok()?;
    let rec = guard.as_mut()?;
    if rec.paused.swap(false, Ordering::SeqCst) {
        let started = rec.pause_started_ms.swap(0, Ordering::SeqCst);
        if started > 0 {
            let now = chrono::Utc::now().timestamp_millis();
            let span = now.saturating_sub(started).max(0) as u64;
            rec.paused_accum_ms.fetch_add(span, Ordering::SeqCst);
        }
        if let Some(s) = &rec.mic_stream {
            let _ = s.play();
        }
        if let Some(s) = &rec.sys_stream {
            let _ = s.play();
        }
    }
    Some(rec.clip_id.clone())
}

/// Total paused time (ms) so far, including an in-progress pause up to `now_ms`.
fn total_paused_ms(rec: &ClipRecorder, now_ms: i64) -> u64 {
    let accum = rec.paused_accum_ms.load(Ordering::SeqCst);
    let started = rec.pause_started_ms.load(Ordering::SeqCst);
    if started > 0 {
        accum + (now_ms.saturating_sub(started).max(0) as u64)
    } else {
        accum
    }
}

/// The in-progress clip's live duration (ms), excluding paused spans, or `None`
/// when nothing is recording. Lets pause/resume return an authoritative elapsed
/// value the desktop timer syncs to instead of counting wall-clock over pauses.
pub fn live_duration_ms() -> Option<u64> {
    let guard = slot().lock().ok()?;
    let rec = guard.as_ref()?;
    let now_ms = chrono::Utc::now().timestamp_millis();
    let gross = now_ms.saturating_sub(rec.t0_epoch_ms).max(0) as u64;
    Some(gross.saturating_sub(total_paused_ms(rec, now_ms)))
}

/// Stop the current clip: halt capture, mux the frames (+audio) to `clip.mp4` via
/// ffmpeg, transcribe the audio through Core, and rewrite the manifest with the
/// final duration + recommended moments. Returns the manifest, or `None` if
/// nothing was recording.
///
/// This blocks on ffmpeg + a network round-trip, so callers must run it off the
/// async runtime (the HTTP handler wraps it in `spawn_blocking`).
pub fn stop() -> anyhow::Result<Option<ClipContext>> {
    let mut recorder = match slot().lock() {
        Ok(mut guard) => match guard.take() {
            Some(r) => r,
            None => return Ok(None),
        },
        Err(_) => return Ok(None),
    };

    recorder.running.store(false, Ordering::Relaxed);
    if let Some(handle) = recorder.screen_thread.take() {
        let _ = handle.join();
    }

    // Drain + drop the cpal streams to halt audio capture.
    let mic = recorder.mic_buf.as_ref().map(|b| b.drain());
    let sys = recorder.sys_buf.as_ref().map(|b| b.drain());
    recorder.mic_stream.take();
    recorder.sys_stream.take();

    let id = recorder.clip_id.clone();
    let frames = recorder.frame_count.load(Ordering::Relaxed);

    // Build the mixed 16 kHz mono WAV (if any audio was captured) and persist it
    // next to the frames so ffmpeg can mux it.
    let audio_wav_path = recorder.dir.join("audio.wav");
    let has_audio = write_mixed_wav(&mic, &sys, &audio_wav_path);

    // Mux frames (+ audio) → clip.mp4. Leave the frame sequence as a fallback on
    // failure so the bundle is never empty.
    if frames > 0 {
        if let Err(e) = mux_clip(&recorder.dir, recorder.fps, has_audio) {
            tracing::warn!("clip {id}: ffmpeg mux failed ({e}); keeping frame sequence");
        }
    }

    // Transcribe via Core's whisper endpoint (fail-soft: empty transcript on any
    // error, never abort stop).
    if has_audio {
        match std::fs::read(&audio_wav_path) {
            Ok(bytes) => {
                let doc = transcribe_via_core(bytes, None);
                let _ = write_doc(&transcript_path(&id), &doc);
            }
            Err(e) => tracing::warn!("clip {id}: reading audio for transcription failed: {e}"),
        }
    }

    // Finalize the manifest: real duration + recommended moments from whatever
    // diagnostics arrived during the recording.
    let now_ms = chrono::Utc::now().timestamp_millis();
    let paused_ms = total_paused_ms(&recorder, now_ms);
    let duration_ms = (now_ms.saturating_sub(recorder.t0_epoch_ms).max(0) as u64)
        .saturating_sub(paused_ms);

    let mut context = read_context(&id).unwrap_or_else(|_| ClipContext {
        id: id.clone(),
        title: recorder
            .opts
            .title
            .clone()
            .unwrap_or_else(|| "Untitled clip".to_string()),
        duration_ms: 0,
        created_at: chrono::Utc::now().to_rfc3339(),
        t0_epoch_ms: recorder.t0_epoch_ms,
        capture: ClipCapture {
            screen: recorder.opts.screen,
            mic: recorder.opts.mic,
            system_audio: recorder.opts.system_audio,
            tab: recorder.opts.tab.clone(),
        },
        video: "clip.mp4".to_string(),
        transcript_path: "agent-transcript.json".to_string(),
        diagnostics_path: "diagnostics.json".to_string(),
        frames_endpoint: format!("/clips/{id}/frame"),
        recommended_moments: Vec::new(),
        scan_warning: None,
    });
    context.duration_ms = duration_ms;
    context.recommended_moments = recommended_from_disk(&id);
    write_doc(&context_path(&id), &context)?;

    Ok(Some(context))
}

/// Read a clip's manifest (`agent-context.json`).
pub fn read_context(id: &str) -> anyhow::Result<ClipContext> {
    let bytes = std::fs::read(context_path(id))?;
    let ctx: ClipContext = serde_json::from_slice(&bytes)?;
    Ok(ctx)
}

/// List all clips, newest first.
pub fn list() -> anyhow::Result<Vec<ClipSummary>> {
    let root = clips_root();
    let mut out: Vec<ClipSummary> = Vec::new();
    let entries = match std::fs::read_dir(&root) {
        Ok(e) => e,
        Err(_) => return Ok(out), // no clips yet
    };
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        if let Ok(ctx) = read_context(&id) {
            out.push(ClipSummary {
                id: ctx.id,
                title: ctx.title,
                duration_ms: ctx.duration_ms,
                created_at: ctx.created_at,
            });
        }
    }
    // Newest first by createdAt (RFC3339 sorts lexicographically by time).
    out.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(out)
}

/// Append diagnostics to a clip, recompute recommended moments into the manifest,
/// and return the new total event count.
pub fn append_diagnostics(id: &str, events: Vec<DiagnosticEvent>) -> anyhow::Result<usize> {
    let path = diagnostics_path(id);
    let mut doc: DiagnosticsDoc = match std::fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(_) => DiagnosticsDoc::default(),
    };
    doc.events.extend(events);
    let count = doc.events.len();
    write_doc(&path, &doc)?;

    // Recompute recommended moments into the manifest (best-effort).
    if let Ok(mut ctx) = read_context(id) {
        ctx.recommended_moments = recommended_from_events(&doc.events);
        let _ = write_doc(&context_path(id), &ctx);
    }

    Ok(count)
}

/// Extract a single frame at `at_ms` as a JPEG, returning its path. Serves from
/// the on-demand cache (`frames/at-<atMs>.jpg`) when present, else shells ffmpeg
/// against `clip.mp4`, else falls back to the nearest capture-time `seq-*.jpg`.
pub fn extract_frame(id: &str, at_ms: u64) -> anyhow::Result<PathBuf> {
    let cached = frames_dir(id).join(format!("at-{at_ms}.jpg"));
    if cached.exists() {
        return Ok(cached);
    }

    let mp4 = clip_file_path(id);
    if mp4.exists() {
        let secs = at_ms as f64 / 1000.0;
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-ss",
                &format!("{secs}"),
                "-i",
                mp4.to_str().unwrap_or(""),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                cached.to_str().unwrap_or(""),
            ])
            .output();
        match output {
            Ok(out) if out.status.success() && cached.exists() => return Ok(cached),
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tracing::debug!("clip {id}: ffmpeg frame extract failed: {stderr}");
            }
            Err(e) => tracing::debug!("clip {id}: ffmpeg not available: {e}"),
        }
    }

    // Fallback: nearest capture-time frame. Frames are ~evenly spaced at the clip
    // fps, so map the requested time onto the sequence index.
    if let Some(nearest) = nearest_seq_frame(id, at_ms) {
        return Ok(nearest);
    }

    anyhow::bail!("no frame available for clip {id} at {at_ms}ms")
}

// ─── Ingest (URL / file → the same agent-context bundle) ─────────────────────────

/// Ingest an existing video into a clip bundle indistinguishable from a recorded
/// one: normalize the input to `clip.mp4`, run the budgeted keyframe extractor
/// (scene-detected `at-<atMs>.jpg` frames), derive recommended moments from those
/// keyframes, transcribe (captions-first, else Core STT), and write the same
/// `agent-context.json` / `diagnostics.json` / `agent-transcript.json` trio via
/// the existing [`write_doc`] + path helpers.
///
/// Blocks on ffmpeg (+ a possible network round-trip for transcription), so the
/// HTTP handler runs it under `spawn_blocking`.
pub fn ingest(opts: ClipIngestOpts) -> anyhow::Result<ClipContext> {
    let src = PathBuf::from(opts.video_path.trim());
    if opts.video_path.trim().is_empty() || !src.is_file() {
        anyhow::bail!("videoPath does not point to an existing file");
    }

    let id = format!("clip_{}", uuid::Uuid::new_v4().simple());
    let clip_dir = clip_dir(&id);
    let frames_dir = frames_dir(&id);
    std::fs::create_dir_all(&frames_dir)?;

    // Normalize the source into the canonical clip.mp4 (fast `-c copy` for mp4).
    let clip_mp4 = clip_file_path(&id);
    normalize_to_mp4(&src, &clip_mp4)?;

    let duration_ms = probe_duration_ms(&clip_mp4);
    let has_audio = probe_has_audio(&clip_mp4);

    // Budgeted keyframe set → recommended moments (keyframe atMs, "scene change").
    let set = crate::capture::frames::extract_keyframes(
        &clip_mp4,
        &frames_dir,
        opts.detail,
        opts.start,
        opts.end,
        duration_ms,
    )?;
    let mut recommended_moments: Vec<RecommendedMoment> = set
        .frames
        .iter()
        .map(|f| RecommendedMoment {
            at_ms: f.at_ms,
            reason: "scene change".to_string(),
        })
        .collect();
    recommended_moments.sort_by_key(|m| m.at_ms);

    // Transcript: captions-first, else extract audio + transcribe through Core.
    let transcript = match opts
        .captions
        .as_deref()
        .map(str::trim)
        .filter(|c| !c.is_empty())
    {
        Some(captions) => TranscriptDoc {
            text: captions.to_string(),
            segments: opts.caption_segments.clone(),
        },
        None => {
            let audio_path = clip_dir.join("audio.wav");
            match extract_audio_wav(&clip_mp4, &audio_path) {
                Ok(true) => match std::fs::read(&audio_path) {
                    Ok(bytes) => transcribe_via_core(bytes, opts.stt_engine.as_deref()),
                    Err(e) => {
                        tracing::warn!("clip {id}: reading extracted audio failed: {e}");
                        TranscriptDoc::default()
                    }
                },
                _ => TranscriptDoc::default(),
            }
        }
    };

    let now = chrono::Utc::now();
    let title = opts
        .title
        .clone()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| default_ingest_title(&src));

    let context = ClipContext {
        id: id.clone(),
        title,
        duration_ms,
        created_at: now.to_rfc3339(),
        t0_epoch_ms: now.timestamp_millis(),
        capture: ClipCapture {
            screen: true,
            mic: false,
            system_audio: has_audio,
            tab: None,
        },
        video: "clip.mp4".to_string(),
        transcript_path: "agent-transcript.json".to_string(),
        diagnostics_path: "diagnostics.json".to_string(),
        frames_endpoint: format!("/clips/{id}/frame"),
        recommended_moments,
        scan_warning: set.scan_warning,
    };

    // Same bundle writer + trio as the recorded path (mirrors append_diagnostics).
    write_doc(&context_path(&id), &context)?;
    write_doc(&diagnostics_path(&id), &DiagnosticsDoc::default())?;
    write_doc(&transcript_path(&id), &transcript)?;

    Ok(context)
}

/// A readable default clip title from the source file stem.
fn default_ingest_title(src: &std::path::Path) -> String {
    src.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Ingested clip".to_string())
}

/// Normalize an arbitrary input video into `dest` (`clip.mp4`). Fast-path a
/// stream copy when the source is already mp4 (mirrors `mux_clip`'s ffmpeg
/// shell-out); otherwise re-encode H.264/AAC yuv420p for broad playability.
fn normalize_to_mp4(src: &std::path::Path, dest: &std::path::Path) -> anyhow::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let is_mp4 = src
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("mp4"))
        .unwrap_or(false);

    if is_mp4 {
        let copied = std::process::Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                src.to_str().unwrap_or(""),
                "-c",
                "copy",
                dest.to_str().unwrap_or(""),
            ])
            .output();
        if let Ok(out) = copied {
            if out.status.success() && dest.exists() {
                return Ok(());
            }
        }
        // Fall through to a full re-encode if the copy failed (e.g. odd codec).
    }

    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            src.to_str().unwrap_or(""),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            dest.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("ffmpeg not found: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg normalize failed: {stderr}");
    }
    Ok(())
}

/// Probe the clip duration in ms. Prefers `ffprobe format=duration`; falls back
/// to parsing `Duration:` out of `ffmpeg -i` stderr. Returns 0 when unknown.
fn probe_duration_ms(mp4: &std::path::Path) -> u64 {
    if let Ok(out) = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            mp4.to_str().unwrap_or(""),
        ])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Ok(secs) = text.trim().parse::<f64>() {
                if secs > 0.0 {
                    return (secs * 1000.0) as u64;
                }
            }
        }
    }

    // Fallback: ffmpeg prints `Duration: HH:MM:SS.xx` to stderr.
    if let Ok(out) = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-i", mp4.to_str().unwrap_or("")])
        .output()
    {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if let Some(ms) = parse_ffmpeg_duration(&stderr) {
            return ms;
        }
    }
    0
}

/// Parse a `Duration: HH:MM:SS.xx` token from ffmpeg's stderr into ms.
fn parse_ffmpeg_duration(stderr: &str) -> Option<u64> {
    let idx = stderr.find("Duration:")?;
    let rest = &stderr[idx + "Duration:".len()..];
    let token = rest.trim_start().split(',').next()?.trim();
    let mut parts = token.split(':');
    let h: f64 = parts.next()?.trim().parse().ok()?;
    let m: f64 = parts.next()?.trim().parse().ok()?;
    let s: f64 = parts.next()?.trim().parse().ok()?;
    let total = (h * 3600.0 + m * 60.0 + s) * 1000.0;
    if total > 0.0 {
        Some(total as u64)
    } else {
        None
    }
}

/// Whether the video carries an audio stream (drives `capture.system_audio`).
fn probe_has_audio(mp4: &std::path::Path) -> bool {
    std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            mp4.to_str().unwrap_or(""),
        ])
        .output()
        .map(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false)
}

/// Extract a 16 kHz mono WAV for transcription. Returns `Ok(false)` when the
/// source has no audio (an empty/near-empty WAV), so the caller skips STT.
fn extract_audio_wav(mp4: &std::path::Path, dest: &std::path::Path) -> anyhow::Result<bool> {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i",
            mp4.to_str().unwrap_or(""),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            dest.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("ffmpeg not found: {e}"))?;

    if !output.status.success() {
        // No audio stream (or a decode error) — not fatal for ingest.
        return Ok(false);
    }
    // A bare WAV header is 44 bytes; anything larger carries samples.
    let has_samples = std::fs::metadata(dest)
        .map(|m| m.len() > 44)
        .unwrap_or(false);
    Ok(has_samples)
}

// ─── Helpers ────────────────────────────────────────────────────────────────────

fn write_doc<T: Serialize>(path: &std::path::Path, doc: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec_pretty(doc)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Encode a BGRA `Frame` to a JPEG on disk (q80), downscaling wide frames. Mirrors
/// `video::VideoEncoder::save_keyframe` (BGRA→RGB swap, drop alpha).
fn save_frame_jpeg(
    frame: &crate::capture::screen::Frame,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let (w, h) = (frame.width, frame.height);
    let expected = w as usize * h as usize * 4;
    if frame.data.len() < expected {
        anyhow::bail!("frame buffer too small: {} < {expected}", frame.data.len());
    }

    let mut rgb = image::RgbImage::new(w, h);
    for (px, src) in rgb.pixels_mut().zip(frame.data.chunks_exact(4)) {
        *px = image::Rgb([src[2], src[1], src[0]]);
    }

    let img = if w > MAX_FRAME_WIDTH {
        let new_h = ((h as u64 * MAX_FRAME_WIDTH as u64) / w as u64).max(1) as u32;
        image::imageops::resize(&rgb, MAX_FRAME_WIDTH, new_h, image::imageops::FilterType::Triangle)
    } else {
        rgb
    };

    let mut buf = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 80);
    encoder.encode_image(&img)?;
    std::fs::write(path, &buf)?;
    Ok(())
}

/// Build a mic capture stream + its mono buffer.
fn build_mic_stream(host: &cpal::Host) -> anyhow::Result<(Stream, Arc<SourceBuffer>)> {
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no microphone input device available"))?;
    let cfg = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("no default input config: {e}"))?;
    let buf = SourceBuffer::new(cfg.sample_rate().0);
    let stream = build_capture_stream(&device, &cfg, Arc::clone(&buf))?;
    stream.play()?;
    Ok((stream, buf))
}

/// Build a system-audio (loopback) capture stream + its mono buffer. On Windows
/// this is WASAPI loopback (an input stream on the default output device); on
/// other platforms it looks for a monitor / virtual device by name.
fn build_loopback_stream(host: &cpal::Host) -> anyhow::Result<(Stream, Arc<SourceBuffer>)> {
    #[cfg(windows)]
    {
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("no default output device"))?;
        let cfg = device
            .default_output_config()
            .map_err(|e| anyhow::anyhow!("no default output config: {e}"))?;
        let buf = SourceBuffer::new(cfg.sample_rate().0);
        let stream = build_capture_stream(&device, &cfg, Arc::clone(&buf))?;
        stream.play()?;
        Ok((stream, buf))
    }

    #[cfg(not(windows))]
    {
        let device = find_loopback_device(host)
            .ok_or_else(|| anyhow::anyhow!("no system-audio capture device found"))?;
        let cfg = device
            .default_input_config()
            .map_err(|e| anyhow::anyhow!("no default input config for loopback device: {e}"))?;
        let buf = SourceBuffer::new(cfg.sample_rate().0);
        let stream = build_capture_stream(&device, &cfg, Arc::clone(&buf))?;
        stream.play()?;
        Ok((stream, buf))
    }
}

#[cfg(not(windows))]
fn find_loopback_device(host: &cpal::Host) -> Option<cpal::Device> {
    const HINTS: &[&str] = &[
        "monitor",
        "blackhole",
        "loopback",
        "aggregate",
        "soundflower",
        "pipewire",
    ];
    let devices = host.input_devices().ok()?;
    for device in devices {
        if let Ok(name) = device.name() {
            let lower = name.to_lowercase();
            if HINTS.iter().any(|h| lower.contains(h)) {
                return Some(device);
            }
        }
    }
    None
}

/// Build an input (capture) stream that downmixes every frame to mono and appends
/// it to `buf`. Works for a mic (capture device) and loopback (render device).
fn build_capture_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    buf: Arc<SourceBuffer>,
) -> anyhow::Result<Stream> {
    let channels = config.channels().max(1) as usize;
    let stream = device
        .build_input_stream(
            &config.clone().into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut samples) = buf.samples.lock() {
                    for frame in data.chunks(channels) {
                        let sum: f32 = frame.iter().copied().sum();
                        samples.push(sum / channels as f32);
                    }
                }
            },
            |e| tracing::warn!("clip: audio stream error: {e}"),
            None,
        )
        .map_err(|e| anyhow::anyhow!("build_input_stream: {e}"))?;
    Ok(stream)
}

/// Mix captured mic + system audio to a single 16 kHz mono WAV. Returns true when
/// a non-empty track was written.
fn write_mixed_wav(
    mic: &Option<(Vec<f32>, u32)>,
    sys: &Option<(Vec<f32>, u32)>,
    path: &std::path::Path,
) -> bool {
    let (mic_samples, mic_rate) = mic
        .as_ref()
        .map(|(s, r)| (s.as_slice(), *r))
        .unwrap_or((&[], mixer::TARGET_RATE));
    let (sys_samples, sys_rate) = sys
        .as_ref()
        .map(|(s, r)| (s.as_slice(), *r))
        .unwrap_or((&[], mixer::TARGET_RATE));

    if mic_samples.is_empty() && sys_samples.is_empty() {
        return false;
    }

    // Reuse the meeting DSP: resample each side to 16 kHz, normalize, sum, limit.
    let mixed = mixer::mix_for_asr(mic_samples, mic_rate, sys_samples, sys_rate, false);
    if mixed.is_empty() {
        return false;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: mixer::TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let writer = hound::WavWriter::create(path, spec);
    let mut writer = match writer {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("clip: WAV create failed: {e}");
            return false;
        }
    };
    for &s in &mixed {
        let pcm = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        if writer.write_sample(pcm).is_err() {
            return false;
        }
    }
    writer.finalize().is_ok()
}

/// Mux the JPEG frame sequence (+ optional audio) into `clip.mp4` via the ffmpeg
/// CLI (same shell-out pattern as `video::FrameExtractor`, not the `video` cargo
/// feature).
fn mux_clip(dir: &std::path::Path, fps: u32, has_audio: bool) -> anyhow::Result<()> {
    let frames_pattern = dir.join("frames").join("seq-%06d.jpg");
    let audio = dir.join("audio.wav");
    let out = dir.join("clip.mp4");

    let mut args: Vec<String> = vec![
        "-y".into(),
        "-framerate".into(),
        fps.to_string(),
        "-i".into(),
        frames_pattern.to_string_lossy().to_string(),
    ];
    if has_audio {
        args.push("-i".into());
        args.push(audio.to_string_lossy().to_string());
    }
    args.push("-c:v".into());
    args.push("libx264".into());
    args.push("-pix_fmt".into());
    args.push("yuv420p".into());
    if has_audio {
        args.push("-c:a".into());
        args.push("aac".into());
        args.push("-shortest".into());
    }
    args.push(out.to_string_lossy().to_string());

    let output = std::process::Command::new("ffmpeg")
        .args(&args)
        .output()
        .map_err(|e| anyhow::anyhow!("ffmpeg not found: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg mux failed: {stderr}");
    }
    Ok(())
}

/// POST the clip audio to Core's whisper endpoint and wrap the text in a
/// `TranscriptDoc`. `engine` selects the Core STT engine (`?engine=<engine>`);
/// `None` uses Core's default. Fail-soft: any error yields an empty transcript.
fn transcribe_via_core(bytes: Vec<u8>, engine: Option<&str>) -> TranscriptDoc {
    let url = match core_transcribe_url(engine) {
        Ok(u) => u,
        Err(e) => {
            tracing::warn!("clip: transcription target rejected: {e}");
            return TranscriptDoc::default();
        }
    };

    // A private current-thread runtime so this sync fn (called under
    // spawn_blocking) can issue the async multipart POST without a blocking dep.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            tracing::warn!("clip: transcription runtime build failed: {e}");
            return TranscriptDoc::default();
        }
    };

    rt.block_on(async move {
        let part = match reqwest::multipart::Part::bytes(bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")
        {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("clip: building transcription upload failed: {e}");
                return TranscriptDoc::default();
            }
        };
        let form = reqwest::multipart::Form::new().part("file", part);
        let client = reqwest::Client::new();
        match client.post(&url).multipart(form).send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<serde_json::Value>().await {
                Ok(body) => {
                    let text = body
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let segments = parse_core_segments(&body);
                    TranscriptDoc { text, segments }
                }
                Err(e) => {
                    tracing::warn!("clip: parsing transcription response failed: {e}");
                    TranscriptDoc::default()
                }
            },
            Ok(resp) => {
                tracing::warn!("clip: transcription returned HTTP {}", resp.status());
                TranscriptDoc::default()
            }
            Err(e) => {
                tracing::warn!("clip: transcription request failed: {e}");
                TranscriptDoc::default()
            }
        }
    })
}

/// Parse timestamped segments from Core's `/api/voice/transcribe` response. Core
/// serializes segments camelCase (`startMs`/`endMs`/`text`); an absent or
/// malformed `segments` array yields an empty vec (the transcript keeps its text).
fn parse_core_segments(body: &serde_json::Value) -> Vec<TranscriptSegment> {
    body.get("segments")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| {
                    let start_ms = s.get("startMs").and_then(serde_json::Value::as_u64)?;
                    let end_ms = s.get("endMs").and_then(serde_json::Value::as_u64)?;
                    let text = s
                        .get("text")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    Some(TranscriptSegment {
                        start_ms,
                        end_ms,
                        text,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Resolve + validate Core's transcription URL (`RYU_CORE_URL` else loopback
/// Core). The SSRF guard mirrors `server::resolve_meeting_ingest_url`: audio may
/// only be POSTed to loopback Core on its known port. `engine`, when present,
/// is appended as `?engine=<engine>` so the clip path can pick a swappable STT
/// engine (e.g. the Gateway-routed Whisper) without changing Core's default.
fn core_transcribe_url(engine: Option<&str>) -> Result<String, String> {
    let base = std::env::var("RYU_CORE_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("http://127.0.0.1:{CORE_PORT}"));
    let parsed = reqwest::Url::parse(&base).map_err(|_| "invalid RYU_CORE_URL".to_string())?;
    if parsed.scheme() != "http" {
        return Err("Core URL must use http".to_string());
    }
    let host = parsed.host_str().unwrap_or_default();
    if !matches!(host, "127.0.0.1" | "localhost" | "::1") {
        return Err("Core URL must point to loopback".to_string());
    }
    if parsed.port_or_known_default() != Some(CORE_PORT) {
        return Err("Core URL must use Core's loopback port".to_string());
    }
    let mut url = reqwest::Url::parse(&format!(
        "http://{host}:{CORE_PORT}/api/voice/transcribe"
    ))
    .map_err(|_| "could not build transcription URL".to_string())?;
    if let Some(engine) = engine.map(str::trim).filter(|e| !e.is_empty()) {
        url.query_pairs_mut().append_pair("engine", engine);
    }
    Ok(url.to_string())
}

/// Recompute recommended moments from the on-disk diagnostics.
fn recommended_from_disk(id: &str) -> Vec<RecommendedMoment> {
    match std::fs::read(diagnostics_path(id)) {
        Ok(bytes) => {
            let doc: DiagnosticsDoc = serde_json::from_slice(&bytes).unwrap_or_default();
            recommended_from_events(&doc.events)
        }
        Err(_) => Vec::new(),
    }
}

/// Map notable diagnostics (exceptions, network >=400, console errors) to
/// jump-to moments. `t` may be negative (before t0); clamp to 0.
fn recommended_from_events(events: &[DiagnosticEvent]) -> Vec<RecommendedMoment> {
    let mut out = Vec::new();
    for e in events {
        let notable = match e.kind.as_str() {
            "exception" => true,
            "network" => e.status.map(|s| s >= 400).unwrap_or(false),
            "console" => e.level.as_deref() == Some("error"),
            _ => false,
        };
        if !notable {
            continue;
        }
        let at_ms = e.t.max(0) as u64;
        let reason = match e.kind.as_str() {
            "network" => {
                let status = e.status.map(|s| s.to_string()).unwrap_or_default();
                let method = e.method.clone().unwrap_or_default();
                let url = e.url.clone().unwrap_or_default();
                format!("network {status} {method} {url}").trim().to_string()
            }
            other => {
                let text = e.text.clone().unwrap_or_default();
                format!("{other}: {text}").trim().to_string()
            }
        };
        out.push(RecommendedMoment { at_ms, reason });
    }
    out
}

/// Find the capture-time `seq-*.jpg` nearest to `at_ms`, mapping the requested
/// time onto the fps-spaced sequence.
fn nearest_seq_frame(id: &str, at_ms: u64) -> Option<PathBuf> {
    let dir = frames_dir(id);
    let mut frames: Vec<(u64, PathBuf)> = std::fs::read_dir(&dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let stem = path.file_stem()?.to_str()?;
            let seq = stem.strip_prefix("seq-")?.parse::<u64>().ok()?;
            Some((seq, path))
        })
        .collect();
    if frames.is_empty() {
        return None;
    }
    frames.sort_by_key(|(seq, _)| *seq);

    // Frames are 1-indexed at the clip fps; the desired index is at_ms/interval.
    let fps = clip_fps().max(1) as u64;
    let target_seq = (at_ms * fps / 1000).saturating_add(1);
    let best = frames
        .iter()
        .min_by_key(|(seq, _)| seq.abs_diff(target_seq))
        .map(|(_, p)| p.clone());
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recommended_filters_and_clamps() {
        let events = vec![
            DiagnosticEvent {
                t: -100,
                kind: "console".into(),
                level: Some("error".into()),
                text: Some("boom".into()),
                url: None,
                method: None,
                status: None,
                source: Some("chrome-extension".into()),
            },
            DiagnosticEvent {
                t: 4200,
                kind: "network".into(),
                level: Some("error".into()),
                text: None,
                url: Some("https://h/api".into()),
                method: Some("POST".into()),
                status: Some(500),
                source: None,
            },
            DiagnosticEvent {
                t: 900,
                kind: "console".into(),
                level: Some("warning".into()),
                text: Some("noise".into()),
                url: None,
                method: None,
                status: None,
                source: None,
            },
            DiagnosticEvent {
                t: 1000,
                kind: "network".into(),
                level: None,
                text: None,
                url: Some("https://h/ok".into()),
                method: Some("GET".into()),
                status: Some(200),
                source: None,
            },
        ];
        let moments = recommended_from_events(&events);
        // Only the console error (clamped to 0) and the 500 survive.
        assert_eq!(moments.len(), 2);
        assert_eq!(moments[0].at_ms, 0);
        assert_eq!(moments[1].at_ms, 4200);
    }

    #[test]
    fn context_round_trips_camel_case() {
        let ctx = ClipContext {
            id: "clip_abc".into(),
            title: "Test".into(),
            duration_ms: 1200,
            created_at: "2026-07-09T10:22:11.014Z".into(),
            t0_epoch_ms: 1_752_055_331_014,
            capture: ClipCapture {
                screen: true,
                mic: true,
                system_audio: false,
                tab: Some(ClipTab {
                    url: "https://app/checkout".into(),
                    title: Some("Checkout".into()),
                }),
            },
            video: "clip.mp4".into(),
            transcript_path: "agent-transcript.json".into(),
            diagnostics_path: "diagnostics.json".into(),
            frames_endpoint: "/clips/clip_abc/frame".into(),
            recommended_moments: vec![RecommendedMoment {
                at_ms: 4200,
                reason: "network 500".into(),
            }],
            scan_warning: None,
        };
        let json = serde_json::to_string(&ctx).unwrap();
        assert!(json.contains("\"durationMs\":1200"));
        assert!(json.contains("\"t0EpochMs\":1752055331014"));
        assert!(json.contains("\"systemAudio\":false"));
        assert!(json.contains("\"framesEndpoint\":\"/clips/clip_abc/frame\""));
        let back: ClipContext = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "clip_abc");
        assert_eq!(back.recommended_moments[0].at_ms, 4200);
    }
}
