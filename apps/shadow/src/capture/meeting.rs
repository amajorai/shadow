//! Meeting recorder — device-local audio capture for the meeting-notes feature.
//!
//! Granola-style meeting notes need **both** sides of a call: your microphone
//! AND the system audio (the other participants, played through your speakers).
//! cpal's WASAPI backend captures system audio transparently when you build an
//! *input* stream on an *output* (render) device — that is loopback capture. So
//! this recorder runs two cpal input streams:
//!   - the default **input** device (microphone), and
//!   - the default **output** device (system loopback — Windows only; elsewhere
//!     it degrades to mic-only).
//!
//! Each side is downmixed to mono and accumulated; a background task resamples
//! both to 16 kHz (anti-aliased, see [`mixer`]) and feeds them to a
//! [`segmenter::VadSegmenter`] that cuts chunks on **silence** (not a fixed
//! timer, so words aren't split) with sample-accurate offsets. Each chunk is
//! encoded as an **interleaved stereo** WAV (L = mic, R = system) and POSTed to
//! Core's `POST /api/meetings/:id/chunk?offset_ms=…`. Keeping the two sides in
//! separate channels lets Core persist the split diarization needs (mic is always
//! "you"); Core downmixes to mono for whisper. This is pure device-local
//! plumbing — the "sensor" half of the Core-vs-sensor split.
//!
//! Lifecycle is a process-global (mirroring `server.rs`'s capture-control flags)
//! so the `/meeting/start` + `/meeting/stop` HTTP handlers reach it without
//! threading new fields through `AppState`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;

use crate::capture::{mixer, segmenter};

/// How often the uploader drains captured audio and feeds the segmenter. The
/// segmenter decides the actual *chunk* boundaries (on silence); this is just the
/// polling cadence, kept short so cuts are responsive.
const SEGMENT_POLL_SECS: u64 = 1;

/// The 16 kHz target whisper/parakeet expect.
const TARGET_RATE: u32 = 16_000;

/// One source's rolling mono sample buffer plus the rate it was captured at.
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

/// An active meeting recording: the live cpal streams (kept alive so capture
/// continues) and the running flag the uploader task watches.
pub struct MeetingRecorder {
    meeting_id: String,
    _mic_stream: Option<Stream>,
    _loopback_stream: Option<Stream>,
    running: Arc<AtomicBool>,
}

// cpal `Stream` holds a non-Send callback. We only ever touch the streams from
// the handler thread (build on start, drop on stop) and never send them across
// threads, so this is sound — same justification as `PlatformAudioCapture`.
unsafe impl Send for MeetingRecorder {}
unsafe impl Sync for MeetingRecorder {}

static RECORDER: OnceLock<Mutex<Option<MeetingRecorder>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<MeetingRecorder>> {
    RECORDER.get_or_init(|| Mutex::new(None))
}

/// Whether a meeting is currently being recorded.
pub fn is_recording() -> bool {
    slot().lock().map(|g| g.is_some()).unwrap_or(false)
}

/// The id of the meeting currently recording, if any.
pub fn current_meeting_id() -> Option<String> {
    slot()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|r| r.meeting_id.clone()))
}

/// Start recording `meeting_id`, uploading mixed 16 kHz WAV chunks to `ingest_url`
/// (Core's `/api/meetings/:id/chunk`). Replaces any in-progress recording.
///
/// Mic capture is required; loopback (system audio) is best-effort — on platforms
/// where building an input stream on the output device isn't loopback, the meeting
/// records mic-only with a logged warning.
pub fn start(meeting_id: String, ingest_url: String) -> anyhow::Result<()> {
    stop(); // tear down any prior recording first

    let host = cpal::default_host();
    let running = Arc::new(AtomicBool::new(true));

    // --- microphone (required) ---------------------------------------------
    let mic_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no microphone input device available"))?;
    let mic_cfg = mic_device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("no default input config: {e}"))?;
    let mic_buf = SourceBuffer::new(mic_cfg.sample_rate().0);
    let mic_stream = build_capture_stream(&mic_device, &mic_cfg, Arc::clone(&mic_buf))
        .map_err(|e| anyhow::anyhow!("failed to start microphone capture: {e}"))?;
    mic_stream
        .play()
        .map_err(|e| anyhow::anyhow!("failed to play microphone stream: {e}"))?;

    // --- system loopback (best-effort) -------------------------------------
    let (loopback_stream, loopback_buf) = match build_loopback(&host) {
        Ok((stream, buf)) => {
            tracing::info!("meeting recorder: system-audio loopback active");
            (Some(stream), Some(buf))
        }
        Err(e) => {
            tracing::warn!(
                "meeting recorder: loopback capture unavailable ({e}); recording microphone only"
            );
            (None, None)
        }
    };

    // --- chunk uploader -----------------------------------------------------
    let task_running = Arc::clone(&running);
    let client = reqwest::Client::new();
    let mid = meeting_id.clone();
    tokio::spawn(async move {
        let mut segmenter = segmenter::VadSegmenter::default();
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(SEGMENT_POLL_SECS));
        loop {
            ticker.tick().await;
            let running_now = task_running.load(Ordering::Relaxed);

            // Drain whatever audio arrived this tick, resample each side to 16 kHz
            // (anti-aliased), and hand it to the segmenter.
            let (mic_samples, mic_rate) = mic_buf.drain();
            let (loop_samples, loop_rate) = match &loopback_buf {
                Some(b) => b.drain(),
                None => (Vec::new(), TARGET_RATE),
            };
            if !mic_samples.is_empty() || !loop_samples.is_empty() {
                let mic16 = mixer::resample_to_16k(&mic_samples, mic_rate);
                let sys16 = mixer::resample_to_16k(&loop_samples, loop_rate);
                segmenter.push(&mic16, &sys16);
            }

            // Emit every ready chunk; force-flush the tail when the meeting stops.
            while let Some(chunk) = segmenter.try_cut(!running_now) {
                match encode_wav_16k_stereo(&chunk.stereo) {
                    Ok(wav) => upload_chunk(&client, &ingest_url, wav, &mid, chunk.offset_ms).await,
                    Err(e) => tracing::warn!("meeting recorder: WAV encode failed: {e}"),
                }
            }

            if !running_now {
                break;
            }
        }
        tracing::info!("meeting recorder: uploader task for {mid} stopped");
    });

    let recorder = MeetingRecorder {
        meeting_id,
        _mic_stream: Some(mic_stream),
        _loopback_stream: loopback_stream,
        running,
    };
    if let Ok(mut guard) = slot().lock() {
        *guard = Some(recorder);
    }
    Ok(())
}

/// Stop the current recording (if any), tearing down streams + the uploader task.
pub fn stop() {
    if let Ok(mut guard) = slot().lock() {
        if let Some(rec) = guard.take() {
            rec.running.store(false, Ordering::Relaxed);
            tracing::info!("meeting recorder: stopped {}", rec.meeting_id);
            // Streams drop here, halting capture.
        }
    }
}

/// Build a system-audio (loopback) capture. The mechanism differs per OS, but the
/// result is the same: an already-playing input stream carrying "the other side"
/// of the call plus its buffer.
///
/// - **Windows**: build an *input* stream on the default *output* (render) device
///   — that is WASAPI loopback, no extra setup.
/// - **Linux** (PulseAudio/PipeWire): capture the output's **monitor** source,
///   which those servers expose as a normal input device named `…Monitor…`.
/// - **macOS**: CoreAudio has no built-in loopback; capture a **virtual output
///   device** (BlackHole / Loopback / an Aggregate) if the user has one. Absent
///   that, this errors and the meeting records mic-only. (A native ScreenCaptureKit
///   path is the eventual upgrade.)
fn build_loopback(host: &cpal::Host) -> anyhow::Result<(Stream, Arc<SourceBuffer>)> {
    #[cfg(windows)]
    {
        let out_device = host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("no default output device"))?;
        // For a render device, cpal derives the loopback format from the output
        // config; building an *input* stream on it enables loopback.
        let out_cfg = out_device
            .default_output_config()
            .map_err(|e| anyhow::anyhow!("no default output config: {e}"))?;
        let buf = SourceBuffer::new(out_cfg.sample_rate().0);
        let stream = build_capture_stream(&out_device, &out_cfg, Arc::clone(&buf))?;
        stream.play()?;
        Ok((stream, buf))
    }

    #[cfg(not(windows))]
    {
        let device = find_system_audio_device(host).ok_or_else(|| {
            anyhow::anyhow!(
                "no system-audio capture device found. On Linux, enable a PulseAudio/\
PipeWire monitor source; on macOS, install a virtual device (e.g. BlackHole) and \
route output through it, or an Aggregate device"
            )
        })?;
        let cfg = device
            .default_input_config()
            .map_err(|e| anyhow::anyhow!("no default input config for system-audio device: {e}"))?;
        let buf = SourceBuffer::new(cfg.sample_rate().0);
        let stream = build_capture_stream(&device, &cfg, Arc::clone(&buf))?;
        stream.play()?;
        Ok((stream, buf))
    }
}

/// Find a system-audio capture device among the host's **input** devices by name:
/// PulseAudio/PipeWire monitor sources (`…Monitor`) on Linux, or a virtual/
/// aggregate output device on macOS. cpal enumerates these cross-platform even
/// though the loopback mechanism itself is OS-specific.
#[cfg(not(windows))]
fn find_system_audio_device(host: &cpal::Host) -> Option<cpal::Device> {
    const HINTS: &[&str] = &[
        "monitor",     // PulseAudio / PipeWire monitor source (Linux)
        "blackhole",   // BlackHole virtual device (macOS)
        "loopback",    // Rogue Amoeba Loopback (macOS)
        "aggregate",   // Core Audio aggregate device (macOS)
        "soundflower", // legacy virtual device (macOS)
        "pipewire",    // PipeWire default sink monitor
    ];
    let devices = host.input_devices().ok()?;
    for device in devices {
        if let Ok(name) = device.name() {
            let lower = name.to_lowercase();
            if HINTS.iter().any(|h| lower.contains(h)) {
                tracing::info!("meeting recorder: using system-audio device '{name}'");
                return Some(device);
            }
        }
    }
    None
}

/// Build an input (capture) stream on `device` that downmixes every frame to mono
/// and appends it to `buf`. Works for both a capture device (mic) and a render
/// device (loopback).
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
                    // Downmix interleaved frames to mono (average the channels).
                    for frame in data.chunks(channels) {
                        let sum: f32 = frame.iter().copied().sum();
                        samples.push(sum / channels as f32);
                    }
                }
            },
            |e| tracing::warn!("meeting recorder: stream error: {e}"),
            None,
        )
        .map_err(|e| anyhow::anyhow!("build_input_stream: {e}"))?;
    Ok(stream)
}

/// Encode interleaved stereo f32 samples (L = mic, R = system) as a 16 kHz /
/// 16-bit PCM WAV in memory.
fn encode_wav_16k_stereo(interleaved: &[f32]) -> anyhow::Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::<u8>::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in interleaved {
            let pcm = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(pcm)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// POST one stereo WAV chunk to Core's meeting ingest endpoint (best-effort),
/// tagging it with its sample-accurate `offset_ms` so Core times the transcript
/// segment from audio position rather than wall-clock.
async fn upload_chunk(
    client: &reqwest::Client,
    ingest_url: &str,
    wav: Vec<u8>,
    meeting_id: &str,
    offset_ms: i64,
) {
    let part = match reqwest::multipart::Part::bytes(wav)
        .file_name("chunk.wav")
        .mime_str("audio/wav")
    {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("meeting recorder: building upload part failed: {e}");
            return;
        }
    };
    let form = reqwest::multipart::Form::new().part("file", part);
    let sep = if ingest_url.contains('?') { '&' } else { '?' };
    let url = format!("{ingest_url}{sep}offset_ms={offset_ms}");
    match client.post(&url).multipart(form).send().await {
        Ok(resp) if resp.status().is_success() => {}
        Ok(resp) => tracing::warn!(
            "meeting recorder: chunk upload for {meeting_id} returned {}",
            resp.status()
        ),
        Err(e) => tracing::warn!("meeting recorder: chunk upload for {meeting_id} failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_stereo_wav_has_riff_header_and_two_channels() {
        let wav = encode_wav_16k_stereo(&[0.0, 0.1, -0.1, 0.2]).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        // Channel count is a u16 at byte offset 22 in the fmt chunk.
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
    }
}
