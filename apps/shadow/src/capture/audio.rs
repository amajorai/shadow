use anyhow::Result;
use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Stream};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc, Arc, Mutex,
};

/// Audio capture trait.
#[async_trait]
pub trait AudioCapture: Send + Sync {
    async fn start(&mut self) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    fn is_mic_active(&self) -> bool;
}

/// Length of one microphone segment. Samples are held in memory until the
/// segment closes and then written once, sealed: a crash loses at most this
/// much audio, and no PCM ever reaches the disk in plaintext.
const SEGMENT_SECS: u64 = 120;

/// Timeline source name of microphone segments.
const MIC_SOURCE: &str = "mic";

/// Cross-platform audio capture using cpal.
///
/// Mic audio is buffered in memory and written as sealed (`RYU-SF01`) WAV
/// segments of [`SEGMENT_SECS`] under `media/audio/`, each registered in the
/// timeline. Capture refuses to start unless Shadow can seal.
pub struct PlatformAudioCapture {
    host: Host,
    input_device: Option<Device>,
    stream: Option<Stream>,
    is_active: Arc<AtomicBool>,
    last_active_ts: Arc<AtomicU64>,
    audio_dir: PathBuf,
    recorder: Option<Recorder>,
}

// Stream holds a non-Send callback; mark manually since we only access from single task
unsafe impl Send for PlatformAudioCapture {}
unsafe impl Sync for PlatformAudioCapture {}

impl PlatformAudioCapture {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let input_device = host.default_input_device();

        let audio_dir = crate::paths::data_dir().join("media").join("audio");

        Ok(Self {
            host,
            input_device,
            stream: None,
            is_active: Arc::new(AtomicBool::new(false)),
            last_active_ts: Arc::new(AtomicU64::new(0)),
            audio_dir,
            recorder: None,
        })
    }

    pub fn with_audio_dir(mut self, dir: PathBuf) -> Self {
        self.audio_dir = dir;
        self
    }

    /// Point the recorder at `<data dir>/media/audio` (the engine's data dir).
    pub fn set_audio_dir(&mut self, dir: PathBuf) {
        self.audio_dir = dir;
    }

    /// Start capture once `writable` (Shadow's sealing state) allows it.
    async fn start_gated(&mut self, writable: Result<(), shadow_core::ShadowError>) -> Result<()> {
        writable.map_err(|error| {
            anyhow::anyhow!("microphone capture is off until Shadow can seal ({error})")
        })?;

        let device = self
            .input_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No audio input device available"))?;

        let config = device
            .default_input_config()
            .map_err(|e| anyhow::anyhow!("No default input config: {}", e))?;

        tracing::info!("Audio capture: {:?} {:?}", device.name(), config);

        let spec = hound::WavSpec {
            channels: config.channels(),
            sample_rate: config.sample_rate().0,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let is_active = Arc::clone(&self.is_active);
        let last_active_ts = Arc::clone(&self.last_active_ts);

        let recorder = Recorder::spawn(self.audio_dir.clone(), spec)?;
        let mut sink = recorder.sink();

        let is_active_cb = Arc::clone(&is_active);
        let last_ts_cb = Arc::clone(&last_active_ts);

        let stream = device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Compute RMS energy level
                    let rms: f32 =
                        (data.iter().map(|&s| s * s).sum::<f32>() / data.len() as f32).sqrt();
                    let active = rms > 0.01; // -40 dBFS threshold

                    if active {
                        is_active_cb.store(true, Ordering::Relaxed);
                        last_ts_cb.store(now_us(), Ordering::Relaxed);
                    } else {
                        // 30s post-quiet timeout
                        let last = last_ts_cb.load(Ordering::Relaxed);
                        if now_us().saturating_sub(last) > 30 * 1_000_000 {
                            is_active_cb.store(false, Ordering::Relaxed);
                        }
                    }

                    sink.push(data);
                },
                |e| tracing::warn!("Audio stream error: {}", e),
                None,
            )
            .map_err(|e| anyhow::anyhow!("Failed to build audio stream: {}", e))?;

        stream
            .play()
            .map_err(|e| anyhow::anyhow!("Failed to start audio stream: {}", e))?;

        self.stream = Some(stream);
        self.recorder = Some(recorder);
        tracing::info!("Audio capture started");
        Ok(())
    }
}

#[async_trait]
impl AudioCapture for PlatformAudioCapture {
    async fn start(&mut self) -> Result<()> {
        self.start_gated(shadow_core::seal::require_writable())
            .await
    }

    async fn stop(&mut self) -> Result<()> {
        // Dropping the stream stops the callback; then the open segment is
        // written (sealed) before stop returns.
        self.stream = None;
        if let Some(recorder) = self.recorder.take() {
            tokio::task::spawn_blocking(move || recorder.finish())
                .await
                .map_err(|e| anyhow::anyhow!("audio writer panicked: {e}"))?;
        }
        self.is_active.store(false, Ordering::Relaxed);
        tracing::info!("Audio capture stopped");
        Ok(())
    }

    fn is_mic_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }
}

impl Default for PlatformAudioCapture {
    fn default() -> Self {
        Self::new().expect("Failed to create PlatformAudioCapture")
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

// ─── Segment recorder ──────────────────────────────────────────────────────────

/// One closed segment of interleaved i16 PCM.
struct PcmSegment {
    start_ts: u64,
    end_ts: u64,
    samples: Vec<i16>,
}

#[derive(Default)]
struct PcmBuffer {
    start_ts: Option<u64>,
    samples: Vec<i16>,
}

impl PcmBuffer {
    fn take(&mut self, end_ts: u64) -> Option<PcmSegment> {
        let start_ts = self.start_ts.take()?;
        let samples = std::mem::take(&mut self.samples);
        (!samples.is_empty()).then_some(PcmSegment {
            start_ts,
            end_ts,
            samples,
        })
    }
}

/// The audio callback's end: buffers samples and hands each full segment to
/// the writer thread (no disk or crypto work on the audio thread).
struct SegmentSink {
    buffer: Arc<Mutex<PcmBuffer>>,
    tx: mpsc::Sender<PcmSegment>,
    segment_samples: usize,
}

impl SegmentSink {
    fn push(&mut self, data: &[f32]) {
        self.push_at(data, now_us());
    }

    fn push_at(&mut self, data: &[f32], now: u64) {
        let full = {
            let Ok(mut buffer) = self.buffer.lock() else {
                return;
            };
            buffer.start_ts.get_or_insert(now);
            buffer.samples.extend(
                data.iter()
                    .map(|&sample| (sample * 32767.0).clamp(-32768.0, 32767.0) as i16),
            );
            if buffer.samples.len() >= self.segment_samples {
                buffer.take(now)
            } else {
                None
            }
        };
        if let Some(segment) = full {
            let _ = self.tx.send(segment);
        }
    }
}

/// Owns the writer thread that seals and registers closed segments.
struct Recorder {
    buffer: Arc<Mutex<PcmBuffer>>,
    tx: Option<mpsc::Sender<PcmSegment>>,
    worker: Option<std::thread::JoinHandle<()>>,
    segment_samples: usize,
}

impl Recorder {
    fn spawn(audio_dir: PathBuf, spec: hound::WavSpec) -> Result<Self> {
        let samples_per_sec = spec.sample_rate as usize * spec.channels.max(1) as usize;
        Self::spawn_with(audio_dir, spec, samples_per_sec * SEGMENT_SECS as usize)
    }

    fn spawn_with(
        audio_dir: PathBuf,
        spec: hound::WavSpec,
        segment_samples: usize,
    ) -> Result<Self> {
        shadow_core::seal::create_private_dir(&audio_dir)?;
        let (tx, rx) = mpsc::channel::<PcmSegment>();
        let worker = std::thread::Builder::new()
            .name("shadow-audio-writer".into())
            .spawn(move || {
                for segment in rx {
                    if let Err(error) = write_segment(&audio_dir, spec, &segment) {
                        tracing::warn!("shadow: dropped a microphone segment: {error}");
                    }
                }
            })?;
        Ok(Self {
            buffer: Arc::new(Mutex::new(PcmBuffer::default())),
            tx: Some(tx),
            worker: Some(worker),
            segment_samples: segment_samples.max(1),
        })
    }

    fn sink(&self) -> SegmentSink {
        SegmentSink {
            buffer: Arc::clone(&self.buffer),
            tx: self.tx.clone().expect("recorder is open"),
            segment_samples: self.segment_samples,
        }
    }

    /// Write the open segment and wait for every queued one to be sealed.
    fn finish(mut self) {
        let last = self
            .buffer
            .lock()
            .ok()
            .and_then(|mut buffer| buffer.take(now_us()));
        if let Some(tx) = self.tx.take() {
            if let Some(segment) = last {
                let _ = tx.send(segment);
            }
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Encode a segment as WAV in memory.
fn encode_wav(spec: hound::WavSpec, samples: &[i16]) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        let mut i16_writer = writer.get_i16_writer(samples.len() as u32);
        for &sample in samples {
            i16_writer.write_sample(sample);
        }
        i16_writer.flush()?;
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// Seal one segment to `<audio_dir>/<UTC start>.wav` and register it.
fn write_segment(audio_dir: &Path, spec: hound::WavSpec, segment: &PcmSegment) -> Result<PathBuf> {
    let media_root = audio_dir
        .parent()
        .ok_or_else(|| anyhow::anyhow!("audio dir has no media parent"))?;
    let secs = (segment.start_ts / 1_000_000) as i64;
    let micros = segment.start_ts % 1_000_000;
    let started = chrono::DateTime::from_timestamp(secs, 0).unwrap_or_else(chrono::Utc::now);
    let path = audio_dir.join(format!(
        "{}-{micros:06}.wav",
        started.format("%Y-%m-%dT%H-%M-%S")
    ));
    let wav = encode_wav(spec, &segment.samples)?;
    shadow_core::media::write_sealed_media_in(media_root, &path, &wav)?;
    let file = path.to_string_lossy().to_string();
    let registered = shadow_core::insert_audio_segment(
        MIC_SOURCE.to_string(),
        segment.start_ts,
        file.clone(),
        None,
        Some(spec.sample_rate),
        Some(u32::from(spec.channels)),
    )
    .and_then(|_| shadow_core::finalize_audio_segment(file, segment.end_ts));
    if let Err(error) = registered {
        tracing::debug!("shadow: microphone segment not indexed: {error}");
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> hound::WavSpec {
        hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        }
    }

    struct TempRoot(PathBuf);

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_root() -> TempRoot {
        TempRoot(std::env::temp_dir().join(format!("shadow-audio-{}", uuid::Uuid::new_v4())))
    }

    fn audio_dir() -> (TempRoot, PathBuf) {
        shadow_core::seal::install_fixed_key_for_tests();
        let tmp = temp_root();
        let dir = tmp.0.join("media").join("audio");
        (tmp, dir)
    }

    fn wav_files(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|entry| entry.path())
            .collect();
        files.sort();
        files
    }

    #[test]
    fn segments_are_sealed_on_rotation_and_finish_with_no_plaintext_wav() {
        let (_tmp, dir) = audio_dir();
        let recorder = Recorder::spawn_with(dir.clone(), spec(), 4).unwrap();
        let mut sink = recorder.sink();
        // Five samples close one segment (rotation); one more stays open.
        sink.push_at(&[0.5, -0.5, 0.25, -0.25, 0.1], 1_000_000);
        sink.push_at(&[0.75], 2_000_000);
        drop(sink);
        recorder.finish();

        let files = wav_files(&dir);
        assert_eq!(
            files.len(),
            2,
            "one rotated segment and the one stop closed"
        );
        let media_root = dir.parent().unwrap();
        for file in &files {
            let on_disk = std::fs::read(file).unwrap();
            assert!(
                on_disk.starts_with(ryu_crypto::content::SEALED_FILE_MAGIC),
                "{} is not sealed",
                file.display()
            );
            assert!(!on_disk.windows(4).any(|w| w == b"RIFF" || w == b"WAVE"));
            let plain = shadow_core::media::read_media_file_in(media_root, file).unwrap();
            let reader = hound::WavReader::new(std::io::Cursor::new(plain)).unwrap();
            assert_eq!(reader.spec().sample_rate, 8_000);
            assert!(reader.len() > 0);
        }
        let first = shadow_core::media::read_media_file_in(media_root, &files[0]).unwrap();
        let samples: Vec<i16> = hound::WavReader::new(std::io::Cursor::new(first))
            .unwrap()
            .samples::<i16>()
            .map(Result::unwrap)
            .collect();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0], (0.5f32 * 32767.0) as i16);
        // No temp file is left behind.
        assert!(files
            .iter()
            .all(|file| file.extension().is_some_and(|ext| ext == "wav")));
    }

    #[tokio::test]
    async fn start_refuses_without_a_sealing_key() {
        let tmp = temp_root();
        let dir = tmp.0.join("media").join("audio");
        let mut capture = PlatformAudioCapture::new()
            .unwrap()
            .with_audio_dir(dir.clone());
        let error = capture
            .start_gated(Err(shadow_core::ShadowError::StorageError {
                msg: "sealing: no-key".into(),
            }))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("until Shadow can seal"),
            "{error}"
        );
        assert!(capture.recorder.is_none());
        assert!(!dir.exists(), "nothing is written without a key");
    }
}
