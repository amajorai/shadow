use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::capture::screen::Frame;

/// A segment is closed (and sealed) once it is this old, whether or not frames
/// keep arriving: rotation runs on the wall clock, not on frame arrival.
pub const MAX_SEGMENT_AGE_US: u64 = 3600 * 1_000_000;
/// A segment that has encoded no frame for this long is closed and sealed, so
/// an unchanged screen (user away, screen locked) does not keep it plaintext.
pub const SEGMENT_IDLE_CLOSE_US: u64 = 5 * 60 * 1_000_000;

/// Whether an open segment must be closed now. `recording` is false while
/// capture is paused, frame capture is off or the foreground app is not
/// allowed: the open segment is then closed at once, so a privacy pause never
/// leaves the last recording unencrypted for the length of the pause.
pub fn segment_close_due(
    segment_start_ts: Option<u64>,
    last_frame_ts: Option<u64>,
    now: u64,
    recording: bool,
) -> bool {
    let Some(start) = segment_start_ts else {
        return false;
    };
    if !recording {
        return true;
    }
    let last = last_frame_ts.unwrap_or(start).max(start);
    now.saturating_sub(start) >= MAX_SEGMENT_AGE_US
        || now.saturating_sub(last) >= SEGMENT_IDLE_CLOSE_US
}

/// H.265 fragmented MP4 video encoder.
/// Requires ffmpeg-next feature and FFmpeg system libraries.
pub struct VideoEncoder {
    output_dir: PathBuf,
    display_id: u32,
    target_fps: f64,
    bitrate_kbps: u32,

    // Current segment state
    current_segment_path: Option<PathBuf>,
    segment_start_ts: Option<u64>,
    segment_frame_count: u32,
    frames_since_key: u32,
    last_frame_ts: Option<u64>,

    #[cfg(feature = "video")]
    encoder_ctx: Option<ffmpeg_next::codec::encoder::video::Video>,
    #[cfg(feature = "video")]
    output_ctx: Option<ffmpeg_next::format::context::Output>,
}

/// Record a closed segment in the timeline, which seals it; when the timeline
/// update fails, seal it directly.
fn record_closed_segment(path: &Path, end_ts: u64) -> Result<()> {
    let recorded = shadow_core::finalize_video_segment(path.to_string_lossy().to_string(), end_ts);
    if recorded.is_err() {
        shadow_core::seal_closed_video_segment(path);
    }
    recorded.map_err(Into::into)
}

impl VideoEncoder {
    /// Create a new encoder for a display.
    pub fn new(output_dir: &Path, display_id: u32) -> Result<Self> {
        #[cfg(feature = "video")]
        ffmpeg_next::init()?;

        std::fs::create_dir_all(output_dir)?;

        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            display_id,
            target_fps: 0.5,
            bitrate_kbps: 300,
            current_segment_path: None,
            segment_start_ts: None,
            segment_frame_count: 0,
            frames_since_key: 0,
            last_frame_ts: None,
            #[cfg(feature = "video")]
            encoder_ctx: None,
            #[cfg(feature = "video")]
            output_ctx: None,
        })
    }

    /// Ingest a frame. Starts a new segment if needed.
    pub fn encode_frame(&mut self, frame: &Frame) -> Result<()> {
        // Start new segment if needed (hourly rotation)
        let should_rotate = match self.segment_start_ts {
            None => true,
            Some(start) => frame.timestamp.saturating_sub(start) >= MAX_SEGMENT_AGE_US,
        };

        if should_rotate {
            self.rotate_segment(frame.timestamp, frame.width, frame.height)?;
        }

        self.segment_frame_count += 1;
        self.frames_since_key += 1;
        self.last_frame_ts = Some(frame.timestamp);

        #[cfg(feature = "video")]
        {
            self.encode_frame_ffmpeg(frame)?;
        }

        #[cfg(not(feature = "video"))]
        {
            // Without FFmpeg, just log
            tracing::trace!(
                "Video frame {}: display={}, ts={}",
                self.segment_frame_count,
                frame.display_id,
                frame.timestamp
            );
        }

        Ok(())
    }

    /// Close and seal the open segment when it is due by wall-clock time (see
    /// [`segment_close_due`]). Called on every capture tick, including ticks
    /// that encode nothing. Returns whether a segment was closed.
    pub fn close_segment_if_due(&mut self, now: u64, recording: bool) -> Result<bool> {
        self.close_segment_if_due_with(now, recording, record_closed_segment)
    }

    fn close_segment_if_due_with(
        &mut self,
        now: u64,
        recording: bool,
        record: impl FnOnce(&Path, u64) -> Result<()>,
    ) -> Result<bool> {
        if !segment_close_due(self.segment_start_ts, self.last_frame_ts, now, recording) {
            return Ok(false);
        }
        self.close_segment_with(now, record)?;
        Ok(true)
    }

    /// Close the open segment (if any) and seal it: the encoder writes the file
    /// in plaintext only while it is open. The next frame opens a new segment.
    pub fn close_segment(&mut self, end_ts: u64) -> Result<()> {
        self.close_segment_with(end_ts, record_closed_segment)
    }

    fn close_segment_with(
        &mut self,
        end_ts: u64,
        record: impl FnOnce(&Path, u64) -> Result<()>,
    ) -> Result<()> {
        self.segment_start_ts = None;
        self.last_frame_ts = None;
        let Some(path) = self.current_segment_path.take() else {
            return Ok(());
        };
        // Seal whatever happens to the trailer: a failed finalize must not leave
        // the plaintext segment behind with its path forgotten.
        #[cfg(feature = "video")]
        let finalized = self.finalize_segment();
        #[cfg(not(feature = "video"))]
        let finalized: Result<()> = Ok(());
        tracing::info!("Video segment closed: {:?}", path);
        let recorded = record(&path, end_ts);
        finalized?;
        recorded?;
        Ok(())
    }

    fn rotate_segment(&mut self, ts: u64, width: u32, height: u32) -> Result<()> {
        self.rotate_segment_gated(ts, width, height, shadow_core::seal::require_writable())
    }

    fn rotate_segment_gated(
        &mut self,
        ts: u64,
        width: u32,
        height: u32,
        writable: Result<(), shadow_core::ShadowError>,
    ) -> Result<()> {
        // Finalize previous segment
        self.close_segment(ts)?;

        // Never open a plaintext encoder output Shadow could not seal on close.
        writable.map_err(|error| {
            anyhow::anyhow!("screen video is off until Shadow can seal ({error})")
        })?;

        // Start new segment
        let ts_secs = ts / 1_000_000;
        let dt =
            chrono::DateTime::from_timestamp(ts_secs as i64, 0).unwrap_or_else(chrono::Utc::now);
        let filename = format!(
            "display-{}/{}.mp4",
            self.display_id,
            // Seconds included: segments now also close on idle and on pause,
            // so one hour can hold several segments that must not overwrite
            // each other.
            dt.format("%Y-%m-%dT%H-%M-%S")
        );
        let path = self.output_dir.join(&filename);
        std::fs::create_dir_all(path.parent().unwrap())?;

        self.current_segment_path = Some(path.clone());
        self.segment_start_ts = Some(ts);
        self.segment_frame_count = 0;
        self.frames_since_key = 0;
        self.last_frame_ts = Some(ts);

        // Registered before the encoder creates the file, so the media sweep
        // never seals the segment while it is being written.
        shadow_core::media::mark_media_open(&path);
        #[cfg(feature = "video")]
        if let Err(error) = self.open_segment_ffmpeg(&path, width, height) {
            self.current_segment_path = None;
            self.segment_start_ts = None;
            // Whatever ffmpeg created is sealed by the next sweep.
            shadow_core::media::mark_media_closed(&path);
            return Err(error);
        }

        shadow_core::insert_video_segment(self.display_id, ts, path.to_string_lossy().to_string())?;

        tracing::info!("Video segment started: {:?}", path);
        Ok(())
    }

    #[cfg(feature = "video")]
    fn open_segment_ffmpeg(&mut self, path: &Path, width: u32, height: u32) -> Result<()> {
        use ffmpeg_next::codec::encoder;
        use ffmpeg_next::codec::Id;
        use ffmpeg_next::format::output;
        use ffmpeg_next::util::format::pixel::Pixel;
        use ffmpeg_next::Rational;

        let mut output_ctx = output(path)?;

        let codec = ffmpeg_next::encoder::find(Id::HEVC)
            .ok_or_else(|| anyhow::anyhow!("H.265 encoder not found"))?;

        let mut stream = output_ctx.add_stream(codec)?;
        let mut encoder = codec.video()?;

        encoder.set_width(width);
        encoder.set_height(height);
        encoder.set_format(Pixel::YUV420P);
        encoder.set_frame_rate(Some(Rational::new(1, 2))); // 0.5 fps
        encoder.set_time_base(Rational::new(1, 1_000_000)); // microseconds
        encoder.set_bit_rate(self.bitrate_kbps as usize * 1000);

        // Fragmented MP4 for streaming-friendly output
        let opts = ffmpeg_next::Dictionary::from_str("movflags=+frag_keyframe+empty_moov")?;
        let encoder = encoder.open_as_with(codec, opts)?;

        stream.set_parameters(&encoder);

        self.encoder_ctx = Some(encoder);
        self.output_ctx = Some(output_ctx);
        Ok(())
    }

    #[cfg(feature = "video")]
    fn encode_frame_ffmpeg(&mut self, frame: &Frame) -> Result<()> {
        use ffmpeg_next::frame::Video;
        use ffmpeg_next::software::scaling::{context::Context as SwsContext, flag::Flags};
        use ffmpeg_next::util::format::pixel::Pixel;

        let encoder = self
            .encoder_ctx
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Encoder not initialized"))?;

        // Create source frame (BGRA)
        let mut src_frame = Video::new(Pixel::BGRA, frame.width, frame.height);
        src_frame.data_mut(0).copy_from_slice(&frame.data);

        // Scale to YUV420P
        let mut sws = SwsContext::get(
            Pixel::BGRA,
            frame.width,
            frame.height,
            Pixel::YUV420P,
            frame.width,
            frame.height,
            Flags::BILINEAR,
        )?;
        let mut dst_frame = Video::new(Pixel::YUV420P, frame.width, frame.height);
        sws.run(&src_frame, &mut dst_frame)?;
        dst_frame.set_pts(Some(frame.timestamp as i64));

        encoder.send_frame(&dst_frame)?;
        self.flush_encoder()
    }

    #[cfg(feature = "video")]
    fn flush_encoder(&mut self) -> Result<()> {
        use ffmpeg_next::codec::packet::Packet;

        let encoder = self.encoder_ctx.as_mut().unwrap();
        let output = self.output_ctx.as_mut().unwrap();
        let mut packet = Packet::empty();

        while encoder.receive_packet(&mut packet).is_ok() {
            packet.write_interleaved(output)?;
        }
        Ok(())
    }

    #[cfg(feature = "video")]
    fn finalize_segment(&mut self) -> Result<()> {
        let result = self.write_segment_trailer();
        // Close the file handles even when the trailer failed, so the seal that
        // follows replaces a file nothing writes to any more.
        self.encoder_ctx = None;
        self.output_ctx = None;
        result
    }

    #[cfg(feature = "video")]
    fn write_segment_trailer(&mut self) -> Result<()> {
        if let Some(encoder) = &mut self.encoder_ctx {
            encoder.send_eof()?;
        }
        self.flush_encoder()?;
        if let Some(output) = &mut self.output_ctx {
            output.write_trailer()?;
        }
        Ok(())
    }

    pub fn current_segment_path(&self) -> Option<&Path> {
        self.current_segment_path.as_deref()
    }

    /// Save a single frame as a JPEG keyframe and register it in the timeline.
    ///
    /// This is the pure-Rust path (the `image` crate, no ffmpeg) that makes the
    /// timeline scrubber show real screenshots out of the box — independent of
    /// the optional `video` feature. The frame is BGRA and tightly packed
    /// (`width * height * 4`, row pitch already stripped by the capturer); we
    /// swap B/R, drop alpha, downscale wide frames to keep JPEGs small, and write
    /// to `<data>/media/keyframes/display-<id>/<ts>.jpg`.
    pub fn save_keyframe(&self, frame: &Frame) -> Result<()> {
        // Keyframes live beside the video dir under media/keyframes.
        let keyframes_root = self
            .output_dir
            .parent()
            .map(|p| p.join("keyframes"))
            .unwrap_or_else(|| self.output_dir.join("keyframes"));
        let out_dir = keyframes_root.join(format!("display-{}", self.display_id));
        std::fs::create_dir_all(&out_dir)?;

        let (w, h) = (frame.width, frame.height);
        let expected = w as usize * h as usize * 4;
        if frame.data.len() < expected {
            anyhow::bail!(
                "frame buffer too small: {} < {}",
                frame.data.len(),
                expected
            );
        }

        // BGRA -> RGB (drop alpha, swap channels).
        let mut rgb = image::RgbImage::new(w, h);
        for (px, src) in rgb.pixels_mut().zip(frame.data.chunks_exact(4)) {
            *px = image::Rgb([src[2], src[1], src[0]]);
        }

        // Downscale wide frames; the timeline thumbnail renders at 640x360.
        const MAX_WIDTH: u32 = 1280;
        let img = if w > MAX_WIDTH {
            let new_h = ((h as u64 * MAX_WIDTH as u64) / w as u64).max(1) as u32;
            image::imageops::resize(
                &rgb,
                MAX_WIDTH,
                new_h,
                image::imageops::FilterType::Triangle,
            )
        } else {
            rgb
        };

        let path = out_dir.join(format!("{}.jpg", frame.timestamp));
        let mut buf = Vec::new();
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 80);
        encoder.encode_image(&img)?;
        // Written sealed (RYU-SF01): the screenshot never touches the disk in plaintext.
        shadow_core::media::write_sealed_media(&path, &buf)?;

        shadow_core::insert_keyframe(
            self.display_id,
            frame.timestamp,
            path.to_string_lossy().to_string(),
            "direct".to_string(),
            Some(buf.len() as u64),
        )?;

        Ok(())
    }
}

/// Extracts keyframes from a recorded MP4 segment for retention.
pub struct FrameExtractor {
    output_dir: PathBuf,
}

impl FrameExtractor {
    pub fn new(keyframes_dir: &Path) -> Self {
        Self {
            output_dir: keyframes_dir.to_path_buf(),
        }
    }

    /// Extract keyframes from an MP4 file using ffmpeg CLI.
    pub fn extract_keyframes(&self, mp4_path: &Path) -> Result<Vec<PathBuf>> {
        let stem = mp4_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("segment");
        let out_dir = self.output_dir.join(stem);
        std::fs::create_dir_all(&out_dir)?;

        // A sealed segment is decrypted into an owner-only temp copy for ffmpeg
        // (deleted when `plain` drops).
        let plain = shadow_core::media::decrypt_media_to(mp4_path, &crate::paths::tmp_dir())?;
        let mut command = std::process::Command::new("ffmpeg");
        ryu_crypto::scrub_seal_env(&mut command);
        let output = command
            .args([
                "-i",
                plain.path().to_str().unwrap_or(""),
                "-vf",
                "select=eq(pict_type\\,I)",
                "-vsync",
                "vfr",
                "-q:v",
                "2",
                out_dir.join("keyframe-%04d.jpg").to_str().unwrap_or(""),
                "-y",
            ])
            .output()
            .map_err(|e| anyhow::anyhow!("ffmpeg not found: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("ffmpeg keyframe extraction failed: {}", stderr);
        }

        let paths: Vec<PathBuf> = std::fs::read_dir(&out_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "jpg").unwrap_or(false))
            .collect();

        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_segment_opens_without_a_sealing_key() {
        let dir = std::env::temp_dir().join(format!("shadow-video-{}", uuid::Uuid::new_v4()));
        let mut encoder = VideoEncoder::new(&dir, 0).unwrap();
        let error = encoder
            .rotate_segment_gated(
                1_700_000_000_000_000,
                64,
                64,
                Err(shadow_core::ShadowError::StorageError {
                    msg: "sealing: no-key".into(),
                }),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("until Shadow can seal"),
            "{error}"
        );
        assert!(encoder.current_segment_path().is_none());
        assert!(
            !dir.join("display-0").exists(),
            "no segment file was opened"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    const T0: u64 = 1_700_000_000_000_000;

    /// An encoder with an open segment, as `rotate_segment` leaves it, without
    /// ffmpeg or the timeline.
    fn encoder_with_open_segment(dir: &Path, start: u64) -> VideoEncoder {
        let mut encoder = VideoEncoder::new(dir, 0).unwrap();
        encoder.current_segment_path = Some(dir.join("display-0/segment.mp4"));
        encoder.segment_start_ts = Some(start);
        encoder.last_frame_ts = Some(start);
        encoder
    }

    #[test]
    fn close_is_due_by_wall_clock_age_idle_time_or_pause() {
        let minute = 60 * 1_000_000;
        assert!(!segment_close_due(None, None, T0, false), "nothing open");
        assert!(!segment_close_due(Some(T0), Some(T0), T0 + minute, true));
        assert!(segment_close_due(Some(T0), Some(T0), T0, false), "paused");
        assert!(segment_close_due(
            Some(T0),
            Some(T0),
            T0 + SEGMENT_IDLE_CLOSE_US,
            true
        ));
        // Frames keep arriving, but the hour is over.
        assert!(segment_close_due(
            Some(T0),
            Some(T0 + MAX_SEGMENT_AGE_US - minute),
            T0 + MAX_SEGMENT_AGE_US,
            true
        ));
        assert!(!segment_close_due(
            Some(T0),
            Some(T0 + 50 * minute),
            T0 + 52 * minute,
            true
        ));
    }

    #[test]
    fn a_segment_is_closed_and_sealed_while_no_frames_arrive() {
        let dir = std::env::temp_dir().join(format!("shadow-video-{}", uuid::Uuid::new_v4()));
        let mut encoder = encoder_with_open_segment(&dir, T0);
        let mut sealed = Vec::new();
        // A tick shortly after the last frame leaves the segment open.
        assert!(!encoder
            .close_segment_if_due_with(T0 + 1_000_000, true, |path, end| {
                sealed.push((path.to_path_buf(), end));
                Ok(())
            })
            .unwrap());
        assert!(encoder.current_segment_path().is_some());
        // No frame for the idle window: the tick alone closes and seals it.
        let now = T0 + SEGMENT_IDLE_CLOSE_US;
        assert!(encoder
            .close_segment_if_due_with(now, true, |path, end| {
                sealed.push((path.to_path_buf(), end));
                Ok(())
            })
            .unwrap());
        assert_eq!(sealed, vec![(dir.join("display-0/segment.mp4"), now)]);
        assert!(encoder.current_segment_path().is_none());
        assert!(encoder.segment_start_ts.is_none() && encoder.last_frame_ts.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pausing_capture_closes_and_seals_the_open_segment_at_once() {
        let dir = std::env::temp_dir().join(format!("shadow-video-{}", uuid::Uuid::new_v4()));
        let mut encoder = encoder_with_open_segment(&dir, T0);
        let mut sealed = None;
        assert!(encoder
            .close_segment_if_due_with(T0 + 2_000_000, false, |path, end| {
                sealed = Some((path.to_path_buf(), end));
                Ok(())
            })
            .unwrap());
        assert_eq!(
            sealed,
            Some((dir.join("display-0/segment.mp4"), T0 + 2_000_000))
        );
        assert!(encoder.current_segment_path().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn closing_with_no_open_segment_is_a_no_op() {
        let dir = std::env::temp_dir().join(format!("shadow-video-{}", uuid::Uuid::new_v4()));
        let mut encoder = VideoEncoder::new(&dir, 0).unwrap();
        encoder.close_segment(1).unwrap();
        assert!(encoder.current_segment_path().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
