//! VAD-based chunk segmentation for the meeting recorder.
//!
//! The old recorder cut a chunk on a fixed 20 s wall-clock timer. That splits
//! words across boundaries (whisper never sees the whole word, so it mis-hears the
//! edges) and the transcript offset drifts because it was measured from wall-clock
//! at *ingest* time, not audio position.
//!
//! This segmenter cuts on **silence** instead. It accumulates 16 kHz audio and,
//! once a chunk is at least [`MIN_CHUNK_SECS`], looks for a quiet gap to cut on so
//! boundaries land between words; it force-cuts at [`MAX_CHUNK_SECS`] so a
//! monologue with no pauses still streams in near-real-time. Offsets are
//! **sample-accurate** — counted from emitted frames, not the clock — which fixes
//! the drift.
//!
//! Both sides are kept separate (mic + system) and emitted as an interleaved
//! stereo chunk (L = mic, R = system) so Core can persist the channel split that
//! diarization uses. Pure logic over `f32` buffers → fully unit-tested.

/// The 16 kHz mono target rate both accumulators run at.
pub const TARGET_RATE: u32 = 16_000;

/// Don't cut a chunk shorter than this even at a silence — keeps request overhead
/// down and gives whisper enough context.
const MIN_CHUNK_SECS: f32 = 8.0;
/// Force a cut here even with no silence, so notes stay live during a monologue.
const MAX_CHUNK_SECS: f32 = 24.0;
/// Envelope level below which a frame counts as silence.
const SILENCE_RMS: f32 = 0.015;
/// A silence gap must be at least this long to be a valid cut point.
const SILENCE_WIN_SECS: f32 = 0.35;

/// One emitted chunk: interleaved stereo 16 kHz audio plus where it sits in the
/// meeting timeline.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Interleaved stereo (L = mic, R = system) 16 kHz samples.
    pub stereo: Vec<f32>,
    /// Offset of this chunk's first frame from the meeting start, in ms.
    pub offset_ms: i64,
    /// Mono frame count (stereo.len() / 2).
    pub frames: usize,
}

/// Accumulates aligned mic + system audio and cuts it into VAD-aligned chunks.
pub struct VadSegmenter {
    mic: Vec<f32>,
    sys: Vec<f32>,
    /// Total frames already emitted — the basis for sample-accurate offsets.
    emitted_frames: u64,
    min_frames: usize,
    max_frames: usize,
    silence_win: usize,
    silence_rms: f32,
}

impl Default for VadSegmenter {
    fn default() -> Self {
        Self::new(SILENCE_RMS)
    }
}

impl VadSegmenter {
    pub fn new(silence_rms: f32) -> Self {
        let r = TARGET_RATE as f32;
        Self {
            mic: Vec::new(),
            sys: Vec::new(),
            emitted_frames: 0,
            min_frames: (MIN_CHUNK_SECS * r) as usize,
            max_frames: (MAX_CHUNK_SECS * r) as usize,
            silence_win: (SILENCE_WIN_SECS * r) as usize,
            silence_rms,
        }
    }

    /// Append one tick of resampled 16 kHz audio. The two sides are padded to the
    /// same length so the accumulators stay frame-aligned (mic[i] and sys[i] are
    /// the same instant).
    pub fn push(&mut self, mic16: &[f32], sys16: &[f32]) {
        let n = mic16.len().max(sys16.len());
        for i in 0..n {
            self.mic.push(mic16.get(i).copied().unwrap_or(0.0));
            self.sys.push(sys16.get(i).copied().unwrap_or(0.0));
        }
    }

    /// Try to cut a chunk. With `force` (meeting ending) it flushes whatever is
    /// buffered. Otherwise it emits when the buffer hits MAX, or when it's past MIN
    /// and a silence gap is found; else `None` (keep accumulating). Call in a loop
    /// until it returns `None`.
    pub fn try_cut(&mut self, force: bool) -> Option<Chunk> {
        let len = self.mic.len();
        if len == 0 {
            return None;
        }
        let cut = if force {
            len
        } else if len >= self.max_frames {
            self.max_frames
        } else if len >= self.min_frames {
            match self.find_silence_cut() {
                Some(c) => c,
                None => return None,
            }
        } else {
            return None;
        };
        Some(self.emit(cut))
    }

    /// Envelope of frame `i` — the louder of the two channels.
    fn envelope(&self, i: usize) -> f32 {
        self.mic[i].abs().max(self.sys[i].abs())
    }

    /// Find a cut index at the middle of the first silence run (≥ `silence_win`
    /// frames below threshold) starting at or after `min_frames`. `None` if the
    /// tail hasn't gone quiet yet.
    fn find_silence_cut(&self) -> Option<usize> {
        let len = self.mic.len();
        let mut run_start: Option<usize> = None;
        for i in self.min_frames..len {
            if self.envelope(i) < self.silence_rms {
                let start = *run_start.get_or_insert(i);
                if i - start + 1 >= self.silence_win {
                    // Cut at the middle of the silence so trailing quiet splits
                    // between this chunk and the next.
                    return Some((start + i) / 2);
                }
            } else {
                run_start = None;
            }
        }
        None
    }

    /// Split off `cut` frames as a stereo chunk and advance the offset counter.
    fn emit(&mut self, cut: usize) -> Chunk {
        let cut = cut.min(self.mic.len());
        let mic_head: Vec<f32> = self.mic.drain(0..cut).collect();
        let sys_head: Vec<f32> = self.sys.drain(0..cut).collect();
        let mut stereo = Vec::with_capacity(cut * 2);
        for i in 0..cut {
            stereo.push(mic_head[i]); // L = mic
            stereo.push(sys_head[i]); // R = system
        }
        let offset_ms = (self.emitted_frames * 1000 / TARGET_RATE as u64) as i64;
        self.emitted_frames += cut as u64;
        Chunk {
            stereo,
            offset_ms,
            frames: cut,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(n: f32) -> usize {
        (n * TARGET_RATE as f32) as usize
    }

    #[test]
    fn does_not_cut_below_min() {
        let mut seg = VadSegmenter::default();
        seg.push(&vec![0.5; secs(3.0)], &vec![0.0; secs(3.0)]);
        assert!(seg.try_cut(false).is_none());
    }

    #[test]
    fn force_flushes_remainder() {
        let mut seg = VadSegmenter::default();
        seg.push(&vec![0.5; secs(3.0)], &vec![0.0; secs(3.0)]);
        let chunk = seg.try_cut(true).expect("force should flush");
        assert_eq!(chunk.frames, secs(3.0));
        assert_eq!(chunk.stereo.len(), secs(3.0) * 2);
        assert_eq!(chunk.offset_ms, 0);
        assert!(seg.try_cut(true).is_none()); // nothing left
    }

    #[test]
    fn force_cuts_at_max_when_no_silence() {
        let mut seg = VadSegmenter::default();
        // 30 s of continuous loud audio, no gaps.
        seg.push(&vec![0.5; secs(30.0)], &vec![0.0; secs(30.0)]);
        let chunk = seg.try_cut(false).expect("should cut at MAX");
        assert_eq!(chunk.frames, seg.max_frames);
        assert_eq!(chunk.offset_ms, 0);
    }

    #[test]
    fn cuts_on_silence_gap_past_min() {
        let mut seg = VadSegmenter::default();
        // 10 s speech, then ~0.5 s silence, then more speech.
        let mut mic = vec![0.5f32; secs(10.0)];
        mic.extend(vec![0.0f32; secs(0.5)]);
        mic.extend(vec![0.5f32; secs(5.0)]);
        let sys = vec![0.0f32; mic.len()];
        seg.push(&mic, &sys);
        let chunk = seg.try_cut(false).expect("should cut at the silence");
        // Cut lands inside the silence window (~10.0–10.5 s).
        assert!(
            chunk.frames >= secs(10.0) && chunk.frames <= secs(10.5),
            "cut at {} frames (~{:.2}s)",
            chunk.frames,
            chunk.frames as f32 / TARGET_RATE as f32
        );
    }

    #[test]
    fn offsets_are_sample_accurate_across_chunks() {
        let mut seg = VadSegmenter::default();
        seg.push(&vec![0.5; secs(12.0)], &vec![0.0; secs(12.0)]);
        let first = seg.try_cut(true).unwrap();
        assert_eq!(first.offset_ms, 0);
        // Next chunk's offset is exactly the first chunk's frame count in ms.
        seg.push(&vec![0.5; secs(6.0)], &vec![0.0; secs(6.0)]);
        let second = seg.try_cut(true).unwrap();
        assert_eq!(second.offset_ms, (first.frames as i64) * 1000 / 16_000);
    }

    #[test]
    fn interleaves_mic_left_system_right() {
        let mut seg = VadSegmenter::default();
        seg.push(&[0.4, 0.4], &[0.1, 0.1]);
        let chunk = seg.try_cut(true).unwrap();
        assert_eq!(chunk.stereo, vec![0.4, 0.1, 0.4, 0.1]);
    }
}
