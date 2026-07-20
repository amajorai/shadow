//! DSP for the meeting recorder — the "real audio mixing" ported from meetily's
//! `audio_v2` pipeline, tuned for **ASR** rather than listenable playback.
//!
//! The transcription wins are the parts that keep the signal clean before whisper
//! sees it:
//!   - a **band-limited (sinc) resampler** that low-passes on the way down to
//!     16 kHz, instead of the old naive linear interpolation that aliased;
//!   - a per-source **RMS normalizer** so a quiet mic is brought up to a level
//!     whisper can actually recognize (with a gain cap so we don't amplify hiss);
//!   - a **peak limiter** on the summed mix so mic + system talking at once can't
//!     hard-clip into a square wave.
//!
//! Ducking (attenuating the mic while system audio speaks) is included but **off
//! by default**: it improves a *listenable* recording but actively hurts
//! recognition of cross-talk, which is exactly when a meeting needs both sides.
//!
//! Everything here is pure math over `f32` mono/stereo buffers, so it is fully
//! unit-tested and platform-independent — the capture side (`meeting.rs`) owns the
//! OS-specific stream plumbing; this module owns the samples.

/// The 16 kHz mono target whisper/parakeet expect.
pub const TARGET_RATE: u32 = 16_000;

/// Resample mono `input` from `src_rate` to [`TARGET_RATE`] using a band-limited
/// sinc filter (rubato). Unlike linear interpolation this applies an anti-aliasing
/// low-pass, so high-frequency content above the 8 kHz Nyquist doesn't fold back
/// as noise — audibly cleaner transcripts on 44.1/48 kHz capture.
///
/// Falls back to linear interpolation if rubato can't be constructed (e.g. a
/// degenerate chunk length), so a resample never fails the pipeline.
pub fn resample_to_16k(input: &[f32], src_rate: u32) -> Vec<f32> {
    if input.is_empty() || src_rate == TARGET_RATE {
        return input.to_vec();
    }
    match resample_sinc(input, src_rate) {
        Some(out) if !out.is_empty() => out,
        _ => resample_linear(input, src_rate),
    }
}

/// High-quality resample via rubato's sinc interpolator. Returns `None` if the
/// resampler couldn't be built or run, so the caller can fall back.
fn resample_sinc(input: &[f32], src_rate: u32) -> Option<Vec<f32>> {
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    // resample_ratio = out_rate / in_rate. SincFixedIn wants a fixed input chunk
    // size; we build one resampler per drain (once every ~20 s), sized to this
    // buffer, and run it a single time.
    let ratio = TARGET_RATE as f64 / src_rate as f64;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, input.len(), 1).ok()?;
    let waves_out = resampler.process(&[input.to_vec()], None).ok()?;
    waves_out.into_iter().next()
}

/// Linear-interpolation resample — the fallback. Cheap, no anti-aliasing.
fn resample_linear(input: &[f32], src_rate: u32) -> Vec<f32> {
    let ratio = TARGET_RATE as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

/// Root-mean-square level of a buffer (0.0 for empty).
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// Bring a source up (or down) toward `target_rms` so whisper sees a consistent
/// level. Gain is capped at `max_gain` so we don't blow up the noise floor of a
/// near-silent buffer, and never boosts a buffer that's already loud enough by
/// more than unity when `max_gain < 1` isn't requested. Silence is left alone.
pub fn normalize_rms(samples: &mut [f32], target_rms: f32, max_gain: f32) {
    let current = rms(samples);
    // A near-silent buffer has no signal to normalize; scaling it just amplifies
    // hiss, so leave it.
    if current < 1e-4 {
        return;
    }
    let gain = (target_rms / current).min(max_gain);
    // Only apply a meaningful change.
    if (gain - 1.0).abs() < 1e-3 {
        return;
    }
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

/// A peak limiter with an envelope follower: tracks the signal's running peak and
/// pulls gain down whenever it would exceed `threshold`, with a fast attack and a
/// slower release so it clamps transients without pumping. Prevents the summed
/// mix from hard-clipping into distortion that wrecks recognition.
pub struct Limiter {
    threshold: f32,
    attack: f32,
    release: f32,
    env: f32,
}

impl Limiter {
    /// A limiter targeting `threshold` (e.g. 0.95) at `sample_rate`, with `attack`
    /// and `release` in seconds.
    pub fn new(threshold: f32, attack_secs: f32, release_secs: f32, sample_rate: u32) -> Self {
        Self {
            threshold,
            attack: time_to_coeff(attack_secs, sample_rate),
            release: time_to_coeff(release_secs, sample_rate),
            env: 0.0,
        }
    }

    /// A sensible default for 16 kHz ASR: catch peaks just under full-scale.
    pub fn default_asr() -> Self {
        Self::new(0.95, 0.005, 0.050, TARGET_RATE)
    }

    /// Limit `samples` in place. After this call no sample exceeds ~`threshold`.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let x = s.abs();
            // Peak-tracking envelope: jump up fast on louder samples, decay slow.
            let coeff = if x > self.env {
                self.attack
            } else {
                self.release
            };
            self.env = coeff * self.env + (1.0 - coeff) * x;
            let gain = if self.env > self.threshold {
                self.threshold / self.env
            } else {
                1.0
            };
            *s = (*s * gain).clamp(-self.threshold, self.threshold);
        }
    }
}

/// Optional mic ducking: attenuates the mic toward `floor` gain while the system
/// (reference) channel is louder than `threshold`. Off by default in the ASR path
/// — see the module docs. Kept for a future "export a listenable recording" path.
pub struct Ducker {
    threshold: f32,
    floor: f32,
    attack: f32,
    release: f32,
    ref_env: f32,
    gain: f32,
}

impl Ducker {
    pub fn new(threshold: f32, floor: f32, sample_rate: u32) -> Self {
        Self {
            threshold,
            floor,
            attack: time_to_coeff(0.010, sample_rate),
            release: time_to_coeff(0.100, sample_rate),
            ref_env: 0.0,
            gain: 1.0,
        }
    }

    /// Attenuate `mic` in place based on the `reference` (system) channel's level.
    /// `mic` and `reference` should be the same length and rate.
    pub fn process(&mut self, mic: &mut [f32], reference: &[f32]) {
        for (i, s) in mic.iter_mut().enumerate() {
            let r = reference.get(i).copied().unwrap_or(0.0).abs();
            let coeff = if r > self.ref_env {
                self.attack
            } else {
                self.release
            };
            self.ref_env = coeff * self.ref_env + (1.0 - coeff) * r;
            let target = if self.ref_env > self.threshold {
                self.floor
            } else {
                1.0
            };
            // Smooth the gain toward the target so ducking doesn't click.
            let g_coeff = if target < self.gain {
                self.attack
            } else {
                self.release
            };
            self.gain = g_coeff * self.gain + (1.0 - g_coeff) * target;
            *s *= self.gain;
        }
    }
}

/// Convert an attack/release time constant (seconds) to a one-pole smoothing
/// coefficient at `sample_rate`. Longer time → coefficient closer to 1.
fn time_to_coeff(secs: f32, sample_rate: u32) -> f32 {
    if secs <= 0.0 {
        return 0.0;
    }
    (-1.0 / (secs * sample_rate as f32)).exp()
}

/// Resample both sources to 16 kHz, normalize each, sum, and limit — the mono
/// track fed to whisper. Optionally ducks the mic under system audio (default
/// `false`, which is best for cross-talk recognition).
pub fn mix_for_asr(
    mic: &[f32],
    mic_rate: u32,
    system: &[f32],
    sys_rate: u32,
    duck: bool,
) -> Vec<f32> {
    let mut mic16 = resample_to_16k(mic, mic_rate);
    let mut sys16 = resample_to_16k(system, sys_rate);

    // Per-source level normalization: lift a quiet mic, tame a loud system feed.
    normalize_rms(&mut mic16, 0.12, 8.0);
    normalize_rms(&mut sys16, 0.12, 8.0);

    if duck && !sys16.is_empty() {
        // Pad reference to mic length for the follower.
        let mut reference = sys16.clone();
        reference.resize(mic16.len(), 0.0);
        Ducker::new(0.05, 0.3, TARGET_RATE).process(&mut mic16, &reference);
    }

    let len = mic16.len().max(sys16.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let a = mic16.get(i).copied().unwrap_or(0.0);
        let b = sys16.get(i).copied().unwrap_or(0.0);
        out.push(a + b);
    }
    Limiter::default_asr().process(&mut out);
    out
}

/// Build an **interleaved stereo** 16 kHz track: left = mic, right = system. This
/// is what Core persists — keeping the two sides separate gives diarization a free,
/// rock-solid "Me vs. everyone-else" split (mic channel is always you) without
/// having to un-mix a mono blob. Each source is normalized (but not summed or
/// ducked — persistence keeps them clean and independent).
pub fn stereo_16k(mic: &[f32], mic_rate: u32, system: &[f32], sys_rate: u32) -> Vec<f32> {
    let mut mic16 = resample_to_16k(mic, mic_rate);
    let mut sys16 = resample_to_16k(system, sys_rate);
    normalize_rms(&mut mic16, 0.12, 8.0);
    normalize_rms(&mut sys16, 0.12, 8.0);
    let frames = mic16.len().max(sys16.len());
    let mut out = Vec::with_capacity(frames * 2);
    for i in 0..frames {
        out.push(mic16.get(i).copied().unwrap_or(0.0)); // L = mic
        out.push(sys16.get(i).copied().unwrap_or(0.0)); // R = system
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity_at_target_rate() {
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_to_16k(&input, TARGET_RATE), input);
    }

    #[test]
    fn resample_from_48k_thirds_the_length() {
        // 48 kHz → 16 kHz is a clean 3:1 decimation.
        let input = vec![0.0f32; 4800];
        let out = resample_to_16k(&input, 48_000);
        // Allow a few samples of filter edge slack.
        assert!((out.len() as i64 - 1600).abs() <= 130, "got {}", out.len());
    }

    #[test]
    fn resample_empty_is_empty() {
        assert!(resample_to_16k(&[], 48_000).is_empty());
    }

    #[test]
    fn rms_of_constant() {
        assert!((rms(&[0.5, -0.5, 0.5, -0.5]) - 0.5).abs() < 1e-6);
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn normalize_lifts_quiet_signal() {
        let mut s = vec![0.02f32, -0.02, 0.02, -0.02]; // rms 0.02
        normalize_rms(&mut s, 0.12, 8.0);
        assert!((rms(&s) - 0.12).abs() < 1e-3);
    }

    #[test]
    fn normalize_respects_gain_cap() {
        let mut s = vec![0.001f32, -0.001, 0.001, -0.001]; // rms 0.001, needs 120x
        normalize_rms(&mut s, 0.12, 8.0);
        // Capped at 8x → rms ~0.008, nowhere near target.
        assert!(rms(&s) <= 0.009);
    }

    #[test]
    fn normalize_leaves_silence_alone() {
        let mut s = vec![0.0f32; 8];
        normalize_rms(&mut s, 0.12, 8.0);
        assert!(s.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn limiter_clamps_loud_mix() {
        // A signal well over full scale must come out at/under threshold.
        let mut s = vec![2.0f32; 2000];
        Limiter::default_asr().process(&mut s);
        assert!(
            s.iter().all(|&x| x.abs() <= 0.96),
            "limiter let a peak through"
        );
    }

    #[test]
    fn limiter_leaves_quiet_signal_untouched() {
        let mut s = vec![0.1f32, -0.1, 0.1, -0.1];
        let orig = s.clone();
        Limiter::default_asr().process(&mut s);
        for (a, b) in s.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn mix_sums_both_sources_and_limits() {
        let mic = vec![0.5f32; 1000];
        let sys = vec![0.5f32; 1000];
        let mixed = mix_for_asr(&mic, TARGET_RATE, &sys, TARGET_RATE, false);
        assert_eq!(mixed.len(), 1000);
        assert!(mixed.iter().all(|&x| x.abs() <= 0.96));
    }

    #[test]
    fn mix_uses_longer_length() {
        let mic = vec![0.3f32, 0.3];
        let sys = vec![0.3f32];
        let mixed = mix_for_asr(&mic, TARGET_RATE, &sys, TARGET_RATE, false);
        assert_eq!(mixed.len(), 2);
    }

    #[test]
    fn stereo_interleaves_mic_left_system_right() {
        // System silent isolates placement: left (mic) must carry signal, right
        // (system) must stay zero. Amplitude can't be used because normalize_rms
        // deliberately equalizes both channels' levels.
        let mic = vec![0.4f32, 0.4];
        let sys = vec![0.0f32, 0.0];
        let st = stereo_16k(&mic, TARGET_RATE, &sys, TARGET_RATE);
        assert_eq!(st.len(), 4); // 2 frames * 2 channels
        assert!(
            st[0].abs() > 0.0 && st[2].abs() > 0.0,
            "mic (L) should have signal"
        );
        assert_eq!(st[1], 0.0); // system (R) silent
        assert_eq!(st[3], 0.0);
    }

    #[test]
    fn ducker_attenuates_mic_under_loud_reference() {
        let mut mic = vec![0.5f32; 2000];
        let reference = vec![0.5f32; 2000]; // loud system audio
        Ducker::new(0.05, 0.3, TARGET_RATE).process(&mut mic, &reference);
        // By the tail, the mic should be pulled well below its original level.
        assert!(
            mic[1999].abs() < 0.4,
            "ducker did not attenuate: {}",
            mic[1999]
        );
    }
}
