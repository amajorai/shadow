//! Where Shadow keeps its state.
//!
//! Shadow is a first-party Ryu service, so everything it persists lives in the
//! Ryu data dir (`RYU_DIR`, the Storage pointer, or the profile default — see
//! `ryu_sidecar_runtime::ryu_dir`):
//!
//! - `services/shadow/` — events, media (video, audio, clips), indices, the API
//!   token and settings;
//! - `models/vision/shadow/` — the models Shadow downloads (whisper, ShowUI, …).
//!
//! Core exports `SHADOW_DATA_DIR` / `SHADOW_MODELS_DIR` when it spawns Shadow,
//! so a CLI run (`shadow search`) and a Core-managed run resolve the same dirs.

use std::path::PathBuf;

use ryu_sidecar_runtime::layout;

/// Env override for Shadow's data dir (Core sets it to `services/shadow`).
pub const DATA_DIR_ENV: &str = "SHADOW_DATA_DIR";
/// Env override for Shadow's model dir (Core sets it to `models/vision/shadow`).
pub const MODELS_DIR_ENV: &str = "SHADOW_MODELS_DIR";

fn env_dir(key: &str) -> Option<PathBuf> {
    let value = std::env::var(key).ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| PathBuf::from(trimmed))
}

/// `<data dir>/services/shadow`, or `SHADOW_DATA_DIR` when set.
pub fn data_dir() -> PathBuf {
    env_dir(DATA_DIR_ENV)
        .unwrap_or_else(|| layout::service_dir(&ryu_sidecar_runtime::ryu_dir(), "shadow"))
}

/// `<data dir>/models/vision/shadow`, or `SHADOW_MODELS_DIR` when set.
pub fn models_dir() -> PathBuf {
    env_dir(MODELS_DIR_ENV).unwrap_or_else(|| {
        layout::model_family_dir(
            &ryu_sidecar_runtime::ryu_dir(),
            layout::Modality::Vision,
            "shadow",
        )
    })
}

/// Owner-only scratch space for plaintext copies of sealed media that a tool
/// needs as a file (ffmpeg, whisper): `<data dir>/tmp/shadow`. Callers delete
/// their copy when done; startup clears what a crash left behind.
pub fn tmp_dir() -> PathBuf {
    layout::tmp_dir_for(&ryu_sidecar_runtime::ryu_dir(), "shadow")
}
