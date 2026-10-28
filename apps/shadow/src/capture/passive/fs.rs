//! Filesystem capture — watches the user's meaningful directories (Documents,
//! Desktop, Downloads, and any `fsWatch` roots in `passive-capture.json`) and emits an event on
//! file create/modify/remove. Cross-platform via `notify`.
//!
//! Home is intentionally NOT watched wholesale: it is dominated by caches and app
//! state that bury real activity. Noise paths are filtered and each path is
//! throttled so a rapid save burst collapses to one event.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use notify::{EventKind, RecursiveMode, Watcher};

const THROTTLE: Duration = Duration::from_secs(3);

/// Path fragments that indicate machine noise rather than user activity.
const NOISE: &[&str] = &[
    "node_modules",
    "/.git/",
    "\\.git\\",
    "target/",
    "target\\",
    "/.cache",
    "\\AppData\\",
    "/Library/",
    "__pycache__",
    ".DS_Store",
    "~$", // Office lock files
];

fn is_noise(path: &Path) -> bool {
    let s = path.to_string_lossy();
    NOISE.iter().any(|n| s.contains(n))
}

/// Directories to watch: the standard user dirs plus the extra roots saved as
/// `fsWatch` in `passive-capture.json`.
fn watch_roots(extra: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(d) = dirs::document_dir() {
        roots.push(d);
    }
    if let Some(d) = dirs::desktop_dir() {
        roots.push(d);
    }
    if let Some(d) = dirs::download_dir() {
        roots.push(d);
    }
    roots.extend(extra);
    roots
}

fn kind_label(kind: &EventKind) -> Option<&'static str> {
    match kind {
        EventKind::Create(_) => Some("created"),
        EventKind::Modify(_) => Some("modified"),
        EventKind::Remove(_) => Some("removed"),
        _ => None, // Access / Any / Other — not user-meaningful
    }
}

/// Start filesystem monitoring on a dedicated thread that owns the watcher for the
/// process lifetime. `extra_roots` are watched on top of the standard user dirs.
pub fn start(extra_roots: Vec<PathBuf>) {
    std::thread::Builder::new()
        .name("shadow-fs".into())
        .spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut watcher = match notify::recommended_watcher(move |res| {
                let _ = tx.send(res);
            }) {
                Ok(w) => w,
                Err(e) => {
                    tracing::warn!("Filesystem capture unavailable: {e}");
                    return;
                }
            };

            let roots = watch_roots(extra_roots);
            let mut watching = 0usize;
            for root in &roots {
                if root.exists() && watcher.watch(root, RecursiveMode::Recursive).is_ok() {
                    watching += 1;
                }
            }
            if watching == 0 {
                tracing::warn!("Filesystem capture: no watchable directories found");
                return;
            }
            tracing::info!("Filesystem capture started ({watching} roots)");

            let mut last_emit: HashMap<PathBuf, Instant> = HashMap::new();
            for res in rx {
                let event = match res {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                let Some(label) = kind_label(&event.kind) else {
                    continue;
                };
                for path in event.paths {
                    if is_noise(&path) {
                        continue;
                    }
                    let now = Instant::now();
                    if let Some(prev) = last_emit.get(&path) {
                        if now.duration_since(*prev) < THROTTLE {
                            continue;
                        }
                    }
                    last_emit.insert(path.clone(), now);
                    if crate::server::is_capture_paused() {
                        continue;
                    }
                    let display = path.to_string_lossy().to_string();
                    super::emit(
                        super::TRACK_FILESYSTEM,
                        "file_change",
                        "Filesystem",
                        &super::truncate(&display, 1024),
                        vec![("change", rmpv::Value::from(label))],
                    );
                }

                // Bound the throttle map so a long session cannot grow it without limit.
                if last_emit.len() > 4096 {
                    let cutoff = Instant::now() - THROTTLE;
                    last_emit.retain(|_, t| *t > cutoff);
                }
            }
        })
        .ok();
}
