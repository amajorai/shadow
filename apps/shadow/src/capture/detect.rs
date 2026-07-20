//! Microphone-in-use detection — the mechanic behind automatic meeting detection.
//!
//! Granola and Notion AI do **not** start notes because a meeting app is focused;
//! they watch the OS for *a process actively using the microphone* (Notion: "the
//! desktop app observes if a process is actively using the microphone, e.g. Zoom…
//! it does not listen to the audio"). That is an OS-level signal, device-local, so
//! Shadow (the local sensor) owns it and reports the owning process to Core. Core
//! is the brain: it decides whether that process is a *meeting* (matching its
//! configured app list) and prompts the user.
//!
//! - **Windows** (primary): the `CapabilityAccessManager\ConsentStore\microphone`
//!   registry records every app's mic usage; an app is using the mic *right now*
//!   when its `LastUsedTimeStop == 0`.
//! - **Linux** (PulseAudio/PipeWire): `pactl list source-outputs` lists every
//!   process *currently recording*; we report the owning application's name. This
//!   is an accurate per-process signal, like Windows.
//! - **macOS**: there is no public per-process "is the mic in use" API. As a
//!   heuristic we scan running processes for a known meeting app (`ps`), which is
//!   coarser — it fires when a meeting app is running, and Core's app-list filter +
//!   debounce keep it from being noisy. A native CoreAudio/TCC signal is the
//!   eventual upgrade.

/// A process currently holding the microphone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicUser {
    /// A best-effort process/app identifier (e.g. `Zoom.exe`, `MSTeams`). Core
    /// substring-matches this against its meeting-app slug list.
    pub process: String,
}

/// Poll the OS for whether (and by whom) the microphone is in use right now.
#[cfg(windows)]
pub fn microphone_in_use() -> Option<MicUser> {
    use winreg::enums::HKEY_CURRENT_USER;
    use winreg::RegKey;

    const BASE: &str = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone";
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let mic = hkcu.open_subkey(BASE).ok()?;

    // Packaged (Store) apps are direct subkeys; classic desktop apps live under
    // the `NonPackaged` subkey. Scan both.
    if let Some(user) = scan_consent_store(&mic) {
        return Some(user);
    }
    if let Ok(non_packaged) = mic.open_subkey("NonPackaged") {
        if let Some(user) = scan_consent_store(&non_packaged) {
            return Some(user);
        }
    }
    None
}

/// Scan one ConsentStore key for a subkey whose `LastUsedTimeStop == 0` (in use
/// now), returning a friendly process name.
#[cfg(windows)]
fn scan_consent_store(key: &winreg::RegKey) -> Option<MicUser> {
    for name in key.enum_keys().flatten() {
        if name == "NonPackaged" {
            continue;
        }
        let Ok(sub) = key.open_subkey(&name) else {
            continue;
        };
        // Present + zero ⇒ started but not stopped ⇒ currently using the mic.
        let stop: u64 = sub.get_value("LastUsedTimeStop").unwrap_or(u64::MAX);
        if stop == 0 {
            return Some(MicUser {
                process: friendly_process_name(&name),
            });
        }
    }
    None
}

/// Turn a ConsentStore subkey name into a short process identifier.
///
/// NonPackaged keys are exe paths with `\` replaced by `#`
/// (`C:#Program Files#Zoom#bin#Zoom.exe`) — take the final segment. Packaged keys
/// are PFNs (`Microsoft.Teams_8wekyb3d8bbwe`) — keep as-is; Core's case-insensitive
/// substring match against `teams` still hits.
#[cfg(windows)]
fn friendly_process_name(key_name: &str) -> String {
    key_name
        .rsplit('#')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(key_name)
        .to_string()
}

/// Linux: ask PulseAudio/PipeWire which processes are recording right now via
/// `pactl list source-outputs`, and return the owning application's name. A
/// source-output whose source is a monitor is playback, not a mic — we skip those.
#[cfg(target_os = "linux")]
pub fn microphone_in_use() -> Option<MicUser> {
    let output = std::process::Command::new("pactl")
        .args(["list", "source-outputs"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // Each "Source Output #N" block has an `application.name = "…"` property and a
    // `Source:` line. A real mic capture reads a non-monitor source.
    let mut app: Option<String> = None;
    let mut on_monitor = false;
    for block in text.split("Source Output #") {
        app = None;
        on_monitor = false;
        for line in block.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("application.name = ") {
                app = Some(rest.trim_matches('"').to_string());
            } else if trimmed.starts_with("Source:") && trimmed.to_lowercase().contains("monitor") {
                on_monitor = true;
            }
        }
        if !on_monitor {
            if let Some(name) = app.clone() {
                if !name.is_empty() {
                    return Some(MicUser { process: name });
                }
            }
        }
    }
    let _ = on_monitor;
    None
}

/// A small built-in list of meeting-app process names used only to *gate* the
/// macOS heuristic below. Core still authoritatively filters detections against
/// its (user-editable) meeting-app list; this just avoids reporting every process.
#[cfg(target_os = "macos")]
const MAC_MEETING_PROCS: &[&str] = &[
    "zoom.us", "zoom", "Microsoft Teams", "Teams", "Webex", "Google Chrome Helper",
    "Slack", "Discord", "FaceTime", "Skype", "GoToMeeting", "Around", "Whereby",
];

/// macOS: no public per-process mic API, so scan running processes for a known
/// meeting app. Coarser than the Windows/Linux paths (it reports when a meeting
/// app is *running*, not strictly while the mic is live).
#[cfg(target_os = "macos")]
pub fn microphone_in_use() -> Option<MicUser> {
    let output = std::process::Command::new("ps")
        .args(["-Ao", "comm="])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let name = line.trim();
        let lower = name.to_lowercase();
        for proc in MAC_MEETING_PROCS {
            if lower.contains(&proc.to_lowercase()) {
                // Report the last path segment (the binary name).
                let short = name.rsplit('/').next().unwrap_or(name);
                return Some(MicUser {
                    process: short.to_string(),
                });
            }
        }
    }
    None
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub fn microphone_in_use() -> Option<MicUser> {
    None
}

/// Spawn the background poller: every few seconds, check OS mic-in-use and, on the
/// rising edge (mic newly in use, and no meeting already recording), report the
/// owning process to Core's `POST /api/meetings/detect`. Core filters it against
/// the meeting-app list and decides whether to prompt.
pub fn spawn_poller(core_url: String) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let url = format!("{}/api/meetings/detect", core_url.trim_end_matches('/'));
        // Track the last-seen process so we only fire on a transition into use,
        // not every poll while a call is ongoing.
        let mut previous: Option<String> = None;
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(4));
        loop {
            ticker.tick().await;

            // Don't prompt while we're already recording a meeting.
            if crate::capture::meeting::is_recording() {
                previous = None;
                continue;
            }

            let current = tokio::task::spawn_blocking(microphone_in_use)
                .await
                .ok()
                .flatten()
                .map(|u| u.process);

            match (&previous, &current) {
                // Rising edge (or the owning process changed) → report to Core.
                (None, Some(proc)) => {
                    report(&client, &url, proc).await;
                }
                (Some(prev), Some(proc)) if prev != proc => {
                    report(&client, &url, proc).await;
                }
                _ => {}
            }
            previous = current;
        }
    });
}

async fn report(client: &reqwest::Client, url: &str, process: &str) {
    let body = serde_json::json!({ "app": process });
    match client
        .post(url)
        .timeout(std::time::Duration::from_secs(5))
        .json(&body)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!("meeting detect: reported mic-in-use by '{process}' to Core");
        }
        Ok(resp) => tracing::trace!("meeting detect: Core returned {}", resp.status()),
        Err(e) => tracing::trace!("meeting detect: Core not reachable ({e})"),
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn friendly_name_takes_last_segment() {
        assert_eq!(
            friendly_process_name("C:#Program Files#Zoom#bin#Zoom.exe"),
            "Zoom.exe"
        );
        assert_eq!(
            friendly_process_name("Microsoft.Teams_8wekyb3d8bbwe"),
            "Microsoft.Teams_8wekyb3d8bbwe"
        );
    }
}
