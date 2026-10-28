//! Persisted preferences for the passive capture sources.
//!
//! Stored as `passive-capture.json` in Shadow's data directory and read once when
//! the sources start. Every field is optional: an absent or empty file keeps the
//! built-in behavior (the standard user directories, auto-discovered git repos,
//! and no calendar).
//!
//! ```json
//! {
//!   "fsWatch": ["D:/Projects"],
//!   "gitRepos": ["D:/Code/ryu"],
//!   "calendarIcs": "C:/Users/me/calendar.ics"
//! }
//! ```

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// File name of the settings document inside Shadow's data directory.
pub const FILE_NAME: &str = "passive-capture.json";
const MAX_FILE_BYTES: u64 = 64 * 1024;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PassiveCaptureSettings {
    /// Extra filesystem roots to watch, on top of Documents, Desktop and Downloads.
    #[serde(default)]
    pub fs_watch: Vec<PathBuf>,
    /// Git repositories to follow, on top of the auto-discovered ones.
    #[serde(default)]
    pub git_repos: Vec<PathBuf>,
    /// An iCalendar (`.ics`) file to poll for upcoming meetings.
    #[serde(default)]
    pub calendar_ics: Option<PathBuf>,
}

impl PassiveCaptureSettings {
    /// Load the settings from `data_dir`, falling back to defaults when the file
    /// is missing, oversized, or invalid.
    pub fn load(data_dir: &Path) -> Self {
        let path = data_dir.join(FILE_NAME);
        if std::fs::metadata(&path)
            .map(|metadata| metadata.len() > MAX_FILE_BYTES)
            .unwrap_or(false)
        {
            tracing::warn!(
                "ignoring oversized passive capture settings at {}",
                path.display()
            );
            return Self::default();
        }
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return Self::default();
        };
        match serde_json::from_str::<Self>(&raw) {
            Ok(settings) => settings.normalized(),
            Err(error) => {
                tracing::warn!(
                    "ignoring invalid passive capture settings at {}: {error}",
                    path.display()
                );
                Self::default()
            }
        }
    }

    /// Drop blank entries so an empty string never becomes a watched path.
    fn normalized(mut self) -> Self {
        let non_empty = |path: &PathBuf| !path.as_os_str().is_empty();
        self.fs_watch.retain(non_empty);
        self.git_repos.retain(non_empty);
        self.calendar_ics = self.calendar_ics.filter(non_empty);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "shadow-passive-capture-{}",
            uuid::Uuid::new_v4().simple()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn missing_file_keeps_the_built_in_behavior() {
        let dir = temp_dir();
        assert_eq!(
            PassiveCaptureSettings::load(&dir),
            PassiveCaptureSettings::default()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn reads_the_persisted_preferences() {
        let dir = temp_dir();
        std::fs::write(
            dir.join(FILE_NAME),
            r#"{"fsWatch":["/data/a",""],"gitRepos":["/code/r"],"calendarIcs":"/cal.ics"}"#,
        )
        .unwrap();
        let settings = PassiveCaptureSettings::load(&dir);
        assert_eq!(settings.fs_watch, vec![PathBuf::from("/data/a")]);
        assert_eq!(settings.git_repos, vec![PathBuf::from("/code/r")]);
        assert_eq!(settings.calendar_ics, Some(PathBuf::from("/cal.ics")));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn invalid_or_unknown_fields_fall_back_to_defaults() {
        for raw in ["not-json", r#"{"fsWatch":[],"unknown":true}"#] {
            let dir = temp_dir();
            std::fs::write(dir.join(FILE_NAME), raw).unwrap();
            assert_eq!(
                PassiveCaptureSettings::load(&dir),
                PassiveCaptureSettings::default()
            );
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn blank_calendar_path_means_no_calendar() {
        let dir = temp_dir();
        std::fs::write(dir.join(FILE_NAME), r#"{"calendarIcs":""}"#).unwrap();
        assert_eq!(PassiveCaptureSettings::load(&dir).calendar_ics, None);
        let _ = std::fs::remove_dir_all(dir);
    }
}
