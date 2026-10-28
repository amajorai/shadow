//! Terminal capture — records shell commands and their exit codes.
//!
//! Unlike the other sources there is no reliable, portable way to observe another
//! process's shell from outside it, so capture is opt-in via a shell hook: this
//! module writes ready-to-source hook scripts into the data dir on startup. The
//! user enables it once by sourcing the snippet from their shell rc file; the
//! path is logged on startup.
//!
//! The hooks never talk to Shadow over the network and never touch its API token.
//! Each one appends a JSON line (`{"ts":…,"exit_code":…,"cmd":"…"}`) to a spool
//! file in a private directory (`<data_dir>/shell-hooks/spool/`: mode 0700 on
//! Unix, an owner-only ACL on Windows, appended under `umask 077`), and a
//! background thread here drains that spool into track 9 (`shell_command`). An
//! earlier design POSTed to `/ingest` with the bearer, which meant every prompt
//! handed the token to whatever process held Shadow's loopback port; a shell has
//! no safe way to verify that listener first (`openssl dgst -hmac` would put the
//! token in argv, readable by other users). File permissions give the same-user
//! guarantee with no secret at all.
//!
//! The drainer writes its PID to `spool/.drainer` and rewrites it on every pass
//! (so a second instance that overwrote it and then exited loses it again within
//! one interval); a graceful stop removes it. The hooks append only while that
//! process is alive (shell builtins only), so the spool does not grow while
//! Shadow is stopped, crashed, or uninstalled with its data dir kept. The marker
//! is a bare PID: if a crash leaves it behind and the OS later reuses that PID,
//! hooks append until Shadow next starts and overwrites it.
//!
//! Hooks honor the shell's own history privacy: a command the shell keeps out of
//! history (zsh `HIST_IGNORE_SPACE`, bash `HISTCONTROL=ignorespace`, a PowerShell
//! space-prefixed line or one PSReadLine's history handler skips, such as lines
//! with `password`/`token`/`secret`) is not recorded. An empty prompt does not
//! re-record the previous command, and the first bash prompt of a session never
//! records a line loaded from `HISTFILE`.
//!
//! Each pass first drains the files a previous pass took aside, then takes the
//! live spool aside for the next pass. The one-interval grace lets a hook that
//! opened the file just before the rename finish its append. A drained file is
//! deleted (or emptied) before its commands are emitted, so a file that cannot be
//! removed is never emitted twice.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Placeholder substituted with the absolute spool file path when the hook
/// scripts are written.
const SPOOL_FILE_PLACEHOLDER: &str = "__SHADOW_TERMINAL_SPOOL__";

/// Placeholder substituted with the absolute drainer marker path.
const MARKER_FILE_PLACEHOLDER: &str = "__SHADOW_TERMINAL_MARKER__";

/// Name of the file the hooks append to, inside [`spool_dir`].
const SPOOL_FILE: &str = "commands.jsonl";

/// Name of the file holding the live drainer's PID, inside [`spool_dir`].
const MARKER_FILE: &str = ".drainer";

/// Suffix of a spool file taken aside for draining.
const DRAINING_SUFFIX: &str = ".draining";

/// How often the spool is drained.
const DRAIN_INTERVAL: Duration = Duration::from_secs(2);

/// Lines longer than this are skipped rather than parsed (a pasted blob is not a
/// command worth indexing, and the reader must stay bounded).
const MAX_LINE_BYTES: usize = 16 * 1024;

/// Most bytes read from one spool file; anything past it is dropped. Two seconds
/// of prompts never come close, so this only bounds a corrupted or forged file.
const MAX_SPOOL_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Longest command text kept on the event.
const MAX_COMMAND_CHARS: usize = 2048;

/// A hook timestamp later than "now" plus this is treated as clock skew and
/// replaced by "now". Past timestamps are kept: a command run while the drain
/// was briefly behind belongs where it happened on the timeline.
const MAX_FUTURE_SKEW_MICROS: u64 = 60 * 1_000_000;

/// Earliest plausible hook timestamp (2001-09-09, 10^15 µs). Anything earlier is
/// a unit mistake or a hand-written line and is replaced by "now".
const MIN_PLAUSIBLE_TS_MICROS: u64 = 1_000_000_000_000_000;

const BASH_HOOK: &str = r#"# Shadow terminal capture (bash/zsh). Source this from ~/.bashrc or ~/.zshrc.
# Records each command + exit code for the local Shadow sidecar by appending a
# line to a user-only spool file while Shadow is running. Nothing is sent over
# the network and no API token is read. Commands your shell keeps out of history
# (HIST_IGNORE_SPACE / HISTCONTROL=ignorespace) are not recorded. In bash 4.4+ a
# PS0 flag marks that a command actually ran, so repeats that HISTCONTROL=ignoredups
# or erasedups folds into one history entry are still recorded, while an empty
# prompt or an entry pulled in by `history -n` is not.
__shadow_spool='__SHADOW_TERMINAL_SPOOL__'
__shadow_marker='__SHADOW_TERMINAL_MARKER__'
__shadow_pending=''
__shadow_last=''
__shadow_ran=''
__shadow_ps0=''
__shadow_preexec() { __shadow_pending=${1-}; }
__shadow_report() {
  local code=$?
  local cmd='' line num pid now esc
  if [ -n "${ZSH_VERSION-}" ]; then
    # zsh: preexec saw the line exactly as typed; an empty prompt runs no preexec.
    cmd=${__shadow_pending-}
    __shadow_pending=''
    [ -z "$cmd" ] && return $code
    # HIST_IGNORE_SPACE keeps a space-prefixed command out of history; honor it.
    if [[ -o histignorespace && $cmd == ' '* ]]; then return $code; fi
  else
    # bash: the newest history entry. PS0 (bash 4.4+) flags that a command ran, so
    # an empty prompt, or an entry another shell added (`history -n`), is not
    # recorded; the history number alone only dedupes on older bash.
    local ran=${__shadow_ran-}
    __shadow_ran=''
    line=$(HISTTIMEFORMAT='' builtin history 1 2>/dev/null) || return $code
    line=${line#"${line%%[![:space:]]*}"}
    num=${line%%[[:space:]]*}
    case $num in
      ''|*[!0-9]*)
        # Empty history at the first prompt: the baseline is "nothing yet".
        [ "${__shadow_last-}" = __init__ ] && __shadow_last=0
        return $code ;;
    esac
    # The first prompt only sets the baseline, so history loaded from HISTFILE
    # (an earlier session's last line) is never recorded as a new command.
    if [ "${__shadow_last-}" = __init__ ]; then __shadow_last=$num; return $code; fi
    if [ -n "${__shadow_ps0-}" ] && [ -z "$ran" ]; then
      __shadow_last=$num
      return $code
    fi
    if [ "$num" = "${__shadow_last-}" ]; then
      # A command ran but history did not advance: bash kept it out. Record it
      # only when that is a duplicate fold (ignoredups/erasedups), never for
      # ignorespace, HISTIGNORE or disabled history, which are privacy choices.
      [ -n "${__shadow_ps0-}" ] || return $code
      [ -z "${HISTIGNORE-}" ] || return $code
      [ -o history ] || return $code
      case ":${HISTCONTROL-}:" in
        *:ignorespace:*|*:ignoreboth:*) return $code ;;
        *:ignoredups:*|*:erasedups:*) ;;
        *) return $code ;;
      esac
    fi
    __shadow_last=$num
    cmd=${line#"$num"}
    cmd=${cmd#"${cmd%%[![:space:]]*}"}
    [ -z "$cmd" ] && return $code
  fi
  # Only while Shadow's spool reader runs, so the spool cannot pile up.
  { read -r pid < "$__shadow_marker"; } 2>/dev/null || return $code
  case $pid in ''|*[!0-9]*) return $code ;; esac
  kill -0 "$pid" 2>/dev/null || return $code
  now=$(date +%s 2>/dev/null) || return $code
  esc=$(printf '%s' "$cmd" | tr '\n\r\t' '   ' | tr -d '[:cntrl:]' | sed 's/\\/\\\\/g; s/"/\\"/g') || return $code
  ( umask 077
    printf '{"ts":%s000000,"exit_code":%s,"cmd":"%s"}\n' "$now" "$code" "$esc" \
      >> "$__shadow_spool" ) 2>/dev/null || :
  return $code
}
if [ -n "${ZSH_VERSION-}" ]; then
  (( ${precmd_functions[(I)__shadow_report]} )) || precmd_functions+=(__shadow_report)
  (( ${preexec_functions[(I)__shadow_preexec]} )) || preexec_functions+=(__shadow_preexec)
elif [ -n "${BASH_VERSION-}" ]; then
  # The first prompt records the baseline: whatever history already holds
  # (including HISTFILE contents loaded after this file was sourced).
  __shadow_last=__init__
  # PS0 (bash 4.4+) runs once per executed command, never for an empty line.
  if [ "${BASH_VERSINFO[0]}" -gt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 4 ]; }; then
    __shadow_ps0=1
    case "${PS0-}" in
      *__shadow_ran*) ;;
      *) PS0="${PS0-}"'${__shadow_nul[__shadow_ran=1]-}' ;;
    esac
  fi
  case "${PROMPT_COMMAND-}" in
    *__shadow_report*) ;;
    *) PROMPT_COMMAND="__shadow_report;${PROMPT_COMMAND-}" ;;
  esac
fi
"#;

const PWSH_HOOK: &str = r#"# Shadow terminal capture (PowerShell). Dot-source this from $PROFILE.
# Records each command + exit code for the local Shadow sidecar by appending a
# line to a spool file in your Shadow data dir while Shadow is running. Nothing
# is sent over the network and no API token is read. Space-prefixed commands and
# ones PSReadLine keeps out of history (for example lines containing "password")
# are not recorded.
$global:__shadowSpool = '__SHADOW_TERMINAL_SPOOL__'
$global:__shadowMarker = '__SHADOW_TERMINAL_MARKER__'
$global:__shadowLastId = (Get-History -Count 1).Id
# Honor the shell's history privacy: a space-prefixed line, or one PSReadLine's
# AddToHistoryHandler keeps out of (or only in memory, not) its history, is not
# recorded. A handler that throws is treated as "skip" (fails closed).
function global:__shadowKeep($line) {
  if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith(' ')) { return $false }
  if (-not (Get-Command Get-PSReadLineOption -ErrorAction SilentlyContinue)) { return $true }
  try {
    $handler = (Get-PSReadLineOption -ErrorAction Stop).AddToHistoryHandler
    if ($handler) {
      $verdict = "$($handler.Invoke($line))"
      if ($verdict -in 'False', 'SkipAdding', 'MemoryOnly') { return $false }
    }
  } catch { return $false }
  return $true
}
function global:prompt {
  $code = if ($?) { 0 } else { 1 }
  $entry = Get-History -Count 1
  if ($entry -and $entry.Id -ne $global:__shadowLastId) {
    $global:__shadowLastId = $entry.Id
    try {
      $drainer = [System.IO.File]::ReadAllText($global:__shadowMarker).Trim()
      if ((__shadowKeep $entry.CommandLine) -and $drainer -match '^\d+$' -and (Get-Process -Id ([int]$drainer) -ErrorAction SilentlyContinue)) {
        $ts = [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) * 1000
        $line = @{ ts = $ts; exit_code = $code; cmd = $entry.CommandLine } | ConvertTo-Json -Compress
        $bytes = (New-Object System.Text.UTF8Encoding $false).GetBytes($line + "`n")
        for ($i = 0; $i -lt 3; $i++) {
          try {
            $fs = New-Object System.IO.FileStream($global:__shadowSpool, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
            try { $fs.Write($bytes, 0, $bytes.Length) } finally { $fs.Dispose() }
            break
          } catch { Start-Sleep -Milliseconds 20 }
        }
      }
    } catch {}
  }
  "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
}
"#;

/// The private directory the hooks append to.
fn spool_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("shell-hooks").join("spool")
}

/// Create `dir` (and parents) and restrict it to the current user, so no other
/// local account can write commands into the spool or read them back. Fails
/// closed: without a private directory terminal capture does not start.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    }
    #[cfg(windows)]
    {
        // Same owner-only ACL Core uses for its credential directories: drop
        // inherited entries (a data dir moved to e.g. D:\ inherits
        // Authenticated Users) and grant only the current account.
        use std::os::windows::process::CommandExt;
        let unavailable = |what: &str| std::io::Error::other(format!("Windows {what} unavailable"));
        let system =
            std::env::var_os("SystemRoot").ok_or_else(|| unavailable("system directory"))?;
        let username = std::env::var("USERNAME").map_err(|_| unavailable("account name"))?;
        let domain = std::env::var("USERDOMAIN").map_err(|_| unavailable("account domain"))?;
        if username.is_empty() || domain.is_empty() {
            return Err(unavailable("account identity"));
        }
        let grant = format!("{domain}\\{username}:(OI)(CI)F");
        let output = std::process::Command::new(PathBuf::from(system).join("System32/icacls.exe"))
            .arg(dir)
            .args(["/inheritance:r", "/grant:r", &grant])
            .creation_flags(0x0800_0000)
            .output()?;
        if !output.status.success() {
            return Err(std::io::Error::other(
                "could not restrict the spool directory ACL",
            ));
        }
    }
    Ok(())
}

/// Escape `path` for a single-quoted bash/zsh literal.
fn bash_single_quoted(path: &str) -> String {
    path.replace('\'', r"'\''")
}

/// Escape `path` for a single-quoted PowerShell literal. PowerShell also treats
/// the curly single quotes U+2018–U+201B as quote characters; each is doubled.
fn pwsh_single_quoted(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for ch in path.chars() {
        if matches!(ch, '\'' | '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}') {
            out.push(ch);
        }
        out.push(ch);
    }
    out
}

/// The hook scripts with the absolute spool and marker paths baked in.
fn render_hooks(spool_file: &Path, marker_file: &Path) -> (String, String) {
    let spool = spool_file.to_string_lossy();
    let marker = marker_file.to_string_lossy();
    (
        BASH_HOOK
            .replace(SPOOL_FILE_PLACEHOLDER, &bash_single_quoted(&spool))
            .replace(MARKER_FILE_PLACEHOLDER, &bash_single_quoted(&marker)),
        PWSH_HOOK
            .replace(SPOOL_FILE_PLACEHOLDER, &pwsh_single_quoted(&spool))
            .replace(MARKER_FILE_PLACEHOLDER, &pwsh_single_quoted(&marker)),
    )
}

/// Publish this process as the live drainer, atomically.
fn write_marker(dir: &Path) -> std::io::Result<()> {
    let tmp = dir.join(format!("{MARKER_FILE}.{}.tmp", std::process::id()));
    fs::write(&tmp, format!("{}\n", std::process::id()))?;
    fs::rename(&tmp, dir.join(MARKER_FILE))
}

/// Remove the marker, but only while it still names this process: after a second
/// instance overwrote it, the first instance's marker belongs to that instance.
fn remove_marker_if_ours(dir: &Path) {
    let marker = dir.join(MARKER_FILE);
    let ours =
        fs::read_to_string(&marker).is_ok_and(|pid| pid.trim() == std::process::id().to_string());
    if ours {
        let _ = fs::remove_file(&marker);
    }
}

/// Spool dir of the running drainer, for [`shutdown`].
static ACTIVE_SPOOL: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Stop publishing the drainer marker and remove it (if it is this process's), so
/// the hooks stop appending as soon as Shadow stops. Called on a graceful stop;
/// a crash leaves the marker, which a later start overwrites.
pub fn shutdown() {
    let mut active = ACTIVE_SPOOL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(dir) = active.take() {
        remove_marker_if_ours(&dir);
    }
}

/// Refresh the marker while the drainer is still active (not shut down). The lock
/// makes this and [`shutdown`] mutually exclusive, so a refresh never recreates a
/// marker that shutdown just removed.
fn refresh_marker(dir: &Path) -> bool {
    let active = ACTIVE_SPOOL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if active.as_deref() != Some(dir) {
        return false;
    }
    if let Err(e) = write_marker(dir) {
        tracing::warn!("Terminal capture: could not publish the spool reader: {e}");
    }
    true
}

/// One command read from the spool.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SpooledCommand {
    ts: u64,
    exit_code: i64,
    command: String,
}

/// Parse one spool line. Returns `None` for blank, oversized or malformed lines
/// (a hook writing a control character it did not escape, a torn write) so a bad
/// line never stalls the drain.
fn parse_spool_line(line: &str, now_micros: u64) -> Option<SpooledCommand> {
    let line = line.trim_start_matches('\u{feff}').trim();
    if line.is_empty() || line.len() > MAX_LINE_BYTES {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let command = value.get("cmd")?.as_str()?.trim();
    if command.is_empty() {
        return None;
    }
    let exit_code = value
        .get("exit_code")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    let ts = value
        .get("ts")
        .and_then(serde_json::Value::as_u64)
        .filter(|ts| {
            *ts >= MIN_PLAUSIBLE_TS_MICROS
                && *ts <= now_micros.saturating_add(MAX_FUTURE_SKEW_MICROS)
        })
        .unwrap_or(now_micros);
    Some(SpooledCommand {
        ts,
        exit_code,
        command: super::truncate(command, MAX_COMMAND_CHARS),
    })
}

/// Return the files earlier passes took aside (oldest first), then take the live
/// spool aside for the next pass. Files taken aside now are not returned: a hook
/// that opened the live file just before the rename gets one interval to finish.
fn take_spool_files(dir: &Path) -> Vec<PathBuf> {
    let mut ready: Vec<PathBuf> = fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(DRAINING_SUFFIX))
                })
                .collect()
        })
        .unwrap_or_default();
    ready.sort();

    let live = dir.join(SPOOL_FILE);
    if live.exists() {
        let aside = dir.join(format!(
            "{SPOOL_FILE}.{:020}-{}{DRAINING_SUFFIX}",
            super::now_micros(),
            uuid::Uuid::new_v4().simple()
        ));
        if let Err(e) = fs::rename(&live, &aside) {
            tracing::trace!("Terminal capture: spool busy ({e}); retrying next pass");
        }
    }
    ready
}

/// Read up to [`MAX_LINE_BYTES`] of the next line into `buf`, consuming the rest
/// of an overlong line without buffering it. `Ok(None)` at end of input,
/// `Ok(Some(false))` for a line that was too long (and skipped).
fn read_bounded_line(
    reader: &mut impl BufRead,
    buf: &mut Vec<u8>,
) -> std::io::Result<Option<bool>> {
    buf.clear();
    let mut overlong = false;
    let mut read_any = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(read_any.then_some(!overlong));
        }
        read_any = true;
        let (chunk, done) = match available.iter().position(|&b| b == b'\n') {
            Some(i) => (&available[..=i], true),
            None => (available, false),
        };
        let used = chunk.len();
        if !overlong {
            if buf.len() + used > MAX_LINE_BYTES + 1 {
                overlong = true;
                buf.clear();
            } else {
                buf.extend_from_slice(chunk);
            }
        }
        reader.consume(used);
        if done {
            return Ok(Some(!overlong));
        }
    }
}

/// Read one taken-aside spool file. `None` when it could not be read completely;
/// the file is then kept for the next pass rather than half-drained.
fn read_spool_file(path: &Path, now_micros: u64) -> Option<Vec<SpooledCommand>> {
    let file = fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file.take(MAX_SPOOL_FILE_BYTES));
    let mut buf = Vec::new();
    let mut commands = Vec::new();
    loop {
        match read_bounded_line(&mut reader, &mut buf) {
            Ok(None) => return Some(commands),
            Ok(Some(false)) => {}
            Ok(Some(true)) => {
                if let Some(command) = parse_spool_line(&String::from_utf8_lossy(&buf), now_micros)
                {
                    commands.push(command);
                }
            }
            Err(e) => {
                tracing::warn!("Terminal capture: could not read {}: {e}", path.display());
                return None;
            }
        }
    }
}

/// Remove a drained file, or failing that empty it, so its commands can never be
/// emitted again. `false` when neither worked.
fn retire_spool_file(path: &Path) -> bool {
    if fs::remove_file(path).is_ok() {
        return true;
    }
    match fs::OpenOptions::new().write(true).truncate(true).open(path) {
        Ok(_) => true,
        Err(e) => {
            tracing::warn!(
                "Terminal capture: could not remove or empty drained spool {}: {e}",
                path.display()
            );
            false
        }
    }
}

/// Drain the spool files earlier passes took aside, handing each command to
/// `sink`, then take the live spool aside for the next pass. Returns how many
/// commands were emitted.
fn drain_spool(dir: &Path, now_micros: u64, mut sink: impl FnMut(SpooledCommand)) -> usize {
    let mut drained = 0;
    for path in take_spool_files(dir) {
        let Some(commands) = read_spool_file(&path, now_micros) else {
            continue;
        };
        // Retire before emitting: a file that cannot be retired is skipped, so
        // it is never emitted again on every pass.
        if !retire_spool_file(&path) {
            continue;
        }
        drained += commands.len();
        commands.into_iter().for_each(&mut sink);
    }
    drained
}

fn emit_command(command: SpooledCommand) {
    super::emit_at(
        command.ts,
        super::TRACK_TERMINAL,
        "shell_command",
        "Terminal",
        &command.command,
        vec![("exit_code", rmpv::Value::from(command.exit_code))],
    );
}

/// Write the shell hook scripts into `<data_dir>/shell-hooks/` (idempotent), log
/// how to enable terminal capture, and start draining the hooks' spool. Sourcing
/// a hook is still the user's opt-in; without it the spool stays empty.
pub fn start(data_dir: &Path) {
    let dir = data_dir.join("shell-hooks");
    let spool = spool_dir(data_dir);
    if let Err(e) = fs::create_dir_all(&dir).and_then(|()| create_private_dir(&spool)) {
        tracing::warn!("Terminal capture disabled: could not create a private spool dir: {e}");
        return;
    }
    let (bash_hook, pwsh_hook) = render_hooks(&spool.join(SPOOL_FILE), &spool.join(MARKER_FILE));
    let bash = dir.join("shadow-hook.sh");
    let pwsh = dir.join("shadow-hook.ps1");
    let _ = fs::write(&bash, bash_hook);
    let _ = fs::write(&pwsh, pwsh_hook);
    tracing::info!(
        "Terminal capture ready (opt-in). Enable by sourcing:\n  bash/zsh: source {}\n  PowerShell: . {}",
        bash.display(),
        pwsh.display()
    );

    *ACTIVE_SPOOL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(spool.clone());
    let spawned = std::thread::Builder::new()
        .name("shadow-terminal".into())
        .spawn(move || {
            // Rewritten every pass: a second instance that overwrote the marker
            // and then exited (its port bind failed) loses it again within one
            // interval, so capture does not stop silently.
            while refresh_marker(&spool) {
                // Drained while capture is paused too: `emit` drops paused
                // events, so commands run during a pause are discarded, not
                // replayed later.
                drain_spool(&spool, super::now_micros(), emit_command);
                std::thread::sleep(DRAIN_INTERVAL);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!("Terminal capture: could not start the spool reader: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000_000_000;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("shadow-terminal-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn hooks() -> (String, String) {
        render_hooks(
            Path::new("/data/shell-hooks/spool/commands.jsonl"),
            Path::new("/data/shell-hooks/spool/.drainer"),
        )
    }

    /// Drain twice: the first pass takes the live spool aside, the second reads it.
    fn drain_all(dir: &Path, now: u64) -> Vec<SpooledCommand> {
        let mut seen = Vec::new();
        drain_spool(dir, now, |command| seen.push(command));
        drain_spool(dir, now, |command| seen.push(command));
        seen
    }

    #[test]
    fn hooks_never_read_the_api_token_or_use_the_network() {
        let (bash, pwsh) = hooks();
        for hook in [&bash, &pwsh] {
            assert!(!hook.contains("api-token"), "{hook}");
            assert!(!hook.contains("SHADOW_API_TOKEN"), "{hook}");
            assert!(!hook.contains("Authorization"), "{hook}");
            assert!(!hook.contains("127.0.0.1"), "{hook}");
            assert!(!hook.contains("curl"), "{hook}");
            assert!(!hook.contains("__SHADOW_TERMINAL"), "{hook}");
            assert!(
                hook.contains("/data/shell-hooks/spool/commands.jsonl"),
                "{hook}"
            );
            assert!(hook.contains("/data/shell-hooks/spool/.drainer"), "{hook}");
        }
        assert!(bash.contains("umask 077"));
        assert!(!pwsh.contains("Start-Job"));
    }

    #[test]
    fn hook_paths_with_quotes_stay_inside_the_literal() {
        let (bash, pwsh) = render_hooks(
            Path::new("/home/o'neil/spool/commands.jsonl"),
            Path::new("/home/o\u{2019}neil/spool/.drainer"),
        );
        assert!(bash.contains(r"'/home/o'\''neil/spool/commands.jsonl'"));
        assert!(pwsh.contains("'/home/o''neil/spool/commands.jsonl'"));
        assert!(pwsh.contains("'/home/o\u{2019}\u{2019}neil/spool/.drainer'"));
    }

    #[test]
    fn parses_a_hook_line() {
        let line = format!(r#"{{"ts":{NOW},"exit_code":2,"cmd":"cargo test \"a b\""}}"#);
        assert_eq!(
            parse_spool_line(&line, NOW),
            Some(SpooledCommand {
                ts: NOW,
                exit_code: 2,
                command: "cargo test \"a b\"".into()
            })
        );
    }

    #[test]
    fn rejects_blank_malformed_oversized_and_empty_commands() {
        assert_eq!(parse_spool_line("", NOW), None);
        assert_eq!(parse_spool_line("{not json", NOW), None);
        assert_eq!(
            parse_spool_line(r#"{"ts":1,"exit_code":0,"cmd":"  "}"#, NOW),
            None
        );
        assert_eq!(parse_spool_line(r#"{"ts":1,"exit_code":0}"#, NOW), None);
        let huge = format!(r#"{{"cmd":"{}"}}"#, "x".repeat(MAX_LINE_BYTES));
        assert_eq!(parse_spool_line(&huge, NOW), None);
    }

    #[test]
    fn keeps_past_timestamps_and_clamps_future_or_implausible_ones() {
        let day_old = NOW - 24 * 60 * 60 * 1_000_000;
        let line = |ts: u64| format!(r#"{{"ts":{ts},"exit_code":0,"cmd":"ls"}}"#);
        assert_eq!(parse_spool_line(&line(day_old), NOW).unwrap().ts, day_old);
        assert_eq!(
            parse_spool_line(&line(NOW + 3_600_000_000), NOW)
                .unwrap()
                .ts,
            NOW
        );
        assert_eq!(parse_spool_line(&line(5), NOW).unwrap().ts, NOW);
        let bom = format!("\u{feff}{{\"ts\":{NOW},\"exit_code\":1,\"cmd\":\"ls\"}}\r\n");
        let parsed = parse_spool_line(&bom, NOW).unwrap();
        assert_eq!((parsed.ts, parsed.exit_code), (NOW, 1));
    }

    #[test]
    fn truncates_long_commands() {
        let line = format!(
            r#"{{"ts":{NOW},"cmd":"{}"}}"#,
            "y".repeat(MAX_COMMAND_CHARS + 50)
        );
        let parsed = parse_spool_line(&line, NOW).unwrap();
        assert_eq!(parsed.command.chars().count(), MAX_COMMAND_CHARS + 1);
        assert!(parsed.command.ends_with('…'));
    }

    #[test]
    fn skips_an_overlong_line_without_losing_the_next_one() {
        let dir = temp_dir();
        let long = format!(
            r#"{{"ts":{NOW},"cmd":"{}"}}"#,
            "z".repeat(MAX_LINE_BYTES * 3)
        );
        fs::write(
            dir.join(SPOOL_FILE),
            format!("{long}\n{{\"ts\":{NOW},\"exit_code\":0,\"cmd\":\"after\"}}\n"),
        )
        .unwrap();
        let seen: Vec<String> = drain_all(&dir, NOW)
            .into_iter()
            .map(|c| c.command)
            .collect();
        assert_eq!(seen, ["after"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn drains_the_live_spool_on_the_next_pass_and_leftovers_now() {
        let dir = temp_dir();
        fs::write(
            dir.join(SPOOL_FILE),
            format!(
                "{{\"ts\":{NOW},\"exit_code\":0,\"cmd\":\"one\"}}\nnot json\n{{\"ts\":{NOW},\"exit_code\":1,\"cmd\":\"two\"}}"
            ),
        )
        .unwrap();
        // A file an interrupted pass took aside but never finished.
        fs::write(
            dir.join(format!("{SPOOL_FILE}.1-crashed{DRAINING_SUFFIX}")),
            format!("{{\"ts\":{NOW},\"exit_code\":0,\"cmd\":\"zero\"}}\n"),
        )
        .unwrap();

        let mut first = Vec::new();
        assert_eq!(drain_spool(&dir, NOW, |c| first.push(c.command)), 1);
        assert_eq!(first, ["zero"], "the live spool waits one pass");
        assert!(
            !dir.join(SPOOL_FILE).exists(),
            "the live spool is taken aside"
        );

        let mut second = Vec::new();
        assert_eq!(drain_spool(&dir, NOW, |c| second.push(c.command)), 2);
        assert_eq!(
            second,
            ["one", "two"],
            "a final line without a newline counts"
        );
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            0,
            "drained files are removed"
        );
        assert_eq!(drain_spool(&dir, NOW, |_| panic!("nothing to drain")), 0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn publishes_the_drainer_pid() {
        let dir = temp_dir();
        write_marker(&dir).unwrap();
        let pid = fs::read_to_string(dir.join(MARKER_FILE)).unwrap();
        assert_eq!(pid.trim(), std::process::id().to_string());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shutdown_removes_only_this_processs_marker() {
        let dir = temp_dir();
        write_marker(&dir).unwrap();
        remove_marker_if_ours(&dir);
        assert!(!dir.join(MARKER_FILE).exists(), "own marker is removed");

        fs::write(dir.join(MARKER_FILE), "999999999\n").unwrap();
        remove_marker_if_ours(&dir);
        assert!(
            dir.join(MARKER_FILE).exists(),
            "another instance's marker is kept"
        );

        // The one test that touches the process-wide drainer state: a refresh
        // rewrites a lost marker, and none runs after shutdown.
        *ACTIVE_SPOOL.lock().unwrap() = Some(dir.clone());
        assert!(refresh_marker(&dir));
        assert_eq!(
            fs::read_to_string(dir.join(MARKER_FILE)).unwrap().trim(),
            std::process::id().to_string(),
            "a surviving drainer reclaims an overwritten marker"
        );
        shutdown();
        assert!(!dir.join(MARKER_FILE).exists());
        assert!(!refresh_marker(&dir), "no refresh after shutdown");
        assert!(!dir.join(MARKER_FILE).exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn powershell_hook_honors_history_privacy() {
        let (_, pwsh) = hooks();
        assert!(pwsh.contains("StartsWith(' ')"), "{pwsh}");
        assert!(pwsh.contains("AddToHistoryHandler"), "{pwsh}");
        assert!(
            pwsh.contains("'False', 'SkipAdding', 'MemoryOnly'"),
            "{pwsh}"
        );
        assert!(pwsh.contains("catch { return $false }"), "{pwsh}");
        assert!(pwsh.contains("__shadowKeep $entry.CommandLine"), "{pwsh}");
    }

    #[cfg(unix)]
    #[test]
    fn spool_dir_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = temp_dir();
        let spool = spool_dir(&dir);
        create_private_dir(&spool).unwrap();
        let mode = fs::metadata(&spool).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(unix)]
    mod shells {
        use super::*;
        use std::process::{Command, Stdio};

        struct Fixture {
            dir: PathBuf,
            spool: PathBuf,
            hook: PathBuf,
        }

        fn fixture(live_drainer: bool) -> Fixture {
            let dir = temp_dir();
            let spool = spool_dir(&dir);
            create_private_dir(&spool).unwrap();
            if live_drainer {
                write_marker(&spool).unwrap();
            }
            let (bash_hook, _) = render_hooks(&spool.join(SPOOL_FILE), &spool.join(MARKER_FILE));
            let hook = dir.join("shadow-hook.sh");
            fs::write(&hook, bash_hook).unwrap();
            Fixture { dir, spool, hook }
        }

        /// Run `lines` in an interactive `shell` that sourced the hook, then
        /// return what the spool recorded.
        fn run(
            fixture: &Fixture,
            shell: &str,
            args: &[&str],
            lines: &str,
        ) -> Option<Vec<(String, i64)>> {
            let mut child = Command::new(shell)
                .args(args)
                .env("HISTFILE", "/dev/null")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let script = format!("source '{}'\n{lines}\nexit\n", fixture.hook.display());
            std::io::Write::write_all(child.stdin.as_mut().unwrap(), script.as_bytes()).unwrap();
            drop(child.stdin.take());
            child.wait().unwrap();
            let commands = drain_all(&fixture.spool, crate::capture::passive::now_micros());
            Some(
                commands
                    .into_iter()
                    .map(|c| (c.command, c.exit_code))
                    .collect(),
            )
        }

        fn cleanup(fixture: Fixture) {
            fs::remove_dir_all(&fixture.dir).unwrap();
        }

        #[test]
        fn bash_records_each_command_once_privately() {
            let fixture = fixture(true);
            let Some(seen) = run(
                &fixture,
                "bash",
                &["--norc", "-i"],
                "HISTCONTROL=ignorespace\nls /nonexistent-shadow-test\n\necho \"a b \\\\ c\"\n echo SECRET=abc",
            ) else {
                return cleanup(fixture);
            };
            assert_eq!(
                seen,
                [
                    ("HISTCONTROL=ignorespace".to_owned(), 0),
                    ("ls /nonexistent-shadow-test".to_owned(), 2),
                    ("echo \"a b \\\\ c\"".to_owned(), 0),
                ],
                "the empty prompt and the space-prefixed command are not recorded"
            );
            cleanup(fixture);
        }

        /// Run `lines` in an interactive bash that sources the hook from its rc
        /// file (as `~/.bashrc` would) with a pre-filled `HISTFILE`.
        fn run_from_rc(
            fixture: &Fixture,
            histfile_lines: &str,
            lines: &str,
        ) -> Option<Vec<(String, i64)>> {
            let histfile = fixture.dir.join("history");
            fs::write(&histfile, histfile_lines).unwrap();
            let rc = fixture.dir.join("rc");
            fs::write(&rc, format!("source '{}'\n", fixture.hook.display())).unwrap();
            let mut child = Command::new("bash")
                .arg("--rcfile")
                .arg(&rc)
                .arg("-i")
                .env("HISTFILE", &histfile)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let script = format!("{lines}\nexit\n");
            std::io::Write::write_all(child.stdin.as_mut().unwrap(), script.as_bytes()).unwrap();
            drop(child.stdin.take());
            child.wait().unwrap();
            let commands = drain_all(&fixture.spool, crate::capture::passive::now_micros());
            Some(
                commands
                    .into_iter()
                    .map(|c| (c.command, c.exit_code))
                    .collect(),
            )
        }

        #[test]
        fn bash_does_not_record_the_previous_sessions_last_history_line() {
            let fixture = fixture(true);
            let Some(seen) =
                run_from_rc(&fixture, "old-secret-cmd one\nold-cmd two\n", "true\nfalse")
            else {
                return cleanup(fixture);
            };
            assert_eq!(
                seen,
                [("true".to_owned(), 0), ("false".to_owned(), 1)],
                "only this session's commands reach the spool"
            );
            cleanup(fixture);
        }

        #[test]
        fn bash_records_repeats_that_ignoredups_and_erasedups_fold() {
            let fixture = fixture(true);
            let Some(seen) = run_from_rc(
                &fixture,
                "",
                "HISTCONTROL=ignoredups\nfalse\ntrue\ntrue\nHISTCONTROL=erasedups\nls /nonexistent-shadow-a\npwd\nls /nonexistent-shadow-a",
            ) else {
                return cleanup(fixture);
            };
            let commands: Vec<&str> = seen.iter().map(|(c, _)| c.as_str()).collect();
            assert_eq!(
                commands,
                [
                    "HISTCONTROL=ignoredups",
                    "false",
                    "true",
                    "true",
                    "HISTCONTROL=erasedups",
                    "ls /nonexistent-shadow-a",
                    "pwd",
                    "ls /nonexistent-shadow-a",
                ]
            );
            assert_eq!(seen[2].1, 0);
            assert_eq!(seen[3].1, 0);
            cleanup(fixture);
        }

        #[test]
        fn bash_ignoreboth_still_skips_space_prefixed_commands() {
            let fixture = fixture(true);
            let Some(seen) = run_from_rc(
                &fixture,
                "",
                "HISTCONTROL=ignoreboth\ntrue\n echo SECRET=abc\ntrue",
            ) else {
                return cleanup(fixture);
            };
            let commands: Vec<&str> = seen.iter().map(|(c, _)| c.as_str()).collect();
            assert_eq!(commands, ["HISTCONTROL=ignoreboth", "true"]);
            cleanup(fixture);
        }

        #[test]
        fn zsh_honors_hist_ignore_space() {
            let fixture = fixture(true);
            let Some(seen) = run(
                &fixture,
                "zsh",
                &["-f", "-i"],
                "setopt HIST_IGNORE_SPACE\nfalse\n echo SECRET=abc\n\ntrue",
            ) else {
                return cleanup(fixture);
            };
            assert_eq!(
                seen,
                [
                    ("setopt HIST_IGNORE_SPACE".to_owned(), 0),
                    ("false".to_owned(), 1),
                    ("true".to_owned(), 0),
                ]
            );
            cleanup(fixture);
        }

        #[test]
        fn nothing_is_spooled_without_a_live_drainer() {
            let fixture = fixture(false);
            if let Some(seen) = run(&fixture, "bash", &["--norc", "-i"], "true") {
                assert!(seen.is_empty());
                assert!(!fixture.spool.join(SPOOL_FILE).exists());
            }
            cleanup(fixture);
        }

        #[test]
        fn a_missing_spool_dir_does_not_kill_set_e_shells() {
            let fixture = fixture(true);
            for (shell, args) in [("bash", &["--norc", "-i"][..]), ("zsh", &["-f", "-i"][..])] {
                let Ok(mut child) = Command::new(shell)
                    .args(args)
                    .env("HISTFILE", "/dev/null")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                else {
                    continue;
                };
                let script = format!(
                    "source '{}'\nset -e\nrm -rf '{}'\ntrue\necho still-alive\nexit\n",
                    fixture.hook.display(),
                    fixture.spool.display()
                );
                std::io::Write::write_all(child.stdin.as_mut().unwrap(), script.as_bytes())
                    .unwrap();
                drop(child.stdin.take());
                let output = child.wait_with_output().unwrap();
                assert!(
                    String::from_utf8_lossy(&output.stdout).contains("still-alive"),
                    "{shell} exited at the prompt"
                );
                create_private_dir(&fixture.spool).unwrap();
                write_marker(&fixture.spool).unwrap();
            }
            cleanup(fixture);
        }
    }
}
