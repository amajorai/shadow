//! Terminal capture — records shell commands and their exit codes.
//!
//! Unlike the other sources there is no reliable, portable way to observe another
//! process's shell from outside it, so capture is opt-in via a shell hook: this
//! module writes ready-to-source hook scripts into the data dir on startup. Each
//! hook POSTs the just-run command + exit code to the existing `/ingest` endpoint
//! on track 9 (`shell_command`). The user enables it once by sourcing the snippet
//! from their shell rc file; the path is logged on startup.
//!
//! `/ingest` is bearer-gated like every non-`/health` route (see `server.rs`), so
//! the emitted hooks resolve the shared-secret token at POST time: the
//! `SHADOW_API_TOKEN` env var when the shell has it exported, else the persisted
//! `<data_dir>/api-token` file (whose absolute path is baked into the script at
//! write time). Reading at POST time — not baking the secret into the script —
//! keeps the hook valid across token rotation and avoids a second on-disk copy.

use std::path::Path;

/// Placeholder substituted with the absolute `<data_dir>/api-token` path when
/// the hook scripts are written.
const TOKEN_FILE_PLACEHOLDER: &str = "__SHADOW_API_TOKEN_FILE__";

const BASH_HOOK: &str = r#"# Shadow terminal capture (bash/zsh). Source this from ~/.bashrc or ~/.zshrc.
# Records each command + exit code to the Shadow sidecar on http://127.0.0.1:3030.
__shadow_report() {
  local code=$?
  local cmd
  cmd=$(history 1 2>/dev/null | sed 's/^ *[0-9]* *//')
  [ -z "$cmd" ] && return $code
  local ts=$(( $(date +%s) * 1000000 ))
  local esc
  esc=$(printf '%s' "$cmd" | sed 's/\\/\\\\/g; s/"/\\"/g')
  # Shadow's /ingest requires the shared-secret bearer: env override, else the
  # token file Shadow persists in its data dir.
  local tok="${SHADOW_API_TOKEN:-$(cat '__SHADOW_API_TOKEN_FILE__' 2>/dev/null)}"
  curl -s -m 1 -X POST http://127.0.0.1:3030/ingest \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer $tok" \
    -d "{\"events\":[{\"ts\":$ts,\"v\":2,\"track\":9,\"type\":\"shell_command\",\"app_name\":\"Terminal\",\"window_title\":\"$esc\",\"exit_code\":$code}]}" \
    >/dev/null 2>&1 &
  return $code
}
if [ -n "$ZSH_VERSION" ]; then
  precmd_functions+=(__shadow_report)
elif [ -n "$BASH_VERSION" ]; then
  case "$PROMPT_COMMAND" in
    *__shadow_report*) ;;
    *) PROMPT_COMMAND="__shadow_report;${PROMPT_COMMAND}" ;;
  esac
fi
"#;

const PWSH_HOOK: &str = r#"# Shadow terminal capture (PowerShell). Dot-source this from $PROFILE.
# Records each command + exit code to the Shadow sidecar on http://127.0.0.1:3030.
function global:prompt {
  $code = if ($?) { 0 } else { 1 }
  $last = (Get-History -Count 1).CommandLine
  if ($last) {
    $ts = [int64]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) * 1000
    $body = @{ events = @(@{ ts = $ts; v = 2; track = 9; type = 'shell_command'
      app_name = 'Terminal'; window_title = $last; exit_code = $code }) } | ConvertTo-Json -Depth 5 -Compress
    Start-Job -ScriptBlock {
      param($b)
      # Shadow's /ingest requires the shared-secret bearer: env override, else
      # the token file Shadow persists in its data dir.
      $tok = $env:SHADOW_API_TOKEN
      if (-not $tok) { try { $tok = (Get-Content -Raw '__SHADOW_API_TOKEN_FILE__').Trim() } catch {} }
      try { Invoke-RestMethod -Uri 'http://127.0.0.1:3030/ingest' -Method Post -Body $b -ContentType 'application/json' -Headers @{ Authorization = "Bearer $tok" } -TimeoutSec 1 } catch {}
    } -ArgumentList $body | Out-Null
  }
  "PS $($executionContext.SessionState.Path.CurrentLocation)$('>' * ($nestedPromptLevel + 1)) "
}
"#;

/// Write the shell hook scripts into `<data_dir>/shell-hooks/` (idempotent) and log
/// how to enable terminal capture. This does not capture anything by itself.
pub fn start(data_dir: &Path) {
    let dir = data_dir.join("shell-hooks");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!("Terminal capture: could not create hook dir: {e}");
        return;
    }
    // Bake the absolute api-token path into each hook so a sourced shell can
    // present the bearer without knowing the Shadow data dir.
    let token_file = data_dir.join("api-token");
    let token_file = token_file.to_string_lossy();
    let bash = dir.join("shadow-hook.sh");
    let pwsh = dir.join("shadow-hook.ps1");
    let _ = std::fs::write(&bash, BASH_HOOK.replace(TOKEN_FILE_PLACEHOLDER, &token_file));
    let _ = std::fs::write(&pwsh, PWSH_HOOK.replace(TOKEN_FILE_PLACEHOLDER, &token_file));
    tracing::info!(
        "Terminal capture ready (opt-in). Enable by sourcing:\n  bash/zsh: source {}\n  PowerShell: . {}",
        bash.display(),
        pwsh.display()
    );
}
