//! The loopback API token: minting, reading and the bearer check.

use super::*;

// ─── API token (loopback auth) ────────────────────────────────────────────────
//
// Shadow is loopback-bound, but "loopback" is not an auth boundary: any local
// process — or any web page via a no-preflight `text/plain` POST or DNS
// rebinding — can reach it. Everything except `/health` therefore requires a
// shared-secret bearer (mirroring the `RYU_EXT_TOKEN` gate the apps-store
// sidecars use, e.g. `apps-store/quests/backend/src/main.rs`): without it a
// hostile page could exfiltrate full screen history (`/search`, `/timeline`,
// `/frame`, …), silently flip `/capture/control` to re-enable capture behind
// the user's back, or poison the timeline via `/ingest`.

/// Name of the persisted token file under the Shadow data dir.
pub(super) const API_TOKEN_FILE: &str = "api-token";

/// Resolve the API token: `SHADOW_API_TOKEN` env if set (Core injects it at
/// spawn; operators may export their own), else the persisted
/// `<data_dir>/api-token` file, generated on first run with owner-only
/// permissions so any same-user local client (Core, sidecars) can read it while
/// web pages and other users cannot. Returns `None` only when the file can
/// neither be read nor created — the auth gate then FAILS CLOSED (rejects all).
pub(super) fn resolve_api_token(data_dir: &std::path::Path) -> Option<String> {
    if let Ok(env_token) = std::env::var("SHADOW_API_TOKEN") {
        let trimmed = env_token.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
    }

    let path = data_dir.join(API_TOKEN_FILE);
    if let Some(existing) = read_api_token_file(&path) {
        return Some(existing);
    }

    // First run: mint a random token (2× UUIDv4 = 64 hex chars from the OS
    // CSPRNG) and persist it owner-only.
    let token = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );
    if let Err(e) = std::fs::create_dir_all(data_dir) {
        tracing::error!("cannot create data dir for API token file: {e}");
        return None;
    }
    // Created owner-only and exclusively: the token is never observable at a
    // permissive umask, and a concurrent minter (Core's
    // `ensure_api_token`) that won the race keeps its value. The file stays
    // plaintext because the opt-in shell hooks `cat` it (see
    // `capture/passive/terminal.rs`).
    let created = match create_api_token_file(&path, &token) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // A concurrent minter won: use its token.
            if let Some(existing) = read_api_token_file(&path) {
                return Some(existing);
            }
            // A blank/unreadable leftover. Never write the token into it — its
            // mode may be loose and a handle opened earlier would see the
            // token — replace it with a fresh owner-only file instead.
            std::fs::remove_file(&path).and_then(|()| create_api_token_file(&path, &token))
        }
        other => other,
    };
    if let Err(e) = created {
        tracing::error!("cannot persist API token to {}: {e}", path.display());
        return None;
    }
    Some(token)
}

/// Non-empty trimmed contents of the persisted token file, if any.
pub(super) fn read_api_token_file(path: &std::path::Path) -> Option<String> {
    let existing = std::fs::read_to_string(path).ok()?;
    let trimmed = existing.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// Create the token file and write `token`. The file is created exclusively
/// (`create_new`, which also refuses to follow a symlink) and owner-only in the
/// same `open(2)`: creating it under the default umask and `chmod`ing it
/// afterwards would let another local user open it in between and keep reading.
pub(super) fn create_api_token_file(path: &std::path::Path, token: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(path)?;
    let written = file
        .write_all(token.as_bytes())
        .and_then(|()| file.sync_all());
    if written.is_err() {
        // Never leave a partial token behind for the next run to adopt.
        drop(file);
        let _ = std::fs::remove_file(path);
    }
    written
}

/// Shared-secret bearer + anti-CSRF gate for everything except `/health`.
///
/// Requests carrying an `Origin` header are rejected outright (browsers always
/// attach it to cross-origin requests, so this is the CSRF/DNS-rebinding
/// kill-switch — no browser context may drive this API, token or not). Native
/// loopback clients present `Authorization: Bearer <token>`.
///
/// **Fail-closed:** `expected == None`/empty (token could not be resolved)
/// rejects every request rather than falling open.
pub(super) async fn require_api_token(
    req: Request,
    next: Next,
    expected: Option<&str>,
) -> Response {
    if req.headers().contains_key(ORIGIN) {
        return (
            StatusCode::FORBIDDEN,
            "cross-origin (browser) requests are not allowed",
        )
            .into_response();
    }
    let provided = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if bearer_ok(provided, expected) {
        next.run(req).await
    } else {
        (StatusCode::UNAUTHORIZED, "unauthorized").into_response()
    }
}

/// Pure bearer check (factored out so the auth decision is unit-testable without
/// an axum `Request`/`Next`). Returns `true` only when `expected` is a non-empty
/// token AND `provided` equals it (constant-time compared). A `None`/empty
/// `expected` is the fail-closed case → always `false`.
pub(super) fn bearer_ok(provided: Option<&str>, expected: Option<&str>) -> bool {
    let Some(expected) = expected.filter(|t| !t.is_empty()) else {
        return false;
    };
    ct_eq(provided.unwrap_or("").as_bytes(), expected.as_bytes())
}

/// Constant-time byte comparison — no early return on the first mismatched byte,
/// so the token check does not leak length/prefix via timing.
pub(super) fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
