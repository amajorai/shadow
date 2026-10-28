//! Sealed text columns for Shadow's side databases (`indices/context.db`,
//! `summaries.db`, `memory.db`, `proactive.db`, `procedures.db`).
//!
//! Each database is its own content store (`shadow-context`, ...) derived from
//! Shadow's component key (`shadow_core::seal`). Content columns are sealed as
//! `encb:v1:`, bound to `<table>.<column>` and the row's TEXT id; ids, times,
//! app names, categories and counters stay clear. Opening a database:
//!
//! 1. turns on `secure_delete` (freed cells are zeroed),
//! 2. checks the store's recorded key (a store sealed under another key flips
//!    Shadow to `sealing: mismatch` and writes are refused),
//! 3. seals any legacy plaintext in place (resumable, idempotent, never drops a
//!    value before its sealed form is written), then scrubs freed pages.
//!
//! Without a key the database still opens: legacy plaintext reads, new writes
//! are refused, except in compat mode (`shadow_core::seal`), where they are
//! stored in plaintext as before sealing and sealed at the first start with a
//! key.

use std::path::Path;

use anyhow::{Context, Result};
use ryu_crypto::content_sqlite::{
    check_store_key, enable_secure_delete, finish_store, migrate_columns, ColumnPlan, KeyKind,
    StoreKeyState,
};

/// Open `db_path` for `store` and seal the legacy plaintext of `columns`
/// (`(table, column)`, each keyed by a TEXT `id` column). Call it AFTER the
/// tables exist (pass a `schema` batch).
pub fn open(
    db_path: &Path,
    store: &str,
    schema: &str,
    columns: &[(&str, &str)],
) -> Result<rusqlite::Connection> {
    #[cfg(test)]
    test_support::key();
    if let Some(parent) = db_path.parent() {
        shadow_core::seal::create_private_dir(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut conn = rusqlite::Connection::open(db_path)?;
    enable_secure_delete(&conn)?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    conn.execute_batch(schema)?;
    if shadow_core::seal::kcv().is_none() {
        return Ok(conn);
    }
    let cipher = shadow_core::seal::content(store)?;
    if let StoreKeyState::Mismatch { .. } = check_store_key(&conn, &cipher)? {
        shadow_core::seal::mark_mismatch(store);
        return Ok(conn);
    }
    if shadow_core::seal::require_writable().is_err() {
        return Ok(conn);
    }
    let plans = columns
        .iter()
        .map(|(table, column)| ColumnPlan::text(table, column, "id", KeyKind::Text))
        .collect::<Result<Vec<_>>>()?;
    match migrate_columns(&mut conn, &cipher, &plans) {
        Ok(report) => {
            if report.sealed > 0 {
                tracing::info!(store, sealed = report.sealed, "sealed legacy Shadow rows");
                if let Err(error) = finish_store(&conn, db_path, store) {
                    tracing::warn!(store, "freed-page scrub deferred: {error}");
                }
            }
        }
        Err(error) => tracing::warn!(
            store,
            "sealing legacy rows deferred to the next start: {error}"
        ),
    }
    Ok(conn)
}

/// Seal one value of `table.column` for the row `id` (refused without a key).
pub fn seal(
    store: &str,
    table: &str,
    column: &str,
    id: &str,
    value: &str,
) -> rusqlite::Result<String> {
    shadow_core::seal::seal_value(store, &format!("{table}.{column}"), "id", id, value)
        .map_err(shadow_core::seal::to_sql_error)
}

/// Open one stored value (legacy plaintext as is). A value that does not open
/// is counted and returned as `None`, so a list skips it instead of failing.
pub fn open_value(
    store: &str,
    table: &str,
    column: &str,
    id: &str,
    stored: String,
) -> Option<String> {
    shadow_core::seal::open_opt(store, &format!("{table}.{column}"), "id", id, Some(stored))
}

/// [`seal`] for a nullable column.
pub fn seal_optional(
    store: &str,
    table: &str,
    column: &str,
    id: &str,
    value: Option<&str>,
) -> rusqlite::Result<Option<String>> {
    value
        .map(|value| seal(store, table, column, id, value))
        .transpose()
}

/// Open a nullable column: `Ok(None)` for NULL, `Err(())` when a stored value
/// does not open (the caller skips the row).
pub fn open_optional(
    store: &str,
    table: &str,
    column: &str,
    id: &str,
    stored: Option<String>,
) -> std::result::Result<Option<String>, ()> {
    match stored {
        None => Ok(None),
        Some(stored) => open_value(store, table, column, id, stored)
            .map(Some)
            .ok_or(()),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    /// Install the fixed test key (shared by every test in this binary).
    pub fn key() {
        shadow_core::seal::install_fixed_key_for_tests();
    }

    /// Whether the database file (or its WAL) holds `needle` in plaintext.
    pub fn db_contains(db_path: &std::path::Path, needle: &str) -> bool {
        let mut wal = db_path.as_os_str().to_owned();
        wal.push("-wal");
        [db_path.to_path_buf(), std::path::PathBuf::from(wal)]
            .iter()
            .filter_map(|path| std::fs::read(path).ok())
            .any(|bytes| {
                bytes
                    .windows(needle.len())
                    .any(|window| window == needle.as_bytes())
            })
    }
}
