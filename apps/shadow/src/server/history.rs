//! Search, timeline, journal, frame and recent-activity handlers.

use super::*;

// ─── Search ────────────────────────────────────────────────────────────────────

pub(super) async fn search_handler(Query(query): Query<SearchQuery>) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(20);
    match shadow_core::search_text(query.q, limit) {
        Ok(results) => {
            let count = results.len();
            Json(json!({ "results": results, "count": count }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

pub(super) async fn semantic_search_handler(
    State(state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(5);
    match shadow_core::vector_search(query.q, limit) {
        Ok(results) => Json(json!({ "results": results })),
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

// ─── Timeline ─────────────────────────────────────────────────────────────────

pub(super) async fn timeline_handler(Query(query): Query<TimelineQuery>) -> impl IntoResponse {
    match shadow_core::query_time_range(query.start, query.end) {
        Ok(entries) => {
            let count = entries.len();
            Json(json!({ "entries": entries, "count": count }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

pub(super) async fn journal_handler(
    State(state): State<AppState>,
    Query(query): Query<JournalQuery>,
) -> impl IntoResponse {
    match shadow_core::query_journal_snapshot(query.start, query.end) {
        Ok(mut snapshot) => {
            if query.narrate {
                if let Some(orchestrator) = &state.orchestrator {
                    snapshot.cards = crate::intelligence::journal_narrator::narrate_cards(
                        orchestrator,
                        snapshot.cards.clone(),
                    )
                    .await;
                    // Card ranges/categories are preserved by the narrator; only
                    // the standup text needs refreshing to use narrated titles.
                    shadow_core::journal::rebuild_derived(&mut snapshot);
                }
            }
            Json(json!({ "journal": snapshot }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

/// Fold the trailing N calendar days into one weekly retrospective. Each day is
/// a separate `query_journal_snapshot` over its local-midnight window; the pure
/// `build_weekly_review` aggregates focus, category/app allocation, and daily
/// rollups. Narration is intentionally NOT run here (too many cards for one
/// pass) — the weekly view uses deterministic card text.
pub(super) async fn journal_weekly_handler(Query(query): Query<WeeklyQuery>) -> Response {
    use chrono::{Duration, Local};

    let day_count = query.days.unwrap_or(7).clamp(1, 31) as i64;

    // Anchor on the local calendar day containing `end`; reject an `end` beyond
    // chrono's representable range instead of panicking.
    let Some(anchor_date) = local_anchor_date(&Local, query.end) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "end is out of range" })),
        )
            .into_response();
    };

    let mut days: Vec<(String, shadow_core::journal::JournalSnapshot)> = Vec::new();
    // Walk oldest → newest so rollups render left-to-right.
    for offset in (0..day_count).rev() {
        let date = anchor_date - Duration::days(offset);
        let Some((start_us, day_end_us)) = local_day_window_us(&Local, date) else {
            continue;
        };
        // Never look past the requested end.
        let end_us = day_end_us.min(query.end);
        if end_us <= start_us {
            continue;
        }
        match shadow_core::query_journal_snapshot(start_us, end_us) {
            Ok(snapshot) => days.push((date.format("%Y-%m-%d").to_string(), snapshot)),
            Err(e) => {
                return Json(json!({ "error": e.to_string() })).into_response();
            }
        }
    }

    let window_start = days.first().map(|(_, s)| s.start_ts).unwrap_or(query.end);
    let review = shadow_core::journal::build_weekly_review(window_start, query.end, &days);
    Json(json!({ "review": review })).into_response()
}

pub(super) const MICROS_PER_SEC: u64 = 1_000_000;

/// The local calendar date in `tz` that contains `end_us`, or `None` when the
/// instant or its local wall-clock reading is outside chrono's range. The local
/// reading uses a checked add: `DateTime::date_naive` panics for an instant on
/// chrono's last UTC day in a zone east of UTC.
pub(super) fn local_anchor_date<Tz: chrono::TimeZone>(
    tz: &Tz,
    end_us: u64,
) -> Option<chrono::NaiveDate> {
    use chrono::Offset as _;

    let secs = i64::try_from(end_us / MICROS_PER_SEC).ok()?;
    let utc = chrono::DateTime::from_timestamp(secs, 0)?.naive_utc();
    let offset = tz.offset_from_utc_datetime(&utc).fix();
    utc.checked_add_offset(offset).map(|local| local.date())
}

/// Longest local-time gap searched for the first instant of a day whose
/// midnight is skipped by a DST shift (real-world gaps are at most a few hours).
pub(super) const MAX_MIDNIGHT_GAP_MINUTES: i64 = 180;

/// Unix seconds at which local calendar `date` begins in `tz`. A fall-back fold
/// at midnight resolves to the earliest instant. In zones that spring forward
/// at 00:00, midnight does not exist, so the day starts at its first local
/// minute that does.
pub(super) fn local_day_start_secs<Tz: chrono::TimeZone>(
    tz: &Tz,
    date: chrono::NaiveDate,
) -> Option<i64> {
    let midnight = date.and_hms_opt(0, 0, 0)?;
    (0..=MAX_MIDNIGHT_GAP_MINUTES).find_map(|minute| {
        let local = midnight.checked_add_signed(chrono::Duration::minutes(minute))?;
        tz.from_local_datetime(&local)
            .earliest()
            .map(|dt| dt.timestamp())
    })
}

/// Inclusive `(start_us, end_us)` window for local calendar `date` in `tz`,
/// matching `query_journal_snapshot`'s inclusive end. It runs from this day's
/// local start to one microsecond before the next day's, so DST days span their
/// real 23h/25h, consecutive days tile without overlap or gap, and an event at
/// midnight lands in exactly one day. Pre-epoch instants clamp to 0.
pub(super) fn local_day_window_us<Tz: chrono::TimeZone>(
    tz: &Tz,
    date: chrono::NaiveDate,
) -> Option<(u64, u64)> {
    let to_us = |secs: i64| {
        secs.checked_mul(MICROS_PER_SEC as i64)
            .map(|us| u64::try_from(us).unwrap_or(0))
    };
    let start_us = to_us(local_day_start_secs(tz, date)?)?;
    let next_start_us = to_us(local_day_start_secs(tz, date.succ_opt()?)?)?;
    Some((start_us, next_start_us.checked_sub(1)?))
}

/// GET /frame?ts=<micros>&display=<id> — the nearest recorded keyframe JPEG.
///
/// Returns the closest keyframe to `ts` for `display` (default 0), letting the
/// timeline scrubber show what was on screen at a moment. Keyframes are written
/// as JPEGs out of the box (pure-Rust, no ffmpeg) whenever frame capture is on
/// and not paused. Responds 404 when no keyframe exists near `ts` (frame capture
/// off/paused, or nothing recorded yet); clients render a graceful fallback.
pub(super) async fn frame_handler(Query(query): Query<FrameQuery>) -> Response {
    let display = query.display.unwrap_or(0);
    let path = match shadow_core::find_nearest_keyframe(display, query.ts) {
        Ok(Some(p)) => p,
        Ok(None) => return StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::debug!("keyframe lookup failed: {e}");
            return StatusCode::NOT_FOUND.into_response();
        }
    };
    // Keyframes are sealed at rest; open them off the async runtime.
    let read_path = std::path::PathBuf::from(&path);
    let read = tokio::task::spawn_blocking(move || shadow_core::media::read_media_file(&read_path))
        .await
        .map_err(|e| e.to_string())
        .and_then(|result| result.map_err(|e| e.to_string()));
    match read {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "image/jpeg")], bytes).into_response(),
        Err(e) => {
            tracing::debug!("keyframe read failed for {path}: {e}");
            StatusCode::NOT_FOUND.into_response()
        }
    }
}

/// GET /activity/recent?minutes=<n> — an EPHEMERAL bundle of the last N minutes
/// of screen keyframes for chat "attach recent activity". Persists NOTHING.
///
/// Gathers keyframes in [now-n_min, now] across all displays, even-subsamples to
/// at most 30, reads each JPEG from disk, and returns base64 data URLs plus a
/// short markdown summary + transcript derived from the timeline. Keyframes
/// (~every 10s) are the source, so this does NOT depend on MP4 segments existing.
/// `minutes` is clamped to 1..=15.
pub(super) async fn recent_activity_handler(
    Query(query): Query<RecentActivityQuery>,
) -> Json<serde_json::Value> {
    use base64::Engine;

    let minutes = query.minutes.unwrap_or(3).clamp(1, 15);
    let now = wall_micros();
    let start = now.saturating_sub(minutes as u64 * 60 * 1_000_000);

    // Keyframes across all displays, ordered oldest→newest.
    let kfs = match shadow_core::keyframes_between(None, start, now) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!("recent_activity keyframes_between failed: {e}");
            Vec::new()
        }
    };

    // Even-subsample to at most 30 frames.
    const MAX_FRAMES: usize = 30;
    let picked: Vec<_> = if kfs.len() <= MAX_FRAMES {
        kfs
    } else {
        let step = kfs.len() as f64 / MAX_FRAMES as f64;
        (0..MAX_FRAMES)
            .map(|i| kfs[((i as f64) * step) as usize].clone())
            .collect()
    };

    // Read + base64-encode each JPEG. Skip unreadable frames (fail-soft).
    let mut frames = Vec::with_capacity(picked.len());
    for kf in &picked {
        let read_path = std::path::PathBuf::from(&kf.file_path);
        let read =
            tokio::task::spawn_blocking(move || shadow_core::media::read_media_file(&read_path))
                .await
                .map_err(|e| e.to_string())
                .and_then(|result| result.map_err(|e| e.to_string()));
        match read {
            Ok(bytes) => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                let at_ms = kf.ts.saturating_sub(start) / 1_000;
                frames.push(json!({
                    "atMs": at_ms,
                    "dataUrl": format!("data:image/jpeg;base64,{b64}"),
                }));
            }
            Err(e) => {
                tracing::debug!(
                    "recent_activity frame read failed for {}: {e}",
                    kf.file_path
                );
            }
        }
    }

    // Short markdown summary + transcript from the timeline OCR/events.
    let (summary, transcript) = match shadow_core::query_time_range(start, now) {
        Ok(entries) => build_activity_text(minutes, &entries),
        Err(_) => (
            format!("Recent activity (last {minutes} min)"),
            String::new(),
        ),
    };

    Json(json!({
        "title": format!("Recent activity (last {minutes} min)"),
        "durationMs": minutes as u64 * 60_000,
        "summary": summary,
        "transcript": transcript,
        "frames": frames,
    }))
}

/// Derive a short markdown summary + line-per-focus transcript from timeline
/// entries. Uses app_name / window_title from `TimelineEntry`.
pub(super) fn build_activity_text(
    minutes: u32,
    entries: &[shadow_core::timeline::TimelineEntry],
) -> (String, String) {
    use std::collections::BTreeSet;

    let mut apps: BTreeSet<String> = BTreeSet::new();
    let mut lines: Vec<String> = Vec::new();
    let mut last: Option<(String, String)> = None;

    for e in entries {
        let app = e.app_name.clone().unwrap_or_default();
        let title = e.window_title.clone().unwrap_or_default();
        if !app.is_empty() {
            apps.insert(app.clone());
        }
        let cur = (app.clone(), title.clone());
        if last.as_ref() != Some(&cur) && (!app.is_empty() || !title.is_empty()) {
            let line = if title.is_empty() {
                app.clone()
            } else if app.is_empty() {
                title.clone()
            } else {
                format!("{app}: {title}")
            };
            lines.push(line);
            last = Some(cur);
        }
    }

    let apps_list: Vec<String> = apps.into_iter().collect();
    let summary = if apps_list.is_empty() {
        format!("Recent activity over the last {minutes} min. No window focus was captured.")
    } else {
        format!(
            "Recent activity over the last {minutes} min. Apps seen: {}.",
            apps_list.join(", ")
        )
    };
    (summary, lines.join("\n"))
}

pub(super) async fn recent_context_handler(Query(query): Query<SearchQuery>) -> impl IntoResponse {
    let minutes: u64 = query.q.parse().unwrap_or(10);
    let now = wall_micros();
    let start = now.saturating_sub(minutes * 60 * 1_000_000);

    match shadow_core::query_time_range(start, now) {
        Ok(entries) => {
            let count = entries.len();
            Json(json!({ "entries": entries, "count": count, "window_minutes": minutes }))
        }
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}
