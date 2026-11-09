use super::*;

/// A test zone with one offset change at the UTC instant `switch`, enough to
/// model a spring-forward gap or a fall-back fold without a tz database.
#[derive(Clone, Copy, Debug)]
struct OneShiftZone {
    switch: chrono::NaiveDateTime,
    before: chrono::FixedOffset,
    after: chrono::FixedOffset,
}

#[derive(Clone, Copy, Debug)]
struct OneShiftOffset {
    zone: OneShiftZone,
    current: chrono::FixedOffset,
}

impl chrono::Offset for OneShiftOffset {
    fn fix(&self) -> chrono::FixedOffset {
        self.current
    }
}

impl chrono::TimeZone for OneShiftZone {
    type Offset = OneShiftOffset;

    fn from_offset(offset: &OneShiftOffset) -> Self {
        offset.zone
    }

    fn offset_from_local_date(
        &self,
        local: &chrono::NaiveDate,
    ) -> chrono::LocalResult<OneShiftOffset> {
        self.offset_from_local_datetime(&local.and_hms_opt(0, 0, 0).unwrap())
    }

    fn offset_from_local_datetime(
        &self,
        local: &chrono::NaiveDateTime,
    ) -> chrono::LocalResult<OneShiftOffset> {
        let utc_for = |offset: chrono::FixedOffset| {
            *local - chrono::Duration::seconds(i64::from(offset.local_minus_utc()))
        };
        let wrap = |current| OneShiftOffset {
            zone: *self,
            current,
        };
        let pre = (utc_for(self.before) < self.switch).then_some(self.before);
        let post = (utc_for(self.after) >= self.switch).then_some(self.after);
        match (pre, post) {
            // The pre-transition reading is the earlier instant.
            (Some(a), Some(b)) => chrono::LocalResult::Ambiguous(wrap(a), wrap(b)),
            (Some(only), None) | (None, Some(only)) => chrono::LocalResult::Single(wrap(only)),
            (None, None) => chrono::LocalResult::None,
        }
    }

    fn offset_from_utc_date(&self, utc: &chrono::NaiveDate) -> OneShiftOffset {
        self.offset_from_utc_datetime(&utc.and_hms_opt(0, 0, 0).unwrap())
    }

    fn offset_from_utc_datetime(&self, utc: &chrono::NaiveDateTime) -> OneShiftOffset {
        let current = if *utc < self.switch {
            self.before
        } else {
            self.after
        };
        OneShiftOffset {
            zone: *self,
            current,
        }
    }
}

fn hours(h: i32) -> chrono::FixedOffset {
    chrono::FixedOffset::east_opt(h * 3600).unwrap()
}

fn ymd(y: i32, m: u32, d: u32) -> chrono::NaiveDate {
    chrono::NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn utc_us(y: i32, m: u32, d: u32, h: u32) -> u64 {
    let secs = ymd(y, m, d)
        .and_hms_opt(h, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    u64::try_from(secs).unwrap() * 1_000_000
}

const HOUR_US: u64 = 3_600 * 1_000_000;

/// Consecutive day windows must neither overlap nor leave a gap.
fn assert_windows_tile<Tz: chrono::TimeZone>(tz: &Tz, first: chrono::NaiveDate, days: u32) {
    let mut date = first;
    let mut prev = local_day_window_us(tz, date).unwrap();
    for _ in 0..days {
        date = date.succ_opt().unwrap();
        let next = local_day_window_us(tz, date).unwrap();
        assert!(prev.0 < prev.1, "{date}: empty window {prev:?}");
        assert_eq!(next.0, prev.1 + 1, "{date}: windows must tile");
        prev = next;
    }
}

#[test]
fn weekly_day_windows_follow_dst_transitions() {
    // New York-like: 02:00 EST → 03:00 EDT on 2026-03-08 (07:00 UTC).
    let spring = OneShiftZone {
        switch: ymd(2026, 3, 8).and_hms_opt(7, 0, 0).unwrap(),
        before: hours(-5),
        after: hours(-4),
    };
    let (start, end) = local_day_window_us(&spring, ymd(2026, 3, 8)).unwrap();
    assert_eq!(start, utc_us(2026, 3, 8, 5));
    assert_eq!(end + 1 - start, 23 * HOUR_US);
    let (start, end) = local_day_window_us(&spring, ymd(2026, 3, 7)).unwrap();
    assert_eq!(end + 1 - start, 24 * HOUR_US);
    assert_windows_tile(&spring, ymd(2026, 3, 5), 6);

    // 02:00 EDT → 01:00 EST on 2026-11-01 (06:00 UTC).
    let fall = OneShiftZone {
        switch: ymd(2026, 11, 1).and_hms_opt(6, 0, 0).unwrap(),
        before: hours(-4),
        after: hours(-5),
    };
    let (start, end) = local_day_window_us(&fall, ymd(2026, 11, 1)).unwrap();
    assert_eq!(start, utc_us(2026, 11, 1, 4));
    assert_eq!(end + 1 - start, 25 * HOUR_US);
    assert_windows_tile(&fall, ymd(2026, 10, 29), 6);
}

#[test]
fn weekly_day_window_handles_a_midnight_gap_and_fold() {
    // Santiago-like: 00:00 → 01:00 on 2026-09-06 (04:00 UTC), so local
    // midnight does not exist and the day starts at 01:00 local.
    let gap = OneShiftZone {
        switch: ymd(2026, 9, 6).and_hms_opt(4, 0, 0).unwrap(),
        before: hours(-4),
        after: hours(-3),
    };
    let (start, end) = local_day_window_us(&gap, ymd(2026, 9, 6)).unwrap();
    assert_eq!(start, utc_us(2026, 9, 6, 4));
    assert_eq!(end + 1 - start, 23 * HOUR_US);
    assert_windows_tile(&gap, ymd(2026, 9, 3), 6);

    // 01:00 → 00:00 on 2026-04-05 (04:00 UTC), so midnight occurs twice:
    // the day starts at its earliest midnight and keeps the repeated hour,
    // while the previous day loses nothing.
    let fold = OneShiftZone {
        switch: ymd(2026, 4, 5).and_hms_opt(4, 0, 0).unwrap(),
        before: hours(-3),
        after: hours(-4),
    };
    let (start, end) = local_day_window_us(&fold, ymd(2026, 4, 5)).unwrap();
    assert_eq!(start, utc_us(2026, 4, 5, 3));
    assert_eq!(end + 1 - start, 25 * HOUR_US);
    let (start, end) = local_day_window_us(&fold, ymd(2026, 4, 4)).unwrap();
    assert_eq!(start, utc_us(2026, 4, 4, 3));
    assert_eq!(end + 1 - start, 24 * HOUR_US);
    assert_windows_tile(&fold, ymd(2026, 4, 1), 6);
}

#[test]
fn weekly_anchor_date_rejects_instants_past_the_last_local_day() {
    let last_secs = chrono::DateTime::<chrono::Utc>::MAX_UTC.timestamp();
    let last_us = u64::try_from(last_secs).unwrap() * 1_000_000;
    // East of UTC the local reading of chrono's last UTC instant falls past
    // `NaiveDate::MAX`, where `date_naive()` panicked.
    assert_eq!(local_anchor_date(&hours(2), last_us), None);
    assert_eq!(
        local_anchor_date(&hours(0), last_us),
        Some(chrono::NaiveDate::MAX)
    );
    assert_eq!(
        local_anchor_date(&hours(-5), last_us),
        Some(chrono::NaiveDate::MAX)
    );
    assert_eq!(local_anchor_date(&hours(0), u64::MAX), None);
    // An ordinary instant anchors on its local calendar day.
    assert_eq!(
        local_anchor_date(&hours(-5), utc_us(2026, 3, 8, 3)),
        Some(ymd(2026, 3, 7))
    );
}

#[test]
fn weekly_day_windows_near_the_last_date_never_panic() {
    let last_secs = chrono::DateTime::<chrono::Utc>::MAX_UTC.timestamp();
    let last_us = u64::try_from(last_secs).unwrap() * 1_000_000;
    // Whatever `TZ` the suite runs under, the system zone resolves (or
    // rejects) the last representable instants without panicking.
    let _ = local_anchor_date(&chrono::Local, last_us);
    for zone in [hours(14), hours(0), hours(-12)] {
        let anchor = local_anchor_date(&zone, last_us - 86_400 * 1_000_000);
        assert!(anchor.is_some(), "{zone}: day before the last is in range");
    }
    let mut date = chrono::NaiveDate::MAX;
    for _ in 0..32 {
        let _ = local_day_window_us(&chrono::Local, date);
        let _ = local_day_window_us(&hours(14), date);
        let _ = local_day_window_us(&hours(-12), date);
        date = date.pred_opt().unwrap();
    }
    // The last date has no successor to bound it, so it has no window.
    assert_eq!(local_day_window_us(&hours(0), chrono::NaiveDate::MAX), None);
}

#[tokio::test]
async fn journal_weekly_handler_accepts_the_last_representable_end() {
    let last_secs = chrono::DateTime::<chrono::Utc>::MAX_UTC.timestamp();
    let resp = journal_weekly_handler(Query(WeeklyQuery {
        end: u64::try_from(last_secs).unwrap() * 1_000_000,
        days: Some(31),
    }))
    .await;
    // 400 east of UTC (the local day is past chrono's range), otherwise a
    // JSON body; never a panic.
    assert!(
        resp.status() == StatusCode::OK || resp.status() == StatusCode::BAD_REQUEST,
        "{}",
        resp.status()
    );
}

#[test]
fn weekly_day_windows_tile_in_fixed_and_local_zones() {
    let utc = chrono::FixedOffset::east_opt(0).unwrap();
    let (start, end) = local_day_window_us(&utc, ymd(2026, 3, 8)).unwrap();
    assert_eq!(start, utc_us(2026, 3, 8, 0));
    assert_eq!(end + 1 - start, 24 * HOUR_US);
    assert_windows_tile(&utc, ymd(2026, 1, 1), 366);
    assert_windows_tile(&chrono::Local, ymd(2026, 1, 1), 366);
}

#[tokio::test]
async fn recent_context_handler_returns_json() {
    let resp = recent_context_handler(Query(SearchQuery {
        q: "10".to_string(),
        limit: None,
        category: None,
    }))
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn recent_activity_handler_returns_json_bundle() {
    let resp = recent_activity_handler(Query(RecentActivityQuery { minutes: Some(3) }))
        .await
        .into_response();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn frame_handler_returns_not_found_when_no_keyframe() {
    let resp = frame_handler(Query(FrameQuery {
        ts: 0,
        display: None,
    }))
    .await
    .into_response();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
