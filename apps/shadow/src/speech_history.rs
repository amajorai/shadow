//! Bounded, read-only projection of Shadow's existing transcript index.
use axum::{extract::Query, http::StatusCode, response::IntoResponse, Json};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SpeechHistoryQuery {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    limit: Option<u32>,
    offset: Option<u32>,
}

impl SpeechHistoryQuery {
    fn bounds(&self) -> Result<(u64, u64, u32, u32), &'static str> {
        let start = self.start.timestamp_micros();
        let end = self.end.timestamp_micros();
        let limit = self.limit.unwrap_or(100);
        let offset = self.offset.unwrap_or(0);
        if start < 0
            || end <= start
            || end - start > 86_400_000_000
            || !(1..=500).contains(&limit)
            || offset > 10_000
        {
            return Err("Choose a range of up to one day, a limit from 1 to 500, and an offset up to 10000.");
        }
        Ok((start as u64, end as u64, limit, offset))
    }
}

pub async fn transcripts_handler(Query(query): Query<SpeechHistoryQuery>) -> impl IntoResponse {
    let (start, end, limit, offset) = match query.bounds() {
        Ok(bounds) => bounds,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "code": "invalid_range", "message": message })),
            )
        }
    };
    let result = tokio::task::spawn_blocking(move || {
        shadow_core::list_transcript_chunks_overlapping(start, end, limit + 1, offset)
    })
    .await;
    match result {
        Ok(Ok(mut chunks)) => {
            let more = chunks.len() > limit as usize;
            chunks.truncate(limit as usize);
            let segments: Vec<_> = chunks.into_iter().filter_map(|chunk| {
                if chunk.text.trim().is_empty() { return None; }
                let started = DateTime::from_timestamp_micros(i64::try_from(chunk.ts_start).ok()?)?;
                let ended = DateTime::from_timestamp_micros(i64::try_from(chunk.ts_end).ok()?)?;
                Some(json!({
                    "id": format!("{}:{}:{}:{}", chunk.audio_source, chunk.audio_segment_id, chunk.ts_start, chunk.ts_end),
                    "startedAt": started.to_rfc3339_opts(SecondsFormat::Micros, true),
                    "endedAt": ended.to_rfc3339_opts(SecondsFormat::Micros, true),
                    "text": chunk.text,
                    "timing": "window",
                }))
            }).collect();
            let truncated = more && offset + limit > 10_000;
            (
                StatusCode::OK,
                Json(
                    json!({ "segments": segments, "nextOffset": if more && !truncated { Some(offset + limit) } else { None }, "truncated": truncated }),
                ),
            )
        }
        _ => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(
                json!({ "code": "speech_unavailable", "message": "Shadow's speech index is unavailable. Check that audio transcription is running." }),
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_unbounded_or_reversed_queries() {
        for (start, end, limit, offset) in [
            ("2026-09-12T12:00:00Z", "2026-09-12T11:00:00Z", 10, 0),
            ("2026-09-01T00:00:00Z", "2026-09-12T00:00:00Z", 10, 0),
            ("2026-09-12T00:00:00Z", "2026-09-12T00:01:00Z", 501, 0),
            ("2026-09-12T00:00:00Z", "2026-09-12T00:01:00Z", 10, 10001),
        ] {
            let query: SpeechHistoryQuery = serde_json::from_value(
                json!({ "start": start, "end": end, "limit": limit, "offset": offset }),
            )
            .unwrap();
            assert!(query.bounds().is_err());
        }
    }
    #[test]
    fn accepts_explicit_fifteen_second_window() {
        let query: SpeechHistoryQuery = serde_json::from_value(
            json!({ "start": "2026-09-12T12:00:00Z", "end": "2026-09-12T12:00:15Z" }),
        )
        .unwrap();
        let (start, end, limit, offset) = query.bounds().unwrap();
        assert_eq!(end - start, 15_000_000);
        assert_eq!((limit, offset), (100, 0));
    }
}
