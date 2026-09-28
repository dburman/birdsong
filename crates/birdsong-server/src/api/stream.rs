//! `GET /stream`: stored detections as Server-Sent Events, with `Last-Event-ID` replay.

use std::collections::VecDeque;
use std::convert::Infallible;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, KeepAlive, Sse};
use birdsong_core::Detection;
use birdsong_store::{DetectionQuery, DetectionRecord, Order, MAX_LIMIT};
use futures_util::stream::{self, Stream};
use tokio::sync::broadcast::{self, error::RecvError};

use super::AppState;

struct Cursor {
    state: AppState,
    live: broadcast::Receiver<Detection>,
    /// Highest id sent (or the client's `Last-Event-ID`).
    last_id: Option<i64>,
    backlog: VecDeque<DetectionRecord>,
    replay: bool,
}

fn event(record: &DetectionRecord) -> Option<Event> {
    Event::default()
        .event("detection")
        .id(record.id.to_string())
        .json_data(record)
        .ok()
}

async fn next_event(mut c: Cursor) -> Option<(Result<Event, Infallible>, Cursor)> {
    loop {
        if let Some(record) = c.backlog.pop_front() {
            if c.last_id.is_some_and(|last| record.id <= last) {
                continue;
            }
            c.last_id = Some(record.id);
            if let Some(ev) = event(&record) {
                return Some((Ok(ev), c));
            }
            continue;
        }

        if c.replay {
            c.replay = false;
            if let Some(after) = c.last_id {
                let query = DetectionQuery {
                    after_id: Some(after),
                    order: Order::Asc,
                    limit: Some(MAX_LIMIT),
                    ..Default::default()
                };
                match c.state.store.list(&query).await {
                    Ok(items) => {
                        c.replay = items.len() as u32 == MAX_LIMIT; // more to fetch after these
                        c.backlog.extend(items);
                        continue;
                    }
                    Err(e) => tracing::warn!(error = %e, "event stream replay failed"),
                }
            }
        }

        tokio::select! {
            _ = c.state.shutdown.cancelled() => return None,
            received = c.live.recv() => match received {
                Ok(detection) => {
                    let Some(id) = detection.id else { continue };
                    if c.last_id.is_some_and(|last| id <= last) {
                        continue;
                    }
                    match c.state.store.get(id).await {
                        Ok(Some(record)) => c.backlog.push_back(record),
                        Ok(None) => {}
                        Err(e) => tracing::warn!(error = %e, id, "event stream lookup failed"),
                    }
                }
                Err(RecvError::Lagged(missed)) => {
                    tracing::warn!(missed, "event stream subscriber lagged; replaying from the database");
                    c.replay = true;
                }
                Err(RecvError::Closed) => return None,
            },
        }
    }
}

/// Subscribe first, then replay anything after `Last-Event-ID`, then forward live detections.
/// Ids already sent are skipped, so a replay and the live feed never duplicate an event.
pub async fn stream(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let last_id = headers
        .get("last-event-id")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<i64>().ok());
    let cursor = Cursor {
        live: state.detections.subscribe(),
        state,
        last_id,
        backlog: VecDeque::new(),
        replay: last_id.is_some(),
    };
    Sse::new(stream::unfold(cursor, next_event)).keep_alive(KeepAlive::default())
}
