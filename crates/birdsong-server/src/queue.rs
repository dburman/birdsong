//! Bounded hand-off from the async chunker tasks to the blocking inference thread.
//!
//! Only chunks count towards the capacity. Control markers (gap, end of stream, missing chunk) are
//! never dropped so per-source state in the worker stays correct. When full, a push either drops
//! the oldest queued chunk (live audio: keep up, lose old audio) or waits (file analysis: lose
//! nothing). A dropped chunk is replaced by a `Missing` marker in the same position, so the
//! privacy filter can treat the gap as possibly containing human speech.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use birdsong_audio::Chunk;
use chrono::{DateTime, Utc};
use tokio::sync::Notify;

/// What a producer does when the queue already holds `capacity` chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backpressure {
    /// Discard the oldest queued chunk (live sources).
    DropOldest,
    /// Wait for the inference thread to take a chunk (finite sources decoded faster than real time).
    Wait,
}

#[derive(Debug)]
pub(crate) enum QueueItem {
    Chunk(Chunk),
    /// The source's audio was discontinuous; flush its privacy state.
    Gap(Arc<str>),
    /// The source finished; flush its privacy state.
    End(Arc<str>),
    /// A chunk of this source was dropped here because inference fell behind.
    Missing(Arc<str>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum PushOutcome {
    Queued,
    DroppedOldest {
        source_id: Arc<str>,
        start_at: DateTime<Utc>,
    },
    /// The inference thread has stopped; nothing will be processed.
    ConsumerGone,
}

#[derive(Debug, Default)]
struct Inner {
    items: VecDeque<QueueItem>,
    chunks: usize,
    producers_done: bool,
    consumer_gone: bool,
}

pub(crate) struct ChunkQueue {
    inner: Mutex<Inner>,
    available: Condvar,
    space: Notify,
    capacity: usize,
}

impl ChunkQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::default(),
            available: Condvar::new(),
            space: Notify::new(),
            capacity: capacity.max(1),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub async fn push(&self, item: QueueItem, policy: Backpressure) -> PushOutcome {
        loop {
            let mut notified = std::pin::pin!(self.space.notified());
            notified.as_mut().enable(); // register before checking, so no wake-up is lost
            {
                let mut g = self.lock();
                if g.consumer_gone {
                    return PushOutcome::ConsumerGone;
                }
                let is_chunk = matches!(item, QueueItem::Chunk(_));
                if !is_chunk || g.chunks < self.capacity {
                    if is_chunk {
                        g.chunks += 1;
                    }
                    g.items.push_back(item);
                    drop(g);
                    self.available.notify_one();
                    return PushOutcome::Queued;
                }
                if policy == Backpressure::DropOldest {
                    let outcome = drop_oldest_chunk(&mut g.items);
                    g.items.push_back(item); // one chunk out, one in: chunk count unchanged
                    drop(g);
                    self.available.notify_one();
                    return outcome;
                }
            }
            notified.await;
        }
    }

    /// Blocking pop for the inference thread. `None` once producers are done and the queue is empty.
    pub fn pop(&self) -> Option<QueueItem> {
        let mut g = self.lock();
        loop {
            if let Some(item) = g.items.pop_front() {
                if matches!(item, QueueItem::Chunk(_)) {
                    g.chunks -= 1;
                    drop(g);
                    self.space.notify_one();
                }
                return Some(item);
            }
            if g.producers_done {
                return None;
            }
            g = self.available.wait(g).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// No more items will be pushed.
    pub fn close_producers(&self) {
        self.lock().producers_done = true;
        self.available.notify_all();
    }

    /// The consumer stopped (normally or by panic); release any waiting producers.
    pub fn consumer_gone(&self) {
        self.lock().consumer_gone = true;
        self.space.notify_waiters();
    }
}

/// Replace the oldest queued chunk with a `Missing` marker for its source. Consecutive markers for
/// the same source are merged so a stalled consumer cannot grow the queue without bound.
fn drop_oldest_chunk(items: &mut VecDeque<QueueItem>) -> PushOutcome {
    let Some(position) = items.iter().position(|i| matches!(i, QueueItem::Chunk(_))) else {
        return PushOutcome::Queued;
    };
    let source_id = match &items[position] {
        QueueItem::Chunk(c) => Arc::clone(&c.source_id),
        _ => return PushOutcome::Queued,
    };
    let merge =
        position > 0 && matches!(&items[position - 1], QueueItem::Missing(s) if *s == source_id);
    let removed = if merge {
        items.remove(position)
    } else {
        Some(std::mem::replace(
            &mut items[position],
            QueueItem::Missing(Arc::clone(&source_id)),
        ))
    };
    match removed {
        Some(QueueItem::Chunk(c)) => PushOutcome::DroppedOldest {
            source_id: c.source_id,
            start_at: c.start_at,
        },
        _ => PushOutcome::Queued,
    }
}

/// Marks the consumer gone when dropped, including during a panic unwind.
pub(crate) struct ConsumerGuard(pub Arc<ChunkQueue>);

impl Drop for ConsumerGuard {
    fn drop(&mut self) {
        self.0.consumer_gone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeDelta, TimeZone};
    use std::time::Duration;

    fn chunk(n: i64) -> QueueItem {
        QueueItem::Chunk(Chunk {
            samples: Arc::from(vec![n as f32]),
            start_at: Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap()
                + TimeDelta::seconds(3 * n),
            source_id: Arc::from("mic0"),
            padded: false,
        })
    }

    fn label(item: &QueueItem) -> String {
        match item {
            QueueItem::Chunk(c) => format!("c{}", c.samples[0]),
            QueueItem::Gap(_) => "gap".into(),
            QueueItem::End(_) => "end".into(),
            QueueItem::Missing(_) => "missing".into(),
        }
    }

    fn drain(q: &ChunkQueue) -> Vec<String> {
        q.close_producers();
        std::iter::from_fn(|| q.pop()).map(|i| label(&i)).collect()
    }

    #[tokio::test]
    async fn drop_oldest_keeps_controls_and_marks_the_gap() {
        let q = ChunkQueue::new(2);
        assert_eq!(
            q.push(chunk(0), Backpressure::DropOldest).await,
            PushOutcome::Queued
        );
        assert_eq!(
            q.push(QueueItem::Gap(Arc::from("mic0")), Backpressure::DropOldest)
                .await,
            PushOutcome::Queued
        );
        assert_eq!(
            q.push(chunk(1), Backpressure::DropOldest).await,
            PushOutcome::Queued
        );
        match q.push(chunk(2), Backpressure::DropOldest).await {
            PushOutcome::DroppedOldest { start_at, .. } => {
                assert_eq!(start_at, Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 0).unwrap())
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(drain(&q), ["missing", "gap", "c1", "c2"]);
    }

    #[tokio::test]
    async fn consecutive_drops_merge_into_one_marker() {
        let q = ChunkQueue::new(2);
        for n in 0..6 {
            q.push(chunk(n), Backpressure::DropOldest).await;
        }
        // c0..c3 dropped one after another: a single marker, then the two newest chunks.
        assert_eq!(drain(&q), ["missing", "c4", "c5"]);
    }

    #[tokio::test]
    async fn wait_blocks_until_space() {
        let q = Arc::new(ChunkQueue::new(1));
        q.push(chunk(0), Backpressure::Wait).await;
        let pusher = {
            let q = Arc::clone(&q);
            tokio::spawn(async move { q.push(chunk(1), Backpressure::Wait).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!pusher.is_finished(), "must wait while full");
        let popped = tokio::task::spawn_blocking({
            let q = Arc::clone(&q);
            move || q.pop().map(|i| label(&i))
        })
        .await
        .unwrap();
        assert_eq!(popped.as_deref(), Some("c0"));
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), pusher)
                .await
                .unwrap()
                .unwrap(),
            PushOutcome::Queued
        );
        assert_eq!(drain(&q), ["c1"]);
    }

    #[tokio::test]
    async fn consumer_gone_releases_waiters() {
        let q = Arc::new(ChunkQueue::new(1));
        q.push(chunk(0), Backpressure::Wait).await;
        let pusher = {
            let q = Arc::clone(&q);
            tokio::spawn(async move { q.push(chunk(1), Backpressure::Wait).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        drop(ConsumerGuard(Arc::clone(&q)));
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), pusher)
                .await
                .unwrap()
                .unwrap(),
            PushOutcome::ConsumerGone
        );
    }

    #[test]
    fn pop_returns_none_after_close() {
        let q = ChunkQueue::new(4);
        q.close_producers();
        assert!(q.pop().is_none());
    }
}
