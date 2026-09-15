# Birdsong HTTP API (v1)

`birdsong run` serves this API on `server.bind` (default `0.0.0.0:8080`). Everything lives under
`/api/v1`, returns JSON unless noted, and needs no authentication: keep the port on your local
network or put a reverse proxy in front of it.

- Timestamps are RFC 3339 in UTC with microsecond precision, for example `2026-05-15T10:00:00.000000Z`.
- `local_date` and `local_hour` use the station time zone (`station.timezone`).
- Errors are JSON with a matching status code: `{"error": "limit must be a positive integer, got \"abc\""}`.
  Bad parameters give `400`, unknown ids, missing files and unknown routes give `404`.
- CORS follows `server.cors_allow_origins` (default `*`). Responses are gzip-compressed when the
  client asks, except audio and the event stream.

Examples use `PI=http://birdsong.local:8080`.

## Ingesting detections

Two ways to receive every detection exactly once, with the time it happened.

**Poll with a cursor.** Detection ids are assigned in the order detections are stored and are
never reused, so the highest id you have processed is a complete cursor.

```bash
PI=http://birdsong.local:8080
cursor=0   # persist this between runs
while true; do
  page=$(curl -s "$PI/api/v1/detections?after_id=$cursor&order=asc&limit=500")
  echo "$page" | jq -c '.items[] | {id, detected_at, common_name, confidence}'
  next=$(echo "$page" | jq '.next_after_id')
  [ "$next" != "null" ] && cursor=$next
  sleep 30
done
```

**Push with Server-Sent Events.** Keep one connection open. Send the last id you processed as
`Last-Event-ID` when you reconnect and anything stored in between is replayed first.

```bash
curl -N "$PI/api/v1/stream"
curl -N -H "Last-Event-ID: 1234" "$PI/api/v1/stream"
```

Browsers do this automatically: `new EventSource("/api/v1/stream")` reconnects with the last id.

## Detection object

Returned by the detection endpoints and as the `data` of stream events.

```json
{
  "id": 1234,
  "detected_at": "2026-05-15T10:00:00.000000Z",
  "local_date": "2026-05-15",
  "local_hour": 6,
  "week": 19,
  "scientific_name": "Poecile atricapillus",
  "common_name": "Black-capped Chickadee",
  "confidence": 0.752,
  "source_id": "mic0",
  "model_id": "birdnet-v2.4",
  "clip_path": "2026-05-15/Black_capped_Chickadee/2026-05-15T10-00-00.000Z_mic0_0.75.flac",
  "clip_bytes": 413702,
  "spectrogram_path": "2026-05-15/Black_capped_Chickadee/2026-05-15T10-00-00.000Z_mic0_0.75.png"
}
```

`detected_at` is the start of the 3-second analysis window. `clip_bytes` is the disk space used by
the clip's audio and spectrogram together, which is what `retention.clip_max_total_mb` counts.
`clip_path` and `spectrogram_path` become `null` when the retention window deletes the files; the
detection itself is kept. A clip is
attached a moment after the detection is stored, so a brand-new detection can briefly have
`clip_path: null`.

## Endpoints

### `GET /api/v1/health`

Liveness and pipeline counters. Always `200` while the process runs.

```bash
curl -s "$PI/api/v1/health"
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_s": 3600,
  "station": "Backyard",
  "model_id": "birdnet-v2.4",
  "last_chunk_at": "2026-05-15T10:59:57.000000Z",
  "seconds_since_last_chunk": 3.1,
  "stats": {
    "chunks_processed": 1200, "chunks_dropped": 0, "masked_chunks": 4, "gaps": 0,
    "detections": 57, "inference_errors": 0, "store_errors": 0,
    "clips_written": 41, "clip_errors": 0, "mean_inference_ms": 312.5,
    "last_chunk_at": "2026-05-15T10:59:57.000000Z",
    "last_processed_at": "2026-05-15T11:00:00.300000Z"
  }
}
```

`last_chunk_at` is the audio time of the newest analysed chunk; `seconds_since_last_chunk` is
measured from when it was processed (wall clock). A growing `seconds_since_last_chunk` means audio stopped arriving; a growing `chunks_dropped`
means the computer cannot keep up with inference.

### `GET /api/v1/detections`

Detections matching every given filter.

| Parameter | Meaning |
|-----------|---------|
| `after_id` | Only ids greater than this. The cursor for ingestion. |
| `before_id` | Only ids less than this. For paging back in time. |
| `since` | `detected_at` at or after this RFC 3339 time. |
| `until` | `detected_at` before this RFC 3339 time. |
| `species` | Exact scientific name, for example `Cardinalis cardinalis`. |
| `min_confidence` | `0` to `1`. |
| `limit` | Page size, default `100`. Values above `1000` are treated as `1000`. |
| `order` | `desc` (newest first, default) or `asc`. |

```bash
curl -s "$PI/api/v1/detections?species=Cardinalis%20cardinalis&min_confidence=0.8&limit=10"
curl -s "$PI/api/v1/detections?since=2026-05-15T00:00:00Z&until=2026-05-16T00:00:00Z&order=asc"
```

```json
{ "items": [ { "id": 1234, "...": "..." } ], "next_after_id": 1234, "next_before_id": null }
```

- `next_after_id` is the highest id on the page, or `null` for an empty page. Use it as `after_id`.
- `next_before_id` is the lowest id when the page is full, otherwise `null`. Use it as `before_id`
  to fetch the previous page of a newest-first listing.

### `GET /api/v1/detections/latest`

The newest detections. `limit` defaults to `20`. Same response shape as above.

```bash
curl -s "$PI/api/v1/detections/latest?limit=5"
```

### `GET /api/v1/detections/{id}`

One detection. `404` if the id does not exist.

```bash
curl -s "$PI/api/v1/detections/1234"
```

### `GET /api/v1/detections/{id}/audio`

The saved clip as `audio/flac` (default) or `audio/wav`, following `storage.clip_format`: mono, 48 kHz, 16-bit, 6 s by default. Supports `Range` requests so
`<audio>` elements can seek. `404` when the clip was never saved or has been purged.

```bash
curl -s -o clip.flac "$PI/api/v1/detections/1234/audio"
curl -s -H "Range: bytes=0-99" -o head.bin "$PI/api/v1/detections/1234/audio"
```

```html
<audio controls src="http://birdsong.local:8080/api/v1/detections/1234/audio"></audio>
```

### `GET /api/v1/detections/{id}/spectrogram.png`

An 800 × 300 indexed-colour PNG spectrogram (0 to 12 kHz) of the clip. `404` when absent.

```bash
curl -s -o spec.png "$PI/api/v1/detections/1234/spectrogram.png"
```

### `GET /api/v1/species`

One entry per species, most detections first. Optional `since` (RFC 3339); all time without it.

```bash
curl -s "$PI/api/v1/species?since=2026-05-01T00:00:00Z"
```

```json
{
  "since": "2026-05-01T00:00:00Z",
  "items": [
    {
      "scientific_name": "Poecile atricapillus",
      "common_name": "Black-capped Chickadee",
      "count": 42,
      "first_seen": "2026-05-01T09:12:03.000000Z",
      "last_seen": "2026-05-15T10:00:00.000000Z",
      "max_confidence": 0.93,
      "best_detection_id": 1102,
      "best_clip_detection_id": 1102
    }
  ]
}
```

`best_clip_detection_id` is the highest-confidence detection that still has audio, or `null`.

### `GET /api/v1/stats/daily`

Detections per species per local hour for one day: the data for a "today by hour" chart.
`date` is `YYYY-MM-DD` in the station time zone and defaults to today.

```bash
curl -s "$PI/api/v1/stats/daily?date=2026-05-15"
```

```json
{
  "date": "2026-05-15",
  "species": [
    {
      "scientific_name": "Poecile atricapillus",
      "common_name": "Black-capped Chickadee",
      "total": 12,
      "by_hour": [0,0,0,0,0,2,5,3,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
    }
  ]
}
```

`by_hour[0]` is midnight to 1 a.m. local time. Species are sorted by `total`, most first.

### `GET /api/v1/stats/recent`

Per-species counts for a recent window: the data for a "latest birds" bar chart. `window` is one
of `1h`, `6h`, `24h` (default), `7d`, `30d`.

```bash
curl -s "$PI/api/v1/stats/recent?window=6h"
```

```json
{
  "window": "6h",
  "since": "2026-05-15T05:00:00Z",
  "until": "2026-05-15T11:00:00Z",
  "species": [ { "common_name": "Black-capped Chickadee", "count": 9, "...": "same fields as /species" } ]
}
```

### `GET /api/v1/stream`

Server-Sent Events (`text/event-stream`). Each stored detection produces:

```text
event: detection
id: 1234
data: {"id":1234,"detected_at":"2026-05-15T10:00:00.000000Z", ...}
```

- With a `Last-Event-ID` header, detections with larger ids are replayed from the database before
  live events. Ids are never sent twice on one connection.
- Comment lines (`:`) are sent periodically to keep proxies from closing idle connections.
- The stream closes when the server shuts down; clients should reconnect with `Last-Event-ID`.

```bash
curl -N "$PI/api/v1/stream"
```

### `GET /metrics`

Prometheus text format (outside `/api/v1`, where scrapers expect it). Counters reset when the
process restarts; `birdsong_species_detections` comes from the database and does not.

```bash
curl -s "http://birdsong.local:8080/metrics"
```

```text
# TYPE birdsong_chunks_processed_total counter
birdsong_chunks_processed_total 1200
birdsong_chunks_dropped_total 0
birdsong_inference_seconds 0.312
birdsong_seconds_since_last_chunk 1.4
birdsong_clip_bytes 25318044
birdsong_species_detections{scientific_name="Poecile atricapillus",common_name="Black-capped Chickadee"} 42
birdsong_build_info{version="0.1.0",model="birdnet-v2.4",station="Backyard"} 1
```

Also exported: `birdsong_uptime_seconds`, `birdsong_masked_chunks_total`,
`birdsong_audio_gaps_total`, `birdsong_detections_total`, `birdsong_inference_errors_total`,
`birdsong_store_errors_total`, `birdsong_clips_written_total`, `birdsong_clip_errors_total`.

### `GET /api/v1/config`

The effective configuration, with credentials in stream URLs replaced by `***`.

```bash
curl -s "$PI/api/v1/config" | jq .detection
```
