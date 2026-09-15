-- Detections. One row per species per 3 s analysis window.
-- Timestamps are RFC 3339 UTC with exactly six fractional digits, so text order = time order.
CREATE TABLE detections (
    -- AUTOINCREMENT: ids are never reused, even after old rows are deleted, so `after_id`
    -- cursors used by API consumers can never skip or repeat a row.
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at_utc   TEXT    NOT NULL,   -- start of the 3 s window
    local_date        TEXT    NOT NULL,   -- YYYY-MM-DD in the station time zone
    local_hour        INTEGER NOT NULL,   -- 0..23 in the station time zone
    scientific_name   TEXT    NOT NULL,
    common_name       TEXT    NOT NULL,
    confidence        REAL    NOT NULL,
    source_id         TEXT    NOT NULL,
    model_id          TEXT    NOT NULL,
    latitude          REAL,
    longitude         REAL,
    week              INTEGER,            -- BirdNET 48-week year
    sensitivity       REAL,
    overlap_seconds   REAL,
    min_confidence    REAL,
    clip_path         TEXT,               -- relative to <data_dir>/clips; NULL when purged or never saved
    clip_bytes        INTEGER,
    spectrogram_path  TEXT,
    created_at_utc    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_detections_time    ON detections(detected_at_utc DESC);
CREATE INDEX idx_detections_species ON detections(scientific_name, detected_at_utc DESC);
CREATE INDEX idx_detections_day     ON detections(local_date, local_hour);
CREATE INDEX idx_detections_clip    ON detections(clip_path) WHERE clip_path IS NOT NULL;
