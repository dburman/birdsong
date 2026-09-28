-- A person's verdict on a detection, from the dashboard: 'correct' or 'wrong'; NULL = not reviewed.
ALTER TABLE detections ADD COLUMN review TEXT CHECK (review IN ('correct', 'wrong'));
ALTER TABLE detections ADD COLUMN reviewed_at_utc TEXT;

CREATE INDEX idx_detections_review ON detections(scientific_name) WHERE review IS NOT NULL;
