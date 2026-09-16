-- What a detection is: 'animal' (a species, or an animal sound such as a dog's bark) or
-- 'sound_event' (engines, rain, music, people). Charts and species lists show animals by default.
ALTER TABLE detections ADD COLUMN kind TEXT NOT NULL DEFAULT 'animal'
    CHECK (kind IN ('animal', 'sound_event'));

-- Rows stored before this column existed. BirdNET V2.4's non-animal classes (the human classes
-- are normally masked by the privacy filter, but not when it is off):
UPDATE detections SET kind = 'sound_event'
WHERE scientific_name IN ('Engine', 'Environmental', 'Fireworks', 'Gun', 'Noise', 'Siren',
                          'Human vocal', 'Human non-vocal', 'Human whistle');

-- Perch v2 sound events have no space in their name; a few of them are animals.
UPDATE detections SET kind = 'sound_event'
WHERE model_id = 'perch-v2'
  AND instr(scientific_name, ' ') = 0
  AND scientific_name NOT IN ('Animal', 'Bark', 'Cat', 'Chicken_and_rooster', 'Chirp_and_tweet',
      'Cricket', 'Crow', 'Dog', 'Domestic_animals_and_pets', 'Fowl', 'Frog', 'Growling',
      'Gull_and_seagull', 'Insect', 'Livestock_and_farm_animals_and_working_animals', 'Meow',
      'Purr', 'Wild_animals');

CREATE INDEX idx_detections_kind_time ON detections(kind, detected_at_utc DESC);
