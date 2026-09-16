# Dashboard manual checklist

Run through this after changing anything under `static/` or the API responses it uses. Quickest
setup: `birdsong run --fast-files` with a `kind = "file"` source pointing at a recording with
several species, then open `http://127.0.0.1:8080/`.

## Loading

- [ ] The page title and header show `station.name`.
- [ ] The status strip reads "Listening · N chunks · N ms per chunk · N detections since start".
- [ ] No errors in the browser console.

## Latest birds

- [ ] Bars are sorted by count, longest first; counts sit at the end of each bar.
- [ ] The 1h / 6h / 24h / 7d / 30d buttons switch data and only one is highlighted.
- [ ] A window with no detections says so instead of drawing an empty chart.

## By hour

- [ ] Columns line up with the hour labels 00, 03, … 21, in the station time zone.
- [ ] Hovering a segment shows the hour, species and count.
- [ ] The legend lists the same species and colours; more than 7 species adds an "Other" entry.
- [ ] Picking another date loads it; future dates cannot be picked.

## Recent detections

- [ ] Newest first; time in the station time zone with "N min ago" below.
- [ ] A new detection appears at the top within a few seconds, briefly highlighted, and the
      "live" indicator is green. Stopping the server turns it to "reconnecting"; restarting
      delivers anything missed.
- [ ] Right after a new detection the recording cell may say "saving…"; within about 10 s the
      play button and thumbnail appear.
- [ ] ▶ plays the clip, changes to ❚❚, and starting another clip stops the first.
- [ ] Clicking a thumbnail opens the viewer with the full spectrogram and an audio player;
      Escape or × closes it and stops audio.
- [ ] Detections whose clips were purged show a disabled button and "not kept".

## Species

- [ ] Sorted by detection count; "Best" is the highest confidence.
- [ ] The best-recording button plays that species' best remaining clip.

## Sound events

- [ ] Non-animal sounds (engine, siren, rain, music) appear only in the "Sound events" card, never in
      the bar chart, the hourly chart, recent detections or the species table.
- [ ] A dog, cat or frog is listed with the animals, not as a sound event.
- [ ] A new sound event updates the card within a few seconds without adding a row to recent
      detections; its best-recording button plays the clip.
- [ ] Sound-event names show once ("Car passing by"), with no duplicate scientific name below.

## Layout and accessibility

- [ ] At about 400 px wide there is no horizontal page scrolling; tables scroll inside their card.
- [ ] Dark mode (system setting) is readable, including chart colours and axis labels.
- [ ] Tab reaches the window buttons, date picker, play buttons and thumbnails, with a visible focus ring.
