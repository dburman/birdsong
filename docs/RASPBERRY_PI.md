# Running Birdsong on a Raspberry Pi

Birdsong runs as one Docker container: it records from a USB microphone, detects birds, keeps
clips for a rolling window, and serves the dashboard and API on port 8080.

## What you need

- Raspberry Pi 5 or 4B (2 GB RAM or more) with **64-bit** Raspberry Pi OS (Bookworm or later).
- A USB microphone or USB sound card with a microphone.
- Docker with the compose plugin: `curl -fsSL https://get.docker.com | sh`, then
  `sudo usermod -aG docker $USER` and log in again.
- Preferably a USB SSD for `data/`: clips and the database write continuously, which wears SD cards.

## 1. Get the models (once)

The BirdNET models are licensed CC BY-NC-SA 4.0 (non-commercial) and are not in the image. The
conversion step needs TensorFlow, so run it on any machine with Docker; a laptop is faster than the Pi.

```bash
git clone https://github.com/dburman/birdsong.git
cd birdsong
scripts/fetch-models.sh
```

It downloads the two official archives (Keras and TFLite, about 200 MB) from Zenodo, verifies their checksums, converts it in a throwaway
container, and checks that the converted model reproduces the reference results. Copy the resulting
`models/` directory to the Pi next to `docker-compose.yml` if you ran it elsewhere.

## 2. Build or copy the image

On the Pi itself (about 15 to 30 minutes on a Pi 5):

```bash
docker compose build
```

Or build on a faster machine and copy it over:

```bash
docker buildx build --platform linux/arm64 -t birdsong:latest --load .
docker save birdsong:latest | gzip > birdsong-arm64.tar.gz
# on the Pi:
gunzip -c birdsong-arm64.tar.gz | docker load
```

## 3. Find the microphone

```bash
arecord -l
```

```text
card 1: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
```

That is card 1, device 0. Test a 5 second recording:

```bash
arecord -D plughw:1,0 -f S16_LE -r 48000 -c 1 -d 5 test.wav && aplay test.wav
```

Card numbers can change between boots when several USB audio devices are attached; the name form
`plughw:CARD=Device,DEV=0` (the name after `card 1:`) is stable. Prefer `plughw` over `hw`: it
converts sample rate and channel count if the microphone cannot record 48 kHz mono natively.

## 4. Configure

```bash
mkdir -p config data
sudo chown 1000:1000 data
cp config/birdsong.example.toml config/birdsong.toml
nano config/birdsong.toml
```

Set at least:

```toml
[station]
name = "Backyard"
latitude = 42.36          # your location: limits detections to species expected there
longitude = -71.06
timezone = "America/New_York"

[[audio.sources]]
id = "mic0"
kind = "alsa"
device = "plughw:CARD=Device,DEV=0"
```

Keep `model.dir = "/models"` and `storage.data_dir = "/data"`; those are the paths inside the
container. Check the file before starting:

```bash
docker compose run --rm birdsong check-config --config /config/birdsong.toml
```

The rolling window for saved audio is under `[retention]`: `clip_max_age_days` (default 14) and
`clip_max_total_mb` (default 4096). Detection history is kept after clips are deleted.
Clips are saved as FLAC by default (`storage.clip_format`), about a third the size of WAV with
identical audio; the size cap counts each clip's audio and spectrogram together.

## 5. Start

```bash
docker compose up -d
docker compose logs -f
```

Within a few seconds you should see `HTTP API listening`, then `species filter updated`. Open
`http://<pi-address>:8080/` for the dashboard. Each detection logs one line:

```text
INFO birdsong_server::pipeline: detection species="Black-capped Chickadee" conf=0.75 source=mic0 id=1
```

`docker compose ps` shows `healthy` once the API answers. `docker compose down` stops it; chunks
already being analysed are finished and stored first.

## Checking that it keeps up

```bash
curl -s http://localhost:8080/api/v1/health
```

- `seconds_since_last_chunk` should stay under about 5. If it grows, audio is not arriving: check
  the device name and `docker compose logs` for ffmpeg errors.
- `mean_inference_ms` is the time to analyse one 3 second window. It must stay well under 3000.
- `chunks_dropped` should stay at 0. If it grows, the Pi cannot keep up: set
  `detection.overlap_seconds = 0`, disable `storage.spectrograms`, or use a faster Pi.

Single-stream inference on the development machine takes about 25 ms per window; expect roughly
10 to 20 times that on a Pi 4 and less on a Pi 5. These Pi figures are estimates until measured.

## Monitoring

`http://<pi-address>:8080/metrics` is in Prometheus format. A scrape job:

```yaml
scrape_configs:
  - job_name: birdsong
    static_configs:
      - targets: ["birdsong.local:8080"]
```

Useful alerts: `birdsong_seconds_since_last_chunk > 60` (audio stopped) and
`rate(birdsong_chunks_dropped_total[10m]) > 0` (the Pi cannot keep up).

## Memory use

Memory is bounded by design; nothing grows with uptime or with the number of detections.

| Part | Size |
|------|------|
| Classifier and location model weights | about 80 MB |
| Ring buffer per audio source (`audio.ring_buffer_seconds`, default 90 s of 48 kHz float) | 17 MB |
| Capture channel per source (64 frames of 100 ms) | 1.2 MB |
| Inference queue (4 chunks, all sources) | 2.3 MB |
| Detection broadcast and pending clip jobs | under 1 MB |

With two sources the container used about 230 MiB in testing, so a 1 GB Pi is enough and 2 GB
leaves room for the OS and Docker.

## Troubleshooting

| Symptom | Likely cause and fix |
|---------|----------------------|
| `ffmpeg: ... Device or resource busy` | Another program holds the microphone. Stop it, or check `fuser -v /dev/snd/*`. |
| `ffmpeg: ... No such file or directory` for the device | Wrong card name or number. Re-run `arecord -l`; use the `plughw:CARD=...` form. |
| `Permission denied` on `/dev/snd` | The container needs `devices: /dev/snd` and `group_add: audio` (both in the compose file). |
| `opening database` / permission errors on `/data` | `sudo chown -R 1000:1000 data`. |
| `loading models ... No such file` | `models/` is not next to `docker-compose.yml`, or `scripts/fetch-models.sh` was not run. |
| Dashboard shows "No audio for N s" | Same as the first two rows; the log shows ffmpeg restarting with its error. |
| Many detections of `Engine`, `Dog` or `Human` | Expected near roads and houses; exclude them with `detection.exclude_species = ["Engine"]`. |
