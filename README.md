# Birdsong

Birdsong listens to a microphone, recognises birds and other animals by their sounds, and keeps a
record you can browse, play back and query. It is a from-scratch Rust replacement for the bird
detection of [BirdNET-Pi](https://github.com/Nachtzuster/BirdNET-Pi), built to run around the clock
on a Raspberry Pi.

- **Two classifiers.** [Google Perch v2](https://www.kaggle.com/models/google/bird-vocalization-classifier)
  (default: about 14 600 species of birds, frogs, insects and mammals) or BirdNET V2.4 (the model
  BirdNET-Pi uses). One setting switches between them.
- **Location-aware.** The [BirdNET Geomodel](https://github.com/birdnet-team/geomodel) keeps only
  species expected at the station for the time of year, so a koala is not reported in Minnesota.
- **A dashboard** with the latest birds, activity by hour for every species, recent detections with
  audio clips and spectrograms, links to Cornell Lab's
  [All About Birds](https://www.allaboutbirds.org) (birds) and [iNaturalist](https://www.inaturalist.org)
  (other animals), and ✓ / ✗ buttons to review detections.
- **Tunable.** Repeat confirmation, dynamic thresholds, per-species thresholds suggested from your
  reviews, a human-voice privacy filter, and non-animal sounds (engines, rain, music) kept apart as
  sound events.
- **An HTTP API** for ingesting detections with a cursor, a live event stream, charts data and
  Prometheus metrics. See [docs/API.md](docs/API.md).
- **Audio is kept for a rolling window** you configure (age and total size), as lossless FLAC.
- **[BirdWeather](https://app.birdweather.com) uploads**, birds only.
- **Pure Rust**, `unsafe` forbidden in every crate, no FFI: models run in
  [tract](https://github.com/sonos/tract); audio capture uses the `ffmpeg` executable.

## Quick start on a Raspberry Pi

A Raspberry Pi 4 or 5 with 2 GB or more running 64-bit Raspberry Pi OS (bookworm or later), and a
USB microphone. Download the `aarch64` archive from the
[releases page](https://github.com/dburman/birdsong/releases), then on the Pi:

```bash
tar xzf birdsong-v0.1.0-aarch64-unknown-linux-gnu.tar.gz && cd birdsong-v0.1.0-aarch64-unknown-linux-gnu
```

```bash
./install-pi.sh --lat 42.36 --lon -71.06
```

The installer finds the microphone, writes `/etc/birdsong/birdsong.toml`, downloads the Perch and
Geomodel files (430 MB), and starts the `birdsong` service. The dashboard is then at
`http://<your-pi>.local:8080`. Docker works too; both are described in
[docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).

## How it performs

Measured on a Raspberry Pi 4 Model B (8 GB), one core, next to a running BirdNET-Pi station:

| Classifier | Per analysis window | Faster than real time | Memory |
|------------|--------------------:|----------------------:|-------:|
| BirdNET V2.4 | 224 ms per 3 s | 13× | 209 MiB |
| Perch v2, regional (999 classes) | 1.7 s per 5 s | 2.9× | 245 MiB |
| Perch v2, full (14 795 classes) | 2.7 s per 5 s | 1.9× | 1.1 GiB |

On 317 clips saved by that BirdNET-Pi station, Birdsong's BirdNET reported the same species on
every one (median confidence difference 0.0004). Running side by side for 24 hours, the two
recorded nearly the same number of detections per species (Red-breasted Nuthatch 251 vs 233, Hairy
Woodpecker 84 vs 82). Perch reports several times more; whether the extra ones are quiet calls
BirdNET misses or false positives still needs listening. Details in [docs/MODEL.md](docs/MODEL.md).

## Models and licences

Birdsong is MIT-licensed. The models are separate downloads with their own licences, and none are
included in the repository, releases or Docker image:

| Model | Licence | Download |
|-------|---------|----------|
| Perch v2 (Google Research; ONNX conversion by justinchuby and tphakala) | Apache-2.0 | `scripts/fetch-perch.sh` |
| BirdNET Geomodel v3.0.4 (BirdNET team) | Apache-2.0 | `scripts/fetch-geomodel.sh` |
| BirdNET V2.4 (Cornell Lab of Ornithology, Chemnitz University of Technology) | CC BY-NC-SA 4.0, non-commercial | `scripts/fetch-models.sh` (needs Docker to convert) |

## Documentation

| | |
|---|---|
| [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md) | Installing and running on a Pi: service or Docker, microphones, BirdWeather, monitoring, backups |
| [docs/API.md](docs/API.md) | The HTTP API and metrics |
| [docs/MODEL.md](docs/MODEL.md) | The models: provenance, checksums, verification, benchmarks, comparisons with BirdNET-Pi |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Why things are the way they are |
| [config/birdsong.example.toml](config/birdsong.example.toml) | Every setting, commented |
| [BUILD_PLAN.md](BUILD_PLAN.md) | The original build plan, with model candidates and ideas for other animals |

## Building from source

Rust (see `rust-toolchain.toml`) and `ffmpeg`:

```bash
cargo build --release
```

```bash
cargo test
```

Tests that need model files skip themselves when `models/` is empty; fetch the models first to run
them. The binary for a Pi, built in Debian bookworm with Docker from any machine:

```bash
docker buildx build --platform linux/arm64 --target binary --output type=local,dest=out .
```

## Acknowledgements

[BirdNET-Pi](https://github.com/Nachtzuster/BirdNET-Pi) and
[BirdNET-Go](https://github.com/tphakala/birdnet-go), whose behaviour Birdsong follows and compares
against; the [BirdNET](https://birdnet.cornell.edu) team for BirdNET and the Geomodel; Google
Research for Perch; and [tract](https://github.com/sonos/tract) for running them in Rust.
