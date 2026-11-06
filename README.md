# Birdsong

Birdsong turns a Raspberry Pi and a USB microphone into an always-on wildlife listening station.
It identifies birds, frogs, insects and mammals by their calls, keeps the audio of every detection,
and gives you a dashboard to browse and review what it heard and an API to take the data elsewhere.
It is one Rust binary; besides the model files it needs only `ffmpeg`.

- **More than birds.** [Google Perch v2](https://www.kaggle.com/models/google/bird-vocalization-classifier)
  recognises about 14 600 species of birds, frogs, insects and mammals. The
  [BirdNET Geomodel](https://github.com/birdnet-team/geomodel) narrows them to what is expected at
  your location for the time of year, so a koala is not reported in Minnesota. BirdNET V2.4 is
  available too, one setting away.
- **The whole soundscape.** About 200 non-animal sounds (engines, rain, music, sirens) are
  recognised and kept apart from the animals as sound events, so they explain a quiet morning
  instead of crowding the bird charts, and double as a record of noise.
- **Learns from your listening.** Mark detections ✓ or ✗ in the dashboard and Birdsong reports
  each species' precision on your station, with a suggested threshold for it. Repeat confirmation
  stops one-off false positives, and dynamic thresholds pick up the quieter calls of a bird that
  is clearly nearby.
- **Keeps conversations private.** When the classifier hears someone speaking, that window and the
  ones either side of it are dropped, so no clip of a conversation is ever saved. With Perch only
  voices trigger it: field measurements showed its footstep and hand classes firing on leaves and
  rain. A privacy log shows what the filter heard.
- **Built to run unattended.** It follows the microphone's clock drift a sample at a time (the
  first station's USB microphone ran 140 ppm slow), keeps audio for a rolling window by age and
  total size as lossless FLAC, and reports its health and Prometheus metrics. It installs as a
  systemd service or runs in Docker.
- **A dashboard** with the latest detections, activity by hour for every species, audio clips and
  spectrograms, and links to Cornell Lab's [All About Birds](https://www.allaboutbirds.org) (birds)
  and [iNaturalist](https://www.inaturalist.org) (other animals).
- **Easy to integrate.** Read detections with a cursor or a live Server-Sent Events stream, and
  fetch chart data; see [docs/API.md](docs/API.md). Birds can also be uploaded to
  [BirdWeather](https://app.birdweather.com).
- **Small and safe.** Pure Rust, `unsafe` forbidden in every crate, no FFI. The models run in
  [tract](https://github.com/sonos/tract), and the dashboard is compiled into the binary with no
  third-party JavaScript.

## Quick start on a Raspberry Pi

A Raspberry Pi 4 or 5 with 2 GB or more running 64-bit Raspberry Pi OS (bookworm or later), and a
USB microphone. Download the `aarch64` archive from the
[releases page](https://github.com/dburman/birdsong/releases), then on the Pi:

```bash
tar xzf birdsong-v0.2.0-aarch64-unknown-linux-gnu.tar.gz && cd birdsong-v0.2.0-aarch64-unknown-linux-gnu
```

```bash
./install-pi.sh --lat 42.36 --lon -71.06
```

The installer finds the microphone, writes `/etc/birdsong/birdsong.toml`, downloads the Perch and
Geomodel files (430 MB), and starts the `birdsong` service. The dashboard is then at
`http://<your-pi>.local:8080`. Docker works too; both are described in
[docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).

## How it performs

Measured on a Raspberry Pi 4 Model B (8 GB), one core:

| Classifier | Per analysis window | Faster than real time | Memory |
|------------|--------------------:|----------------------:|-------:|
| BirdNET V2.4 | 224 ms per 3 s | 13× | 209 MiB |
| Perch v2, regional (999 classes) | 1.7 s per 5 s | 2.9× | 245 MiB |
| Perch v2, full (14 795 classes) | 2.7 s per 5 s | 1.9× | 1.1 GiB |

The BirdNET classifier is checked against the reference implementation. On 317 clips saved by a
[BirdNET-Pi](https://github.com/Nachtzuster/BirdNET-Pi) station running on the same Pi, Birdsong
reported the same species on every clip (median confidence difference 0.0004). Over 24 hours side
by side, the two recorded nearly the same number of detections per species (Red-breasted Nuthatch
251 vs 233, Hairy Woodpecker 84 vs 82). Perch reports several times more. Whether the extra ones
are quiet calls BirdNET misses or false positives still needs listening, and that is what the
dashboard's reviews are for. Details in [docs/MODEL.md](docs/MODEL.md).

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
[BirdNET-Go](https://github.com/tphakala/birdnet-go), the stations that showed what this kind of
project can be and that Birdsong's BirdNET mode is checked against; the
[BirdNET](https://birdnet.cornell.edu) team for BirdNET and the Geomodel; Google
Research for Perch; and [tract](https://github.com/sonos/tract) for running them in Rust.
