# syntax=docker/dockerfile:1.7
#
# Birdsong image for Raspberry Pi (linux/arm64) and x86-64 (linux/amd64).
#
#   docker buildx build --platform linux/arm64 -t birdsong:latest --load .
#
# The Rust build stage always runs on the build machine's own architecture and cross-compiles for
# the target, so building the arm64 image on an x86-64 laptop does not emulate the compiler.
# Models are not part of the image (size and CC BY-NC-SA licence): mount them at /models.

FROM --platform=$BUILDPLATFORM rust:1-bookworm AS build
ARG TARGETARCH
ARG BUILDARCH
WORKDIR /src

RUN set -eux; \
    case "$TARGETARCH" in \
      arm64) triple=aarch64-unknown-linux-gnu; cross_cc=aarch64-linux-gnu-gcc; pkgs="gcc-aarch64-linux-gnu libc6-dev-arm64-cross" ;; \
      amd64) triple=x86_64-unknown-linux-gnu;  cross_cc=x86_64-linux-gnu-gcc;  pkgs="gcc-x86-64-linux-gnu libc6-dev-amd64-cross" ;; \
      *) echo "unsupported TARGETARCH=$TARGETARCH" >&2; exit 1 ;; \
    esac; \
    if [ "$TARGETARCH" = "$BUILDARCH" ]; then \
      cc=gcc; \
    else \
      apt-get update; \
      apt-get install -y --no-install-recommends $pkgs; \
      rm -rf /var/lib/apt/lists/*; \
      cc=$cross_cc; \
    fi; \
    rustup target add "$triple"; \
    printf '[target.%s]\nlinker = "%s"\n' "$triple" "$cc" >> /usr/local/cargo/config.toml; \
    echo "$triple" > /target-triple; \
    echo "$cc" > /target-cc

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY static ./static

RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    set -eux; \
    triple=$(cat /target-triple); \
    export "CC_$(echo "$triple" | tr '-' '_')=$(cat /target-cc)"; \
    cargo build --release --locked --target "$triple" -p birdsong-server --bin birdsong; \
    cp "target/$triple/release/birdsong" /birdsong

FROM debian:bookworm-slim
LABEL org.opencontainers.image.title="birdsong" \
      org.opencontainers.image.description="Bird sound detection (BirdNET V2.4) with an HTTP API and dashboard" \
      org.opencontainers.image.source="https://github.com/dburman/birdsong" \
      org.opencontainers.image.licenses="MIT"

# ffmpeg >= 5.0 captures ALSA devices and streams (Debian bookworm ships 5.1).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    useradd --uid 1000 --user-group --groups audio --home-dir /data --no-create-home --shell /usr/sbin/nologin birdsong; \
    mkdir -p /data /models /config; \
    chown birdsong:birdsong /data

COPY --from=build /birdsong /usr/local/bin/birdsong
COPY config/birdsong.example.toml /usr/share/birdsong/birdsong.example.toml

USER birdsong
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD ["birdsong", "healthcheck"]
ENTRYPOINT ["birdsong"]
CMD ["run", "--config", "/config/birdsong.toml"]
