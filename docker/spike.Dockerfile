# Step 0 spike: compile the golden test for linux/arm64 and run it against mounted models/fixtures.
FROM rust:1-bookworm AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates
RUN cargo test --release -p birdsong-model --test golden_v24 --no-run 2>&1 | tail -3
CMD ["cargo", "test", "--release", "-p", "birdsong-model", "--test", "golden_v24", "--", "--nocapture"]
