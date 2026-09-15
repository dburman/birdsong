.PHONY: check test run docker-build models

## fmt + clippy + tests (what CI runs)
check:
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings
	cargo test --release

test:
	cargo test --release

## run the detector with the local config file
run:
	cargo run --release -p birdsong-server -- run --config config/birdsong.toml

## build the Raspberry Pi image and load it into the local Docker
docker-build:
	docker buildx build --platform linux/arm64 -t birdsong:latest --load .

## download and convert the BirdNET models into ./models (needs Docker)
models:
	scripts/fetch-models.sh
