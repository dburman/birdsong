.PHONY: check test run docker-build

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

## build the Raspberry Pi image (Dockerfile arrives in Step 10; the spike image works today)
docker-build:
	docker buildx build --platform linux/arm64 -f docker/spike.Dockerfile -t birdsong-spike:arm64 .
