#!/usr/bin/env bash
# Install or upgrade Birdsong as a systemd service on a Raspberry Pi (or any systemd Linux), without
# Docker. Run it as the user the service should run as (a member of the `audio` group); it uses sudo
# for the system parts. Running it again upgrades the binary and keeps the configuration.
#
#   ./install-pi.sh [options]
#
#   --binary PATH     birdsong executable to install (default: ./birdsong next to this script, as in
#                     a release archive; from a repository checkout build it and pass it here)
#   --config FILE     configuration to install as /etc/birdsong/birdsong.toml (default: keep the
#                     installed one, or write a new one from the options below)
#   --env FILE        secrets to install as /etc/birdsong/birdsong.env, e.g. the BirdWeather token as
#                     BIRDSONG__BIRDWEATHER__TOKEN=... (default: keep the installed one)
#   --lat LAT --lon LON --timezone ZONE --port PORT
#                     station settings for a newly written configuration
#   --no-models       do not download missing Perch / Geomodel files
#
# Layout: /usr/local/bin/birdsong, /etc/birdsong/{birdsong.toml,birdsong.env},
# /var/lib/birdsong/{data,models} (for a newly written configuration), service `birdsong`.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
binary="" config="" env_file="" lat="" lon="" timezone="" port=8080 fetch_models=1
while [ $# -gt 0 ]; do
  case "$1" in
    --binary) binary=$2; shift 2 ;;
    --config) config=$2; shift 2 ;;
    --env) env_file=$2; shift 2 ;;
    --lat) lat=$2; shift 2 ;;
    --lon) lon=$2; shift 2 ;;
    --timezone) timezone=$2; shift 2 ;;
    --port) port=$2; shift 2 ;;
    --no-models) fetch_models=0; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say() { printf '== %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
find_file() { for p in "$@"; do [ -f "$p" ] && { echo "$p"; return 0; }; done; return 1; }

user=$(id -un)
[ "$user" != root ] || die "run as the user the service should run as, not root (it uses sudo)"
command -v systemctl >/dev/null || die "systemd is required"
state_dir=/var/lib/birdsong
conf_dir=/etc/birdsong

[ -n "$binary" ] || binary="$here/birdsong"
[ -x "$binary" ] || die "no birdsong executable at $binary (pass --binary)"
"$binary" --version >/dev/null 2>&1 || die "$binary does not run on this machine (wrong architecture?)"
template=$(find_file "$here/birdsong.service.in" "$here/../deploy/birdsong.service.in") \
  || die "birdsong.service.in not found next to this script or in ../deploy"

say "checking prerequisites"
# Passwordless sudo needs no prompt; otherwise ask once, up front.
sudo -n true 2>/dev/null || sudo -v
if ! command -v ffmpeg >/dev/null; then
  say "installing ffmpeg (audio capture)"
  sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
fi
id -nG "$user" | tr ' ' '\n' | grep -qx audio \
  || echo "warning: $user is not in the audio group; add it with: sudo usermod -aG audio $user"

say "installing $("$binary" --version) to /usr/local/bin/birdsong"
was_running=0
systemctl is-active --quiet birdsong && was_running=1
sudo install -m 0755 "$binary" /usr/local/bin/birdsong.new
sudo mv /usr/local/bin/birdsong.new /usr/local/bin/birdsong
sudo install -d -m 0755 "$conf_dir"
sudo install -d -o "$user" -g "$(id -gn)" "$state_dir"

stamp=$(date +%Y%m%d-%H%M%S)
if [ -n "$config" ]; then
  [ -f "$config" ] || die "no configuration at $config"
  [ -f "$conf_dir/birdsong.toml" ] && sudo cp "$conf_dir/birdsong.toml" "$conf_dir/birdsong.toml.$stamp"
  sudo install -m 0644 "$config" "$conf_dir/birdsong.toml"
  say "installed configuration from $config"
elif [ ! -f "$conf_dir/birdsong.toml" ]; then
  device=$(/usr/local/bin/birdsong devices | awk '/^plughw:/ {print $1; exit}')
  [ -n "$device" ] || die "no microphone found (birdsong devices); connect one or pass --config"
  [ -n "$timezone" ] || timezone=$(timedatectl show -p Timezone --value 2>/dev/null || echo UTC)
  if [ -z "$lat" ] || [ -z "$lon" ]; then
    echo "warning: no --lat/--lon: the location filter is off until you set station latitude/longitude"
    lat=0.0 lon=0.0
  fi
  install -d -m 0755 "$state_dir/data" "$state_dir/models"
  tmp=$(mktemp)
  cat > "$tmp" <<EOF
# Written by install-pi.sh. See birdsong.example.toml for every setting.
[station]
name = "Birdsong"
latitude = $lat
longitude = $lon
timezone = "$timezone"

[[audio.sources]]
id = "mic0"
kind = "alsa"
device = "$device"

[model]
dir = "$state_dir/models"
kind = "perch-v2"                  # "birdnet-v2.4" for BirdNET (needs scripts/fetch-models.sh)

[storage]
data_dir = "$state_dir/data"

[server]
bind = "0.0.0.0:$port"
EOF
  sudo install -m 0644 "$tmp" "$conf_dir/birdsong.toml"
  rm -f "$tmp"
  say "wrote $conf_dir/birdsong.toml (microphone $device, time zone $timezone)"
else
  say "keeping $conf_dir/birdsong.toml"
fi

if [ -n "$env_file" ]; then
  [ -f "$env_file" ] || die "no file at $env_file"
  sudo install -m 0600 -o root -g root "$env_file" "$conf_dir/birdsong.env"
  say "installed secrets from $env_file (readable by root only)"
elif ! sudo test -f "$conf_dir/birdsong.env"; then
  printf '# Secrets for Birdsong, read by systemd.\n# BIRDSONG__BIRDWEATHER__TOKEN=your-station-token\n' \
    | sudo install -m 0600 -o root -g root /dev/stdin "$conf_dir/birdsong.env"
fi

say "checking the configuration"
effective=$(/usr/local/bin/birdsong check-config --config "$conf_dir/birdsong.toml" 2>/dev/null) \
  || { /usr/local/bin/birdsong check-config --config "$conf_dir/birdsong.toml" >/dev/null; die "invalid configuration"; }
value() { awk -v section="[$1]" -v key="$2" '$0 == section {s = 1; next} /^\[/ {s = 0} s && $1 == key {gsub(/"/, "", $3); print $3; exit}' <<<"$effective"; }
models_dir=$(value model dir)
kind=$(value model kind)
bind=$(value server bind)
port=${bind##*:}

if [ "$kind" = perch-v2 ] && [ "$fetch_models" = 1 ]; then
  classifier=$(value model classifier)
  meta=$(value model meta_model)
  if [ ! -f "$models_dir/$classifier" ] || { [ -n "$meta" ] && [ ! -f "$models_dir/$meta" ]; }; then
    fetch_perch=$(find_file "$here/fetch-perch.sh") || die "fetch-perch.sh not found next to this script"
    fetch_geo=$(find_file "$here/fetch-geomodel.sh") || die "fetch-geomodel.sh not found next to this script"
    say "downloading models into $models_dir (about 430 MB)"
    MODELS_DIR="$models_dir" "$fetch_perch" >/dev/null
    MODELS_DIR="$models_dir" "$fetch_geo" >/dev/null
  fi
elif [ "$kind" = birdnet-v2.4 ] && [ ! -f "$models_dir/$(value model classifier)" ]; then
  echo "warning: BirdNET model files are missing from $models_dir; run scripts/fetch-models.sh (needs Docker)"
fi

say "installing the birdsong service"
sed -e "s|@USER@|$user|g" -e "s|@STATE_DIR@|$state_dir|g" "$template" \
  | sudo tee /etc/systemd/system/birdsong.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --quiet birdsong
if [ "$was_running" = 1 ]; then sudo systemctl restart birdsong; else sudo systemctl start birdsong; fi

say "waiting for the API on port $port"
for _ in $(seq 1 45); do
  if /usr/local/bin/birdsong healthcheck --url "http://127.0.0.1:$port/api/v1/health" --timeout-secs 2 >/dev/null 2>&1; then
    say "Birdsong is running: http://$(hostname).local:$port"
    echo "   logs:           journalctl -u birdsong -f"
    echo "   configuration:  $conf_dir/birdsong.toml (then: sudo systemctl restart birdsong)"
    echo "   secrets:        $conf_dir/birdsong.env"
    exit 0
  fi
  sleep 2
done
die "the service did not answer within 90 s; see: journalctl -u birdsong -n 50"
