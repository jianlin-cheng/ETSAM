#!/usr/bin/env bash
set -euo pipefail

# Download the ETSAM stage 1 / stage 2 weights into the bind-mounted checkpoints/
# directory. Runs once at first container-up; files already present are reused, so
# the weights are fetched only once and persist on the host across rebuilds.
#
# Uses aria2 to fetch the files in parallel, each split into multiple connections
# (multipart) so large weights download quickly. Partial downloads resume.
mkdir -p checkpoints
base="https://zenodo.org/records/17571925/files"
files=(etsam_stage1_v1.pt etsam_stage2_v1.pt)

# Build an aria2 input list containing only the missing files.
list="$(mktemp)"
trap 'rm -f "$list"' EXIT
missing=0
for f in "${files[@]}"; do
  if [[ -f "checkpoints/$f" ]]; then
    echo "checkpoints/$f already present, skipping"
  else
    printf '%s\n  out=%s\n' "$base/$f" "$f" >> "$list"
    missing=$((missing + 1))
  fi
done

if [[ "$missing" -gt 0 ]]; then
  echo "Downloading $missing checkpoint(s) with aria2 (parallel, multi-connection)..."
  aria2c \
    --dir=checkpoints \
    --input-file="$list" \
    --max-concurrent-downloads=2 \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --continue=true \
    --auto-file-renaming=false \
    --console-log-level=warn \
    --summary-interval=0
fi

echo "ETSAM ready!"
