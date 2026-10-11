#!/usr/bin/env bash
# Build the ETSAM installer for the current platform (Linux: .sh, Windows: .exe).
# Needs rattler-build and constructor on PATH, e.g.:
#   conda create -n etsam-build -c conda-forge rattler-build constructor
#   conda activate etsam-build && installer/build.sh
# Output: installer/build/installer/ETSAM-<version>-<OS>-x86_64.{sh,exe}
set -euo pipefail
cd "$(dirname "$0")"

# Version from pyproject.toml, used for the conda package and the installer.
ETSAM_VERSION="${ETSAM_VERSION:-$(python -c "import tomllib; print(tomllib.load(open('../pyproject.toml', 'rb'))['project']['version'])")}"
export ETSAM_VERSION
echo "Building the ETSAM $ETSAM_VERSION installer"

# 1. The etsam conda package (noarch), into a local channel in a temporary folder.
rm -rf build
CHANNEL_DIR="$(mktemp -d)/channel"
# Git Bash on Windows: give native tools a Windows path.
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else echo "$1"; fi; }
CHANNEL_DIR="$(winpath "$CHANNEL_DIR")"
trap 'rm -rf "$(dirname "$CHANNEL_DIR")"' EXIT
rattler-build build --recipe recipe/recipe.yaml --output-dir "$CHANNEL_DIR" -c conda-forge
rm -rf "$CHANNEL_DIR/bld" "$CHANNEL_DIR/src_cache"

# 2. The installer, from the local channel and conda-forge.
ETSAM_CHANNEL="$(python -c "import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve().as_uri())" "$CHANNEL_DIR")"
export ETSAM_CHANNEL
mkdir -p build/channel && cp -r "$CHANNEL_DIR/." build/channel/  # keep the package for reference
# constructor caches downloaded packages; a cached etsam package from an earlier
# build has the same file name but different content, which breaks the installer.
CACHE_DIR="$(winpath "${CONSTRUCTOR_CACHE:-$HOME/.conda/constructor}")"
rm -rf "$CACHE_DIR"/*/etsam-*
constructor . --cache-dir "$CACHE_DIR" --output-dir build/installer
ls -l build/installer
