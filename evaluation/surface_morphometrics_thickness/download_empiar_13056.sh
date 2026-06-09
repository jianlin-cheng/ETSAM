#!/usr/bin/env bash
# Download the EMPIAR-13056 MEF data (tomograms + voxel segmentations) into the
# directory layout that config_mef_newpipe.yml / config_etsam_newpipe.yml and
# run_thickness_newpipe.sh expect.
#
# Produces (under DATA_ROOT, default .../data/empiar_13056/mef):
#     tomograms/               raw 9.98 A/px tomograms   ({tomo}.mrc_9.98Apx.mrc)   x12
#     segmentations/           all paper voxel labels    (*.labels.mrc)             x24
#     segmentations_mito_er/   the *Mito*  label volumes (copied from segmentations) x15
#     segmentations_other/     the *otherOrg* volumes    (copied from segmentations)  x9
#
# Source (EBI public FTP-over-HTTPS, no credentials needed):
#     https://ftp.ebi.ac.uk/empiar/world_availability/13056/data/morphometrics_thickness/
#       Tomogram_bin6/        -> 9.98 A/px tomograms  (the thickness tomograms)
#       Voxel_segmentations/  -> *.labels.mrc
#
# Idempotent: a file already present (non-empty) is skipped; partial downloads are
# resumed with `wget -c`. NEVER deletes anything.
#
# Usage:
#     bash download_empiar_13056.sh [DATA_ROOT]
set -uo pipefail

DATA_ROOT="${1:-/home/joel/bml_drive/surface_morphometrics/data/empiar_13056/mef}"
BASE="https://ftp.ebi.ac.uk/empiar/world_availability/13056/data/morphometrics_thickness"
CONN="${CONN:-16}"   # parallel byte-range chunks per file (set CONN=1 for single stream)

# Stop ALL in-flight downloads on Ctrl+C / kill. Background `curl ... &` chunks ignore
# the terminal's SIGINT in a non-interactive shell (POSIX), so they keep running
# orphaned after the script dies unless we kill them explicitly.
DL_PIDS=()
cleanup() {
    trap - INT TERM EXIT
    echo ""; echo "Interrupted — stopping all downloads..."
    # polite stop: SIGTERM tracked chunk PIDs + every child of this script
    [ "${#DL_PIDS[@]}" -gt 0 ] && kill "${DL_PIDS[@]}" 2>/dev/null
    pkill -P $$ 2>/dev/null
    sleep 1
    # force-kill anything still alive (curl mid-retry can ignore the first TERM)
    [ "${#DL_PIDS[@]}" -gt 0 ] && kill -9 "${DL_PIDS[@]}" 2>/dev/null
    pkill -9 -P $$ 2>/dev/null
    wait 2>/dev/null
    exit 130
}
trap cleanup INT TERM

TOMO_DIR="$DATA_ROOT/tomograms"
SEG_DIR="$DATA_ROOT/segmentations"
MITO_DIR="$DATA_ROOT/segmentations_mito_er"
OTHER_DIR="$DATA_ROOT/segmentations_other"
mkdir -p "$TOMO_DIR" "$SEG_DIR" "$MITO_DIR" "$OTHER_DIR"

echo "============================================================"
echo "EMPIAR-13056 MEF download"
echo "  data_root : $DATA_ROOT"
echo "  source    : $BASE"
echo "============================================================"

# List the *.mrc files in a remote EMPIAR directory (parses the HTML index).
list_remote() {
    curl -s -L "$1/" | grep -oE 'href="[^"?/][^"]*\.mrc"' | sed 's/href=//; s/"//g'
}

# Download one file using CONN parallel byte-range chunks, then merge them in order.
# Skips if already complete; falls back to a single resumable stream when the server
# does not advertise range support. A chunk already fully present is reused (coarse
# resume), so re-running after an interruption only fetches the missing pieces. The
# transient *.partN chunks are removed only after a verified merge.
fetch() {
    local url="$1" dest="$2" conns="$CONN"
    [ -s "$dest" ] && { echo "  SKIP (exists): $(basename "$dest")"; return 0; }

    # Probe final size + range support (last headers win, after any redirects).
    local hdr size accept
    hdr="$(curl -sIL "$url")"
    size="$(awk 'tolower($1)=="content-length:"{v=$2} END{gsub(/\r/,"",v); print v+0}' <<<"$hdr")"
    accept="$(awk 'tolower($1)=="accept-ranges:"{v=$2} END{gsub(/\r/,"",v); print tolower(v)}' <<<"$hdr")"

    # Fall back to a single resumable stream if multipart is not possible.
    if [ "$conns" -le 1 ] || [ "$size" -le 0 ] || [ "$accept" != "bytes" ]; then
        echo "  GET  $(basename "$dest")  [single stream]"
        curl -s -L -C - -o "$dest.part" "$url" && mv -f "$dest.part" "$dest" \
            || { echo "  FAILED (resume next run): $(basename "$dest")"; return 1; }
        return 0
    fi

    echo "  GET  $(basename "$dest")  [$conns parts, $((size/1024/1024)) MB]"
    local chunk=$(( (size + conns - 1) / conns ))
    local i start end part nparts=0 pids=()
    for (( i=0; i<conns; i++ )); do
        start=$(( i * chunk )); [ "$start" -ge "$size" ] && break
        end=$(( start + chunk - 1 )); [ "$end" -ge "$size" ] && end=$(( size - 1 ))
        nparts=$(( i + 1 )); part="$dest.part$i"
        # reuse a chunk that is already exactly the right size
        if [ -f "$part" ] && [ "$(stat -c%s "$part" 2>/dev/null || echo 0)" -eq $(( end - start + 1 )) ]; then
            continue
        fi
        curl -s -L --fail --retry 3 --retry-delay 2 --range "$start-$end" -o "$part" "$url" &
        pids+=( "$!" ); DL_PIDS+=( "$!" )      # track globally so the trap can kill them
    done
    local ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
    DL_PIDS=()                                  # this file's chunks are done; clear tracker
    [ "$ok" -eq 1 ] || { echo "  a chunk failed — re-run to resume: $(basename "$dest")"; return 1; }

    # Merge the chunks in order and verify the total before committing.
    local merge=(); for (( i=0; i<nparts; i++ )); do merge+=( "$dest.part$i" ); done
    cat "${merge[@]}" > "$dest.merged"
    local got; got="$(stat -c%s "$dest.merged" 2>/dev/null || echo 0)"
    if [ "$got" -eq "$size" ]; then
        mv -f "$dest.merged" "$dest"
        rm -f "${merge[@]}"          # transient chunks, now merged into $dest
    else
        echo "  merge size mismatch ($got != $size) — re-run to resume: $(basename "$dest")"
        mv -f "$dest.merged" "$dest.merged.bad"   # keep for inspection; never rm data
        return 1
    fi
}

# ---- 1. Tomograms (9.98 A/px = Tomogram_bin6) ----
echo ""; echo "[1/3] Tomograms -> $TOMO_DIR"
for name in $(list_remote "$BASE/Tomogram_bin6"); do
    fetch "$BASE/Tomogram_bin6/$name" "$TOMO_DIR/$name"
done
echo "      tomograms present: $(ls "$TOMO_DIR"/*.mrc 2>/dev/null | wc -l)"

# ---- 2. Voxel segmentations ----
echo ""; echo "[2/3] Voxel segmentations -> $SEG_DIR"
for name in $(list_remote "$BASE/Voxel_segmentations"); do
    fetch "$BASE/Voxel_segmentations/$name" "$SEG_DIR/$name"
done
echo "      segmentations present: $(ls "$SEG_DIR"/*.labels.mrc 2>/dev/null | wc -l)"

# ---- 3. Split segmentations into mito_er / other (copy, never symlink/move) ----
# The pipeline reads *Mito* volumes from segmentations_mito_er/ and *otherOrg*
# (vesicle oracle) from segmentations_other/. We copy so the originals stay intact.
echo ""; echo "[3/3] Splitting segmentations (cp)"
for seg in "$SEG_DIR"/*.labels.mrc; do
    [ -e "$seg" ] || continue
    name="$(basename "$seg")"
    case "$name" in
        *otherOrg*) dest="$OTHER_DIR/$name" ;;
        *Mito*)     dest="$MITO_DIR/$name" ;;
        *)          continue ;;   # ignore anything unexpected
    esac
    [ -s "$dest" ] || cp "$seg" "$dest"
done
echo "      segmentations_mito_er: $(ls "$MITO_DIR"/*.labels.mrc 2>/dev/null | wc -l)"
echo "      segmentations_other:   $(ls "$OTHER_DIR"/*.labels.mrc 2>/dev/null | wc -l)"

echo ""
echo "=== DONE. Ready for:"
echo "    - run_thickness_newpipe.sh config_mef_newpipe.yml   (paper segmentations)"
echo "    - prepare_etsam_segmentations.sh                    (then config_etsam_newpipe.yml)"
