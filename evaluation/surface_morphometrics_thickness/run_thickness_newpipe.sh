#!/usr/bin/env bash
# Thickness-only reproduction on the NEW configurable pipeline (PR #56), FRESH run
# using the repo-DEFAULT config values (isotropic remesh, octree 9, radius_hit 9).
#
# Pipeline (refinement / distances / subcompartments / ATP / vesicles SKIPPED):
#     segmentation_to_meshes -> run_pycurv -> sample_density -> measure_thickness
# Only OMM/IMM/ER (set by the config's thickness_measurements.components and
# segmentation_values). *Vesicle* segmentation files are skipped by name.
#
# Unlike the previous (paper-param) driver, this does NOT reuse existing graphs:
# the default params change the mesh AND radius_hit (rh9), so everything is rebuilt.
# Idempotent: every step skips work whose final artefact already exists. NEVER deletes.
#
# Usage:
#   bash run_thickness_newpipe.sh <config.yml>
set -uo pipefail

PY=/home/joel/personal/miniforge3/envs/morphometrics/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CFG="$1"
ORGS=(OMM IMM ER)
MAX_PARALLEL=1    # SEQUENTIAL. pycurv forks 'cores' (12) worker procs, each ~2GB RSS on
                  # the big isotropic rh9 meshes -> ~24GB peak, safe on this 31GB box.
                  # Running 2 jobs (or cores=20) OOM-thrashed the machine; do NOT raise.

read -r SEG_DIR TOMO_DIR WORK_DIR RH < <("$PY" - "$CFG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
rh = c.get("curvature_measurements", {}).get("radius_hit", 9)
print(c["seg_dir"], c["tomo_dir"], c["work_dir"], rh)
PY
)
SEG_DIR="${SEG_DIR%/}"; WORK_DIR="${WORK_DIR%/}"
mkdir -p "$WORK_DIR"

echo "============================================================"
echo "NEW-PIPELINE thickness — DEFAULT params, FRESH run"
echo "  config    : $CFG"
echo "  seg_dir   : $SEG_DIR"
echo "  tomo_dir  : $TOMO_DIR"
echo "  work_dir  : $WORK_DIR"
echo "  radius_hit: $RH ; organelles: ${ORGS[*]} ; parallel: $MAX_PARALLEL"
echo "============================================================"

wait_for_slot() { local max=$1; while [ "$(jobs -rp 2>/dev/null | wc -l)" -ge "$max" ]; do sleep 3; done; }

# ---- Step 1: surface generation (per non-vesicle seg; skip if surfaces present) ----
echo ""; echo "[1/4] segmentation_to_meshes ..."
for seg in "$SEG_DIR"/*.labels.mrc; do
    [ -e "$seg" ] || continue
    case "$seg" in *Vesicle*) continue;; esac
    base="$(basename "$seg" .mrc)"
    if ls "$WORK_DIR/${base}_"*.surface.vtp >/dev/null 2>&1; then
        echo "      SKIP meshing (surfaces exist): $base"
    else
        echo "      meshing: $base"
        "$PY" segmentation_to_meshes.py "$CFG" "$(basename "$seg")" \
            >> "$WORK_DIR/segmentation_to_meshes.log" 2>&1 \
            || echo "      (meshing reported an issue for $base — see log)"
    fi
done
echo "      surface.vtp files: $(ls "$WORK_DIR"/*.surface.vtp 2>/dev/null | wc -l)"

# ---- Step 2: pycurv curvature (parallel across surfaces; skip if .gt present) ----
echo ""; echo "[2/4] run_pycurv (rh$RH, $MAX_PARALLEL parallel) ..."
launched=0
for vtp in "$WORK_DIR"/*.surface.vtp; do
    [ -e "$vtp" ] || continue
    base="$(basename "$vtp" .surface.vtp)"
    case "$base" in
        *_OMM|*_IMM|*_ER) : ;;     # only the three organelles
        *) continue;;
    esac
    [ -f "$WORK_DIR/${base}.AVV_rh${RH}.gt" ] && continue
    wait_for_slot "$MAX_PARALLEL"
    echo "      pycurv: $base"
    "$PY" run_pycurv.py -f "$CFG" "${base}.surface.vtp" > "$WORK_DIR/${base}_pycurv.log" 2>&1 &
    launched=$((launched+1))
done
[ "$launched" -gt 0 ] && { echo "      waiting for $launched pycurv job(s)..."; wait; }
echo "      AVV_rh$RH.gt graphs: $(ls "$WORK_DIR"/*.AVV_rh${RH}.gt 2>/dev/null | grep -vc refined)"

# ---- Step 3: density sampling (skip if already complete) ----
echo ""; echo "[3/4] sample_density.py ..."
n_gt=$(ls "$WORK_DIR"/*.AVV_rh"${RH}".gt 2>/dev/null | grep -vc refined)
n_samp=$(ls "$WORK_DIR"/*.AVV_rh"${RH}"_sampling.csv 2>/dev/null | wc -l)
if [ "$n_gt" -gt 0 ] && [ "$n_samp" -ge "$n_gt" ]; then
    echo "      SKIP ($n_samp sampling CSVs >= $n_gt graphs)"
else
    "$PY" sample_density.py "$CFG" 2>&1 | tail -15
fi
echo "      sampling CSVs: $(ls "$WORK_DIR"/*_sampling.csv 2>/dev/null | wc -l)"

# ---- Step 4: thickness measurement ----
echo ""; echo "[4/4] measure_thickness.py ..."
"$PY" measure_thickness.py "$CFG" 2>&1 | tail -25

echo ""
echo "=== DONE: $CFG"
echo "    component_list: $WORK_DIR/component_list.csv"
echo "    refined graphs: $(ls "$WORK_DIR"/*_refined.gt 2>/dev/null | wc -l)"
