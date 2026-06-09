#!/usr/bin/env bash
# Run ETSAM on the EMPIAR-13056 MEF tomograms and turn the binary membrane masks
# into multi-label (OMM/IMM/ER [+Vesicle]) segmentation MRCs, in the layout that
# config_etsam_newpipe.yml / run_thickness_newpipe.sh expect.
#
# Requires the data already downloaded by download_empiar_13056.sh:
#     <DATA_ROOT>/tomograms/              {tomo}.mrc_9.98Apx.mrc
#     <DATA_ROOT>/segmentations_mito_er/  {tomo}_{Mito*_ER}.labels.mrc   (label oracle)
#     <DATA_ROOT>/segmentations_other/    {tomo}_otherOrg.labels.mrc     (vesicle oracle)
#
# Produces:
#     <DATA_ROOT>/etsam_phase2/etsam_masks/            {tomo}_etsam_predicted_post_processed_mask.mrc
#     <DATA_ROOT>/etsam_phase2/etsam_multilabel_full/  {tomo}_etsam_to_{tag}.labels.mrc  (OMM=1,IMM=2,ER=3)
#                                                      {tomo}_etsam_to_Vesicle.labels.mrc (Vesicle=4; optional)
#
# Steps:
#   1. ETSAM binary mask per tomogram                       (etsam.py,  etsam env, GPU)
#   2. Mito multi-label per *_Mito*_ER oracle               (etsam_to_multilabel.py, morphometrics env)
#   3. Vesicle label per *_otherOrg oracle (optional)       (etsam_to_vesicle.py)
# Tomogram->mito-tag mapping is DISCOVERED from the segmentations_mito_er filenames,
# so multi-mitochondria tomograms (Mito1_ER + Mito2_ER) are handled automatically.
#
# The three underlying tools are already idempotent and never delete (they rename
# suspect outputs to *.corrupted_<ts>); this driver only skips/launches them.
#
# Usage:
#     bash prepare_etsam_segmentations.sh [DATA_ROOT]
#   env knobs:  GEN_VESICLE=0  to skip vesicle generation (thickness run ignores vesicles anyway)
set -uo pipefail

DATA_ROOT="${1:-/home/joel/bml_drive/surface_morphometrics/data/empiar_13056/mef}"
GEN_VESICLE="${GEN_VESICLE:-1}"

# Tooling: ETSAM runs in its own env/repo; the classifiers run in the morphometrics env.
SM_DIR=/home/joel/bml_drive/surface_morphometrics      # holds etsam_to_*.py helpers
ETSAM_REPO=/home/joel/bml_drive/surface_morphometrics/ETSAM                   # holds etsam.py
ETSAM_PY=/home/joel/personal/miniforge3/envs/etsam/bin/python
MORPHO_PY=/home/joel/personal/miniforge3/envs/morphometrics/bin/python

TOMO_DIR="$DATA_ROOT/tomograms"
MITO_DIR="$DATA_ROOT/segmentations_mito_er"
OTHER_DIR="$DATA_ROOT/segmentations_other"
MASK_DIR="$DATA_ROOT/etsam_phase2/etsam_masks"
MULTI_DIR="$DATA_ROOT/etsam_phase2/etsam_multilabel_full"
mkdir -p "$MASK_DIR" "$MULTI_DIR"

echo "============================================================"
echo "ETSAM -> multi-label segmentation preparation"
echo "  data_root : $DATA_ROOT"
echo "  masks     : $MASK_DIR"
echo "  multilabel: $MULTI_DIR"
echo "  vesicles  : $([ "$GEN_VESICLE" = 1 ] && echo yes || echo skipped)"
echo "============================================================"

mask_for() {  # mask path for a given tomogram basename
    echo "$MASK_DIR/${1}_etsam_predicted_post_processed_mask.mrc"
}

# ---- 1. ETSAM binary masks (one GPU job at a time) ----
echo ""; echo "[1/3] ETSAM masks"
for tomo_mrc in "$TOMO_DIR"/*.mrc; do
    [ -e "$tomo_mrc" ] || continue
    tomo="$(basename "$tomo_mrc")"; tomo="${tomo%%.mrc*}"     # MIM..._ts_002
    mask="$(mask_for "$tomo")"
    if [ -s "$mask" ]; then echo "  SKIP (mask exists): $tomo"; continue; fi
    echo "  ETSAM: $tomo"
    ( cd "$ETSAM_REPO" && "$ETSAM_PY" etsam.py "$tomo_mrc" \
        --output-dir "$MASK_DIR" --post-process --post-process-min-slices 10 ) \
        2>&1 | tee "$MASK_DIR/${tomo}_etsam.log"
done
echo "      masks present: $(ls "$MASK_DIR"/*_post_processed_mask.mrc 2>/dev/null | wc -l)"

# ---- 2. Mito multi-label (OMM/IMM/ER) — one per *_Mito*_ER oracle ----
# Discover {tomo, tag} from each mito segmentation filename:
#   MIM019_2_lam4_ts_002_Mito1_ER.labels.mrc -> tomo=MIM019_2_lam4_ts_002  tag=Mito1_ER
echo ""; echo "[2/3] Mito multi-label (OMM/IMM/ER)"
for seg in "$MITO_DIR"/*.labels.mrc; do
    [ -e "$seg" ] || continue
    stem="$(basename "$seg" .labels.mrc)"            # MIM..._Mito1_ER
    tag="$(grep -oE 'Mito[0-9]*_ER$' <<<"$stem")"    # Mito_ER / Mito1_ER / Mito2_ER
    [ -n "$tag" ] || { echo "  SKIP (no Mito tag): $stem"; continue; }
    tomo="${stem%_$tag}"                             # MIM019_2_lam4_ts_002
    mask="$(mask_for "$tomo")"
    [ -s "$mask" ] || { echo "  SKIP (no mask yet): $tomo"; continue; }
    echo "  classify mito: $tomo  tag=$tag"
    "$MORPHO_PY" "$SM_DIR/etsam_to_multilabel.py" "$mask" "$seg" "$MULTI_DIR" --mito-tag "$tag"
done

# ---- 3. Vesicle label (optional; run_thickness_newpipe skips *Vesicle* anyway) ----
if [ "$GEN_VESICLE" = 1 ]; then
    echo ""; echo "[3/3] Vesicle label (otherOrg oracle, with mito competition)"
    for other in "$OTHER_DIR"/*_otherOrg.labels.mrc; do
        [ -e "$other" ] || continue
        tomo="$(basename "$other" _otherOrg.labels.mrc)"
        mask="$(mask_for "$tomo")"
        [ -s "$mask" ] || { echo "  SKIP (no mask yet): $tomo"; continue; }
        # Mito segs for this tomogram, so vesicle voxels compete against mito membranes.
        mito_segs=( "$MITO_DIR/${tomo}_"*.labels.mrc )
        [ -e "${mito_segs[0]}" ] || mito_segs=()
        echo "  classify vesicle: $tomo  (mito competition: ${#mito_segs[@]})"
        "$MORPHO_PY" "$SM_DIR/etsam_to_vesicle.py" "$mask" "$other" "$MULTI_DIR" \
            --mito-segs "${mito_segs[@]}"
    done
else
    echo ""; echo "[3/3] Vesicle label SKIPPED (GEN_VESICLE=0)"
fi

echo ""
echo "      multilabel MRCs: $(ls "$MULTI_DIR"/*.labels.mrc 2>/dev/null | wc -l)"
echo "=== DONE. Ready for:  run_thickness_newpipe.sh config_etsam_newpipe.yml"
