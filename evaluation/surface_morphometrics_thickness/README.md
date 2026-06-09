# Surface Morphometrics — membrane-thickness evaluation of ETSAM segmentations

This folder reproduces the organellar membrane-thickness measurements of
**Medina et al., JCB 2025** — *"Surface Morphometrics reveals local membrane thickness
variation in organellar subcompartments"* ([doi:10.1083/jcb.202505059](https://doi.org/10.1083/jcb.202505059))
— and then **swaps the authors' hand-curated ground-truth segmentations for
ETSAM-predicted segmentations** to test whether the OMM/IMM/ER thickness measurements
still hold.

Per surface the metric is the paper's: the **area-weighted median** of the per-triangle
dual-Gaussian thickness; per organelle we report the mean across surfaces ± a t-based
95% CI. Result (default `config.yml` mesh params — isotropic remesh, octree 9, rh9):

| Organelle | Paper segmentation | ETSAM segmentation | Paper reported |
|-----------|--------------------|--------------------|----------------|
| OMM | 3.174 ± 0.107 (n=15) | 3.157 ± 0.111 (n=15) | 3.20 ± 0.10 |
| IMM | 3.670 ± 0.070 (n=15) | 3.632 ± 0.079 (n=15) | 3.60 ± 0.07 |
| ER  | 3.719 ± 0.065 (n=12) | 3.697 ± 0.060 (n=12) | 3.70 ± 0.05 |

ETSAM-predicted segmentations track the ground-truth ones to within ~0.04 nm, both
within the paper's 95% CI.

---

## What's in this folder

| File | Role |
|------|------|
| `download_empiar_13056.sh`      | Download the EMPIAR-13056 MEF tomograms + voxel segmentations (parallel multipart curl). |
| `config_mef_newpipe.yml`        | Pipeline config for the **paper (ground-truth)** segmentations. |
| `config_etsam_newpipe.yml`      | Pipeline config for the **ETSAM** segmentations (same params, different `seg_dir`). |
| `prepare_etsam_segmentations.sh`| Run ETSAM on the tomograms → OMM/IMM/ER multi-label MRCs. |
| `etsam_to_multilabel.py`        | Helper: assign OMM/IMM/ER to ETSAM membrane voxels (paper labels as oracle). |
| `etsam_to_vesicle.py`           | Helper: assign Vesicle (optional; thickness run ignores vesicles). |
| `run_thickness_newpipe.sh`      | The thickness pipeline: meshes → pycurv → `sample_density` → `measure_thickness`. |
| `compute_thickness_medians.py`  | Reduce each refined surface to its area-weighted median → final CSV. |

These are wrappers around the **stock** surface_morphometrics scripts
(`segmentation_to_meshes.py`, `run_pycurv.py`, `sample_density.py`,
`measure_thickness.py`); they are meant to be **copied into the surface_morphometrics
repo root** and run from there.

---

## Step 0 — Get surface_morphometrics at the exact commit this was built on

```bash
git clone https://github.com/grotjahnlab/surface_morphometrics.git
cd surface_morphometrics
git checkout 0b4f6af03b9513e69b109b7c54899e37fa464e2f   # "Mesh refinement and Thickness Measurement (#56)"

# create + activate the pipeline environment (graph-tool, pycurv, pymeshlab, ...)
conda env create -f environment.yml      # or environment-ubuntu.yml on older Ubuntu
conda activate morphometrics
```

## Step 1 — Copy these evaluation scripts into the repo root

```bash
cp /path/to/ETSAM/evaluation/surface_morphometrics_thickness/*.sh  .
cp /path/to/ETSAM/evaluation/surface_morphometrics_thickness/*.py  .
cp /path/to/ETSAM/evaluation/surface_morphometrics_thickness/*.yml .
```

## Step 2 — Configure paths for your machine

The scripts and configs ship with absolute paths from the original machine. Update the
lines below in each file. There are two prefixes to replace throughout:

- `<SM>`     = your surface_morphometrics repo root (e.g. the `pwd` after Step 0).
- `<MORPHO_PY>` = your `morphometrics` conda env python; `<ETSAM_PY>` = your `etsam` env python.

**`config_mef_newpipe.yml`** (paper segmentations)
- `seg_dir`  → `<SM>/data/empiar_13056/mef/segmentations_mito_er/`
- `tomo_dir` → `<SM>/data/empiar_13056/mef/tomograms/`
- `work_dir` → `<SM>/results/mef_newpipe_default/`

**`config_etsam_newpipe.yml`** (ETSAM segmentations)
- `seg_dir`  → `<SM>/data/empiar_13056/mef/etsam_phase2/etsam_multilabel_full/`
- `tomo_dir` → `<SM>/data/empiar_13056/mef/tomograms/`
- `work_dir` → `<SM>/results/mef_etsam_newpipe_default/`

**`compute_thickness_medians.py`**
- `REPO = Path("…")` → `<SM>`

**`run_thickness_newpipe.sh`**
- `PY=…` → `<MORPHO_PY>`

**`download_empiar_13056.sh`**
- `DATA_ROOT="${1:-…}"` → `<SM>/data/empiar_13056/mef`  *(or just pass it as the first argument: `bash download_empiar_13056.sh <SM>/data/empiar_13056/mef`)*

**`prepare_etsam_segmentations.sh`**
- `DATA_ROOT="${1:-…}"` → `<SM>/data/empiar_13056/mef`
- `SM_DIR=…`     → `<SM>`  (where the `etsam_to_*.py` helpers were copied)
- `ETSAM_REPO=…` → your ETSAM clone (the one holding `etsam.py`)
- `ETSAM_PY=…`   → `<ETSAM_PY>`
- `MORPHO_PY=…`  → `<MORPHO_PY>`

**`etsam_to_multilabel.py`, `etsam_to_vesicle.py`** — no edits needed (all paths are passed in as arguments by `prepare_etsam_segmentations.sh`).

> The configs mirror the repository **default** `config.yml` (isotropic remesh, octree 9,
> `radius_hit = 9`); only the paths, `cores`, and `seg_dir` differ.

## Step 3 — Run the end-to-end flow

Run from the surface_morphometrics repo root with the `morphometrics` env active. Every
step is idempotent (skips finished work) and never deletes (suspect outputs are renamed
to `*.corrupted_<UTC>`).

```bash
# 1. Download EMPIAR-13056 MEF data -> data/empiar_13056/mef/
#    (tomograms/, segmentations/, segmentations_mito_er/, segmentations_other/)
bash download_empiar_13056.sh

# 2. Ground-truth (paper) segmentations -> per-surface OMM/IMM/ER thickness
bash run_thickness_newpipe.sh config_mef_newpipe.yml

# 3. ETSAM segmentations: run ETSAM + build OMM/IMM/ER multi-label MRCs (needs a GPU),
#    then the SAME thickness pipeline on them
bash prepare_etsam_segmentations.sh
bash run_thickness_newpipe.sh config_etsam_newpipe.yml

# 4. Reduce every refined surface to its area-weighted median thickness -> final CSV
python compute_thickness_medians.py
```

## Output

- `results/thickness_default_persurface.csv` — **the desired output**: one row per surface
  (`group, organelle, surface_key, surface_median_nm, n_triangles`) for both the paper and
  ETSAM segmentations. Aggregate per organelle = mean ± t-based 95% CI of these rows.
- `results/mef_newpipe_default/` and `results/mef_etsam_newpipe_default/` — per-surface
  `*_refined.gt` graphs (carry the per-triangle `thickness`) and `component_list.csv`.

> The `.docx`/figure report generator (`generate_default_report.py`) is intentionally
> **not** included here — this evaluation stops at the per-surface thickness CSV.
