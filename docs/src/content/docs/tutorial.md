---
title: Tutorial
description: Step-by-step guide to segmenting cell membranes in cryo-ET tomograms with ETSAM.
prev:
  label: Installation
  link: /ETSAM/installation/
next:
  label: Advanced Usage
  link: /ETSAM/advanced/
---

This guide walks you through a complete run, from organizing your data to inspecting the predicted masks and handing them off to downstream analysis tools.

ETSAM is a two-stage, SAM2-based model that produces fully automatic membrane segmentations. It accepts a standard `.mrc` tomogram and outputs a `.mrc` binary membrane mask: every voxel is labeled as either membrane (=1) or background (=0). No manual prompting or per-tomogram training is required.

## Workflow

A typical ETSAM run follows three steps:

1. **Organize your data** — place your `.mrc` tomogram in a folder ETSAM can read. Ideally under `data/` folder.
2. **Run segmentation** — invoke `etsam.py`, which runs Stage 1 and Stage 2 inference and saves the final mask.
3. **(Optional) Post-process and analyze** — clean up artifacts and pass the mask to a downstream tool such as Surface Morphometrics or MemBrain-pick.

For per-tomogram tuning of thresholds and post-processing, see [Advanced Usage](/ETSAM/advanced/).

## Preparations

ETSAM needs a single input: a tomogram in `.mrc` format.

We recommend keeping inputs and outputs in dedicated folders so that runs stay reproducible. A common layout is:

```text
ETSAM/
|-- data/        # input tomograms (.mrc)
|-- results/     # predicted masks, one subfolder per run
```

Place your tomogram in `data/` (or any path you prefer), and ETSAM will write masks to the directory you pass via `--output-dir`.

:::tip
Denoised tomograms are not required, but may yield better masks.
:::

## Basic Command

```bash
python etsam.py <input_tomogram> --output-dir <output_directory>
```

| Argument | Required | Description |
| --- | --- | --- |
| `input_tomogram` | Yes | Path to the input tomogram file in `.mrc` format. |
| `--output-dir` | No | Directory where predicted membrane masks are saved. Default: current directory. |
| `--post-process` | No | Run post-processing to clean up artifacts. |
| `--logit-threshold` | No | Logit threshold for binarizing the mask. Default: `-0.25`. |
| `--store-logits` | No | Save raw logit scores as an additional `.mrc` file. |
| `--split-processing` | No | Split the tomogram into quadrants and segment each independently; useful for complex tomograms with closely apposed membranes. |

The default run produces a single binary mask. These flags — along with prompt selection, normalization, and fine-grained post-processing via `postprocess.py` — are covered in detail in [Advanced Usage](/ETSAM/advanced/).

## Example 1: Prokaryotic Cell

This example downloads an experimental tomogram of *Hylemonella gracilis* from the CryoET Data Portal (Dataset 10160, Run 8354) and runs ETSAM without post-processing.

### Download the Tomogram

```bash
wget -P data/ https://files.cryoetdataportal.cziscience.com/10160/ycw2013-01-03-15/Reconstructions/VoxelSpacing16.145/Tomograms/100/ycw2013-01-03-15.mrc
```

### Run ETSAM

```bash
python etsam.py data/ycw2013-01-03-15.mrc --output-dir results/cdp_run_8354/
```

### Expected Output

As ETSAM runs, it reports each stage of the pipeline to the terminal — reading the tomogram, running Stage 1 and Stage 2 inference, and saving the final mask:

```text
==> Reading input tomogram: data/ycw2013-01-03-15.mrc
Input tomogram shape: (500, 924, 956), voxel size: (16.145, 16.145, 16.145)
==> Running Stage 1 Prediction with Prompt Method: grid_zero
==> Loading ETSAM with checkpoint: checkpoints/etsam_stage1_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using grid of points at first slice as mask prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|████████| 500/500 [00:25<00:00, 19.83it/s]
==> Running Stage 2 Prediction with Prompt Method: zero
==> Loading ETSAM with checkpoint: checkpoints/etsam_stage2_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using zero initialized empty mask as prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|████████| 500/500 [00:31<00:00, 15.66it/s]
==> Saving predicted mask
-------------------------------------------------------------------------------------------------
```

:::note
Inference takes approximately 90 seconds on an NVIDIA RTX 4070. Actual time varies by GPU, tomogram size, and available memory.
:::

### Output Files

The predicted binary membrane mask is saved in the output directory:

```text
results/cdp_run_8354/
|-- ycw2013-01-03-15_etsam_predicted_mask.mrc
```

The output mask has the same shape and voxel spacing as the input tomogram, so it overlays directly on the original volume in any tomogram viewer.

### Visualization

<figure class="docs-figure">
  <img src="https://raw.githubusercontent.com/jianlin-cheng/ETSAM/main/assets/github-example-1.png" alt="ETSAM segmentation result for Hylemonella gracilis" loading="lazy" />
  <figcaption>Visualization of the Hylemonella gracilis tomogram slice and the ETSAM predicted membrane mask.</figcaption>
</figure>

## Example 2: Eukaryotic Cell with Post-processing

This example demonstrates the `--post-process` flag using an experimental tomogram of *Schizosaccharomyces pombe 972h-* from the CryoET Data Portal (Dataset 10000, Run 247).

### Download the Tomogram

```bash
wget -P data/ https://files.cryoetdataportal.cziscience.com/10000/TS_037/Reconstructions/VoxelSpacing13.480/Tomograms/100/TS_037.mrc
```

### Run ETSAM with Post-processing

```bash
python etsam.py data/TS_037.mrc \
    --post-process \
    --output-dir results/cdp_run_247/
```

### Expected Output

```text
==> Reading input tomogram: data/TS_037.mrc
Input tomogram shape: (500, 928, 960), voxel size: (13.480796, 13.480796, 13.480796)
==> Running Stage 1 Prediction with Prompt Method: grid_zero
==> Loading ETSAM with checkpoint: checkpoints/etsam_stage1_v1.pt
...
propagate in video: 100%|████████| 500/500 [00:24<00:00, 20.19it/s]
==> Running Stage 2 Prediction with Prompt Method: zero
==> Loading ETSAM with checkpoint: checkpoints/etsam_stage2_v1.pt
...
propagate in video: 100%|████████| 500/500 [00:31<00:00, 15.70it/s]
==> Saving predicted mask
==> Post-processing the predicted mask
==> Saving post-processed mask
-------------------------------------------------------------------------------------------------
```

### Output Files

When using `--post-process`, ETSAM saves both the raw and post-processed masks:

```text
results/cdp_run_247/
|-- TS_037_etsam_predicted_mask.mrc
|-- TS_037_etsam_predicted_post_processed_mask.mrc
```

### Visualization

<figure class="docs-figure">
  <img src="https://raw.githubusercontent.com/jianlin-cheng/ETSAM/main/assets/github-example-2.png" alt="ETSAM segmentation result for S. pombe" loading="lazy" />
  <figcaption>Comparison of the S. pombe tomogram, raw ETSAM prediction, and post-processed mask.</figcaption>
</figure>

:::caution
The blob-removal step may eliminate thin or small biologically relevant membrane regions. Always compare the raw and post-processed results visually before relying on the post-processed output.
:::

## Next Steps: Downstream Analysis

The binary mask ETSAM produces is a standard `.mrc` volume aligned to your tomogram, which makes it a convenient starting point for membrane-focused downstream analysis. Common next steps include:

- **[Surface Morphometrics](https://github.com/GrotjahnLab/surface_morphometrics)** — convert the segmentation into a triangle-mesh surface to quantify membrane curvature, thickness, and inter-membrane distances.
- **[MemBrain-pick](https://github.com/teamtomo/membrain-pick)** — detect and pick membrane-associated particles (e.g. membrane proteins) directly on the segmented surface.
- **Manual refinement** — load the mask alongside the tomogram in [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/) or [IMOD](https://bio3d.colorado.edu/imod/) to inspect, correct, or extract specific membrane regions.

To fine-tune the mask before passing it downstream — for example, by adjusting the binarization threshold or saving raw confidence scores — continue to [Advanced Usage](/ETSAM/advanced/).
