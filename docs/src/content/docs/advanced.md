---
title: Advanced Usage
description: Tune logit thresholds, normalization, prompts, split-processing, and post-processing for per-tomogram optimization of ETSAM.
prev:
  label: napari Plugin
  link: /ETSAM/napari/
next:
  label: Training
  link: /ETSAM/training/
---

ETSAM exposes several inference flags for per-tomogram optimization. These options are useful when you want to trade sensitivity against precision, inspect confidence values, handle complex tomograms with closely apposed membranes, or tune artifact removal.

## All Options

`etsam [options] TOMOGRAM` segments one tomogram (`.mrc`) per run. The sections below explain each option in detail.

| Option | Default | Description |
| --- | --- | --- |
| `TOMOGRAM` | (required) | Input tomogram in `.mrc` format. |
| `-V`, `--version` | | Print the installed ETSAM version and exit. |
| `--output-dir DIR` | current folder | Folder where the predicted mask (and the other outputs) are saved. |
| `--logit-threshold T` | `-0.25` | Threshold for binarizing the Stage 2 logits into the final mask. See [Custom Logit Threshold](#custom-logit-threshold). |
| `--store-logits` | off | Also save the raw Stage 2 logits (membrane scores) as an `.mrc` file. See [Store Predicted Logit Scores](#store-predicted-logit-scores). |
| `--split-processing` | off | Segment the four Y/X quadrants of the tomogram separately; useful for complex tomograms with closely apposed membranes. See [Split Processing](#split-processing). |
| `--normalize-method M` | `min_max_positive_values` | Input normalization: `min_max_positive_values` or `softplus_minmax`. See [Tomogram Normalization](#tomogram-normalization). |
| `--stage1-prompt P` | `grid_zero` | Stage 1 prompt method: `zero`, `grid` or `grid_zero`. See [Prompt Methods](#prompt-methods). |
| `--stage2-prompt P` | `zero` | Stage 2 prompt method: `zero`, `grid`, `grid_zero` or `etsam_stage1_partial`. See [Prompt Methods](#prompt-methods). |
| `--post-process` | off | Run all post-processing steps after inference. See [Run post-processing integrated with etsam](#run-post-processing-integrated-with-etsam). |
| `--postprocess-remove-thin-noise` | off | Run only the thin-noise removal step. See [Remove Small Noise](#remove-small-noise). |
| `--postprocess-remove-parallel-membrane-misconnections` | off | Run only the step that cuts short bridges between parallel membranes. See [Remove Parallel Membrane Misconnections](#remove-parallel-membrane-misconnections). |
| `--post-process-min-slices N` | `10` | Minimum number of Z slices a connected blob must span to be kept by thin-noise removal. |
| `--post-process-max-bridge-len N` | `5` | Longest skeleton bridge (px) between two junctions that misconnection removal cuts. |
| `--post-process-max-spur-len N` | `0` (off) | If greater than 0, also prune dead-end spurs up to this length (px). |

The post-processing options are also available in the separate `etsam-postprocess` command, which works on an existing mask; see [Post-processing with `etsam-postprocess`](#post-processing-with-etsam-postprocess).

## Custom Logit Threshold

For each pixel in a tomogram slice, ETSAM's mask decoder predicts a logit value: a continuous score indicating the likelihood that a cell membrane is present. Larger values indicate higher confidence. ETSAM converts logits into a binary mask by classifying pixels with a score greater than the threshold as membrane.

:::note
The default logit threshold is `-0.25`, selected from ablation studies for the best overall precision-recall trade-off. Individual tomograms may benefit from different values.
:::

Use `--logit-threshold` to override the default:

```bash
etsam input.mrc \
    --logit-threshold -0.5 \
    --output-dir results/
```

| Threshold | Effect | Best Used When |
| --- | --- | --- |
| `-0.25` | Default - Overall balanced precision and recall | General-purpose starting point based on our evaluations, might not be ideal for every tomogram |
| `-0.5` to `-1.0` | Higher recall, lower precision | Detecting faint or thin membranes in noisy tomograms, but can introduce more false positives |
| `0.0` to `0.5` | Higher precision, lower recall | Reducing false positive noises in predicted mask, but can miss weaker membrane signal. |

## Store Predicted Logit Scores

ETSAM can save raw logit scores as an `.mrc` file in addition to the binary mask. This lets you visualize model confidence, explore thresholds interactively in UCSF ChimeraX or IMOD, and build custom workflows over continuous scores.

```bash
etsam input.mrc \
    --store-logits \
    --output-dir results/
```

This saves an additional file alongside the binary mask:

```text
results/
|-- input_etsam_predicted_mask.mrc
|-- input_etsam_predicted_logits.mrc
```

:::tip
A useful workflow is to first run with `--store-logits`, inspect the logit volume in a viewer, choose an appropriate threshold, and then re-run with ideal `--logit-threshold`.
:::

## Tomogram Normalization

Before inference, ETSAM normalizes the input tomogram. The `--normalize-method` flag selects the strategy:

```bash
etsam input.mrc \
    --normalize-method softplus_minmax \
    --output-dir results/
```

| Method | Description |
| --- | --- |
| `min_max_positive_values` (default) | Clips negative values to zero, then scales by the maximum. General-purpose starting point for ETSAM inference. |
| `softplus_minmax` | Applies a softplus transform followed by min-max scaling. May improve membrane detection on certain tomograms. |

If membranes are missed/improperly detected with the default `min_max_positive_values` normalization, switching to `softplus_minmax` is worth trying.

## Prompt Methods

ETSAM is a two-stage model, and each stage uses an automatic prompting strategy that seeds the membrane prediction. The available prompt methods are:

- **`grid_zero`** — seeds only the first slice with a grid of points. This is the default for Stage 1.
- **`grid`** — seeds the first slice and then regularly spaced slices with a grid of points. Can recover membranes that `zero` or `grid_zero` miss.
- **`zero`** — uses an empty prompt, letting the model automatically detect membranes without any input. This is the default for Stage 2.
- **`etsam_stage1_partial`** — reuses the Stage 1 mask to seed the model at the first slice and then regularly spaced slices. Only available for Stage 2.

They can be overridden and experimented with in the following way:

```bash
etsam input.mrc \
    --stage1-prompt grid \
    --stage2-prompt etsam_stage1_partial \
    --output-dir results/
```

| Flag | Default | Choices |
| --- | --- | --- |
| `--stage1-prompt` | `grid_zero` | `zero`, `grid`, `grid_zero` |
| `--stage2-prompt` | `zero` | `zero`, `grid`, `grid_zero`, `etsam_stage1_partial` |

## Split Processing

For complex tomograms — for example, those with closely apposed cell membranes that tend to merge into a single blob — `--split-processing` divides the tomogram into four quadrants (halving the Y and X axes), runs the full two-stage pipeline on each independently, and merges the results back into a full-size volume. The smaller regions are seeded with a denser prompt grid (grid of points every 10 voxel instead of default 50 voxel space) to provide better delineation of individual membranes.

```bash
etsam input.mrc \
    --split-processing \
    --output-dir results/
```

Outputs from a split-processing run are tagged accordingly:

```text
results/
|-- input_etsam_predicted_split_processing_mask.mrc
```

<details>
<summary>Example terminal output (TS_037 from the <a href="/ETSAM/tutorial/#example-2-eukaryotic-cell-with-post-processing">tutorial</a>)</summary>

```text
$ etsam data/TS_037.mrc --split-processing --output-dir results/
==> Reading input tomogram: /home/user/demo/data/TS_037.mrc
Input tomogram shape: (500, 928, 960), voxel size: (13.480796, 13.480796, 13.480796)
==> Normalizing tomogram data with min_max_positive_values method
==> Split tomogram into 4 sub-tomograms (split-processing mode, grid_interval=10)
    top-left: shape (500, 464, 480), Y[0:464] X[0:480]
    top-right: shape (500, 464, 480), Y[0:464] X[480:960]
    bottom-left: shape (500, 464, 480), Y[464:928] X[0:480]
    bottom-right: shape (500, 464, 480), Y[464:928] X[480:960]
==> Processing quadrant: top-left
==> Stage 1 prediction (prompt: grid_zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage1_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using grid of points at first slice as mask prompt (grid_interval=10)
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:28<00:00, 17.28it/s]
==> Stage 2 prediction (prompt: zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage2_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using zero initialized empty mask as prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:29<00:00, 17.12it/s]
==> Processing quadrant: top-right
==> Stage 1 prediction (prompt: grid_zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage1_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using grid of points at first slice as mask prompt (grid_interval=10)
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:29<00:00, 17.07it/s]
==> Stage 2 prediction (prompt: zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage2_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using zero initialized empty mask as prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:29<00:00, 17.05it/s]
==> Processing quadrant: bottom-left
==> Stage 1 prediction (prompt: grid_zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage1_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using grid of points at first slice as mask prompt (grid_interval=10)
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:29<00:00, 17.07it/s]
==> Stage 2 prediction (prompt: zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage2_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using zero initialized empty mask as prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:28<00:00, 17.25it/s]
==> Processing quadrant: bottom-right
==> Stage 1 prediction (prompt: grid_zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage1_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using grid of points at first slice as mask prompt (grid_interval=10)
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:29<00:00, 17.24it/s]
==> Stage 2 prediction (prompt: zero)
==> Loading ETSAM with checkpoint:  /home/user/etsam_env/checkpoints/etsam_stage2_v1.pt
Adjusting hydra overrides for SAM2 video predictor
==> Preprocessing the input map
Loading video frames from numpy array as Floating point array
using zero initialized empty mask as prompt
==> Running ETSAM inference on preprocessed map
propagate in video: 100%|██████████| 500/500 [00:29<00:00, 17.23it/s]
==> Combining quadrant logits into full volume
==> Saving predicted mask
-------------------------------------------------------------------------------------------------
```

</details>

:::note
Split-processing runs inference four times (on smaller regions), so it takes roughly three to four times longer than a standard run: 293 seconds instead of 93 for the tutorial's TS_037 tomogram on an NVIDIA RTX 4070.
:::

## Post-processing with `etsam-postprocess`

Post-processing operates on an existing binary mask, so you can run it as a standalone step on a previously generated mask without re-running inference. `etsam-postprocess` takes an input mask and an output path:

```bash
etsam-postprocess results/cdp_run_247/TS_037_etsam_predicted_mask.mrc \
    results/cdp_run_247/TS_037_post_processed_mask.mrc
```

Example output, using the mask from [Example 2 of the tutorial](/ETSAM/tutorial/#example-2-eukaryotic-cell-with-post-processing):

```text
==> Reading input mask: /home/user/demo/results/cdp_run_247/TS_037_etsam_predicted_mask.mrc
Input mask shape: (500, 928, 960), voxel size: (13.480796, 13.480796, 13.480796)
==> Post-processing: remove small noise (min_z_span=10)
==> Post-processing: remove parallel membrane misconnections (max_bridge_len=5, max_spur_len=0)
==> Saving post-processed mask: /home/user/demo/results/cdp_run_247/TS_037_post_processed_mask.mrc
Done.
```

With no method flags, **all** post-processing steps run.

:::caution
Always visually compare the orignal predicted mask and post-processed mask to verify that no biologically relevant membranes are removed during post-processing.
:::

The following two post-processing steps are available, and you can select them individually as specified below.

### Remove Small Noise

This step identifies 3D connected components in the mask and removes blobs that do not span a minimum number of consecutive Z-slices, eliminating small, isolated noise while preserving larger continuous membranes.

```bash
etsam-postprocess results/cdp_run_247/TS_037_etsam_predicted_mask.mrc \
    results/cdp_run_247/TS_037_min_slices_5_mask.mrc \
    --postprocess-remove-thin-noise \
    --post-process-min-slices 5
```

```text
==> Reading input mask: /home/user/demo/results/cdp_run_247/TS_037_etsam_predicted_mask.mrc
Input mask shape: (500, 928, 960), voxel size: (13.480796, 13.480796, 13.480796)
==> Post-processing: remove small noise (min_z_span=5)
==> Saving post-processed mask: /home/user/demo/results/cdp_run_247/TS_037_min_slices_5_mask.mrc
Done.
```

The minimum span is controlled by `--post-process-min-slices` (default `10`):

| Value | Effect | Recommendation |
| --- | --- | --- |
| `1-5` | Removes only very small blobs | Tomograms with many thin membranes |
| `10` | Removes small and medium thin noise blobs | General use; good starting point |
| `15-30` | Removes larger noise blobs | But may also remove small membrane structures |

### Remove Parallel Membrane Misconnections

This step skeletonizes each Z-slice and severs short "rung" interconnections that incorrectly bridge two parallel membranes, helping to delineate closely apposed membranes that the segmentation fused together.

```bash
etsam-postprocess results/cdp_run_247/TS_037_etsam_predicted_mask.mrc \
    results/cdp_run_247/TS_037_bridge_8_mask.mrc \
    --postprocess-remove-parallel-membrane-misconnections \
    --post-process-max-bridge-len 8 \
    --post-process-max-spur-len 5
```

```text
==> Reading input mask: /home/user/demo/results/cdp_run_247/TS_037_etsam_predicted_mask.mrc
Input mask shape: (500, 928, 960), voxel size: (13.480796, 13.480796, 13.480796)
==> Post-processing: remove parallel membrane misconnections (max_bridge_len=8, max_spur_len=5)
==> Saving post-processed mask: /home/user/demo/results/cdp_run_247/TS_037_bridge_8_mask.mrc
Done.
```

| Flag | Default | Description |
| --- | --- | --- |
| `--post-process-max-bridge-len` | `5` | Maximum skeleton length (px) of a rung between two junctions to remove. |
| `--post-process-max-spur-len` | `0` | If `> 0`, also prune dead-end spurs up to this length (px). |

### Run post-processing integrated with etsam

Post-processing can also be applied in an integrated manner directly from the `etsam` command using the same flags as `etsam-postprocess`. For example: `etsam input.mrc --post-process --output-dir results/` will run all post-processing steps. or `etsam input.mrc --postprocess-remove-thin-noise --output-dir results/` to only remove small and thin noise blobs.
