---
title: Advanced Usage
description: Tune logit thresholds, normalization, prompts, split-processing, and post-processing for per-tomogram optimization of ETSAM.
prev:
  label: Tutorial
  link: /ETSAM/tutorial/
next:
  label: Training
  link: /ETSAM/training/
---

ETSAM exposes several inference flags for per-tomogram optimization. These options are useful when you want to trade sensitivity against precision, inspect confidence values, handle complex tomograms with closely apposed membranes, or tune artifact removal.

## Custom Logit Threshold

For each pixel in a tomogram slice, ETSAM's mask decoder predicts a logit value: a continuous score indicating the likelihood that a cell membrane is present. Larger values indicate higher confidence. ETSAM converts logits into a binary mask by classifying pixels with a score greater than the threshold as membrane.

:::note
The default logit threshold is `-0.25`, selected from ablation studies for the best overall precision-recall trade-off. Individual tomograms may benefit from different values.
:::

Use `--logit-threshold` to override the default:

```bash
python etsam.py input.mrc \
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
python etsam.py input.mrc \
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
python etsam.py input.mrc \
    --normalize-method min_max_positive_values \
    --output-dir results/
```

| Method | Description |
| --- | --- |
| `softplus_minmax` (default) | Applies a softplus transform followed by min-max scaling. Robust general-purpose default. |
| `min_max_positive_values` | Clips negative values to zero, then scales by the maximum. May improve membrane detection on certain tomograms. |

If membranes are missed with the default softplus normalization, switching to `min_max_positive_values` is worth trying.

## Prompt Methods

ETSAM is a two-stage model, and each stage uses an automatic prompting strategy that seeds the membrane prediction. The available prompt methods are:

- **`grid_zero`** — seeds only the first slice with a grid of points. Sufficient in most cases.
- **`grid`** — seeds the first and every 50th slice with a grid of points. Can recover membranes that `zero` or `grid_zero` miss.
- **`zero`** — uses an empty prompt, letting the model automatically detect membranes without any input.
- **`etsam_stage1_partial`** — reuses the Stage 1 mask to seed the model at the first and every 50th slice. Only available for Stage 2.

They can be overridden and experimented with in the following way:

```bash
python etsam.py input.mrc \
    --stage1-prompt grid \
    --stage2-prompt etsam_stage1_partial \
    --output-dir results/
```

| Flag | Default | Choices |
| --- | --- | --- |
| `--stage1-prompt` | `grid_zero` | `zero`, `grid`, `grid_zero` |
| `--stage2-prompt` | `etsam_stage1_partial` | `zero`, `grid`, `grid_zero`, `etsam_stage1_partial` |

## Split Processing

For complex tomograms — for example, those with closely apposed cell membranes that tend to merge into a single blob — `--split-processing` divides the tomogram into four quadrants (halving the Y and X axes), runs the full two-stage pipeline on each independently, and merges the results back into a full-size volume. The smaller regions are seeded with a denser prompt grid (grid of points every 10 voxel instead of default 50 voxel space) to provide better delineation of individual membranes.

```bash
python etsam.py input.mrc \
    --split-processing \
    --output-dir results/
```

Outputs from a split-processing run are tagged accordingly:

```text
results/
|-- input_etsam_predicted_split_processing_mask.mrc
```

:::note
Split-processing runs inference four times, so it takes roughly four times longer than a standard run.
:::

## Post-processing with `postprocess.py`

Post-processing operates on an existing binary mask, so you can run it as a standalone step on a previously generated mask without re-running inference. `postprocess.py` takes an input mask and an output path:

```bash
python postprocess.py results/input_etsam_predicted_mask.mrc \
    results/input_post_processed_mask.mrc
```

With no method flags, **all** post-processing steps run.

:::caution
Always visually compare the orignal predicted mask and post-processed mask to verify that no biologically relevant membranes are removed during post-processing.
:::

The following two post-processing steps are available, and you can select them individually as specified below.

### Remove Small Noise

This step identifies 3D connected components in the mask and removes blobs that do not span a minimum number of consecutive Z-slices, eliminating small, isolated noise while preserving larger continuous membranes.

```bash
python postprocess.py mask_in.mrc mask_out.mrc \
    --postprocess-remove-small-noise \
    --post-process-min-slices 5
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
python postprocess.py mask_in.mrc mask_out.mrc \
    --postprocess-remove-parallel-membrane-misconnections \
    --post-process-max-bridge-len 8 \
    --post-process-max-spur-len 5
```

| Flag | Default | Description |
| --- | --- | --- |
| `--post-process-max-bridge-len` | `5` | Maximum skeleton length (px) of a rung between two junctions to remove. |
| `--post-process-max-spur-len` | `0` | If `> 0`, also prune dead-end spurs up to this length (px). |

### Run post-processing integrated with etsam

Post-processing can also be applied in an integrated manner directly from `etsam.py` using the same flags from postprocess.py as discussed above. For example: `python etsam.py input.mrc --post-process --output-dir results/` will run all post-processing steps. or `python etsam.py input.mrc --postprocess-remove-small-noise --output-dir results/` to only remove small and thin noise blobs.
