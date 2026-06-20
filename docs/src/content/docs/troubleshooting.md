---
title: Troubleshooting
description: Fix common ETSAM installation and runtime issues, including CUDA detection and version overrides.
prev:
  label: Citation
  link: /ETSAM/citation/
next: false
---

Common issues encountered when installing or running ETSAM, and how to resolve them.

## Empty or Incomplete Membrane Predictions

If ETSAM produces an empty mask or misses parts of the membranes, several options are worth trying before giving up on a tomogram:

- **Use a grid prompt for Stage 1.** Seeding Stage 1 with a grid of points every 50th slice can help in better detection of membranes: `python etsam.py input.mrc --stage1-prompt grid --output-dir results/`. This is not recommended unless you have issues with default prompt method, as it can introduce more noise into prediction.
- **Try `softplus_minmax` normalization.** Switching from the default `min_max_positive_values` normalization method to `--normalize-method softplus_minmax` can improve membrane contrast on some tomograms.
- **Use `--split-processing`.** Splitting the tomogram into quadrants and segmenting each with a denser prompt grid improves detection in complex tomograms, especially where membranes are closely apposed.
- **Denoise the tomogram.** can help in better detection of membranes.
- **Lower the threshold by inspecting the logits.** Run etsam `--store-logits`, open the resulting logit volume in UCSF ChimeraX or other visualization software, and check whether a lower threshold would capture the missing membrane signal.

See [Advanced Usage](/ETSAM/advanced/) for details on each of these flags.

## CUDA not Detected

Ensure that your NVIDIA drivers and CUDA toolkit are installed before creating the conda environment:

```bash
nvidia-smi
nvcc --version
```

When using a [Dev Container](/ETSAM/installation/#method-2-installation-using-dev-containers), run `nvidia-smi` inside the container's terminal to confirm the GPU is accessible from within the container.

## Overriding the CUDA Version

PyTorch and the CUDA runtime are installed from `conda-forge`. By default the CUDA version is **decided at install time** — conda detects your installed NVIDIA driver and selects the newest compatible CUDA build automatically, so no manual configuration is needed in most cases.

If you need to force a specific CUDA version (for example, to match an older driver), pin it in one of the following ways.

**Override when creating the environment:**

```bash
CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yml
```

**Or install into an existing `etsam` environment:**

```bash
conda install -n etsam -c conda-forge "cuda-version=12.4"
```

The available `cuda-version` values are those built in [conda-forge](https://anaconda.org/channels/conda-forge/packages/cuda-version/overview).

## Conda Environment Creation Fails

Try updating conda first, then recreate the environment:

```bash
conda update -n base -c conda-forge conda
conda env create -f environment.yml
```

## Out-of-memory Errors

:::caution
ETSAM requires at least 2 GB of GPU VRAM. If you encounter out-of-memory errors, close other GPU-intensive applications and try again. 4GB of GPU VRAM is recommended.
:::
