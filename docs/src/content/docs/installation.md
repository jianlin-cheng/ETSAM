---
title: Installation
description: Install ETSAM via conda or Dev Containers, download model weights, and verify the environment.
prev:
  label: Overview
  link: /ETSAM/
next:
  label: Tutorial
  link: /ETSAM/tutorial/
---

Set up ETSAM on your system. ETSAM can be installed in two ways: directly from the terminal using conda, or as a reproducible [Dev Container](https://containers.dev/). Installation typically takes 5-15 minutes depending on internet speed and package cache state.

## Requirements

### Hardware

- NVIDIA GPU with at least 2 GB VRAM; 4 GB VRAM is recommended.
- At least 24 GB CPU RAM; 32 GB is recommended.

:::note
AMD GPUs may work, but might require an [AMD-specific PyTorch](https://pytorch.org/get-started/locally/) installation inside the `etsam` conda environment. This is currently untested.
:::

### Operating System

- Linux (tested on Fedora 43 and Arch Linux).
- Windows may work, but is currently untested.

### Tested Setup

- Intel Core i7-14700K CPU
- NVIDIA GeForce RTX 4070 with 12 GB VRAM
- 32 GB DDR5 CPU RAM
- NVIDIA CUDA 13.0 and driver version 580.119.02
- conda 25.11.0

## Get the Code

Download the project as a [ZIP archive](https://github.com/jianlin-cheng/ETSAM/archive/refs/heads/main.zip) and extract **ETSAM-main.zip** into a preferred folder, or clone it from the terminal:

```bash
git clone https://github.com/jianlin-cheng/ETSAM
cd ETSAM
```

## Method 1: Installation Using the Terminal

This installs ETSAM into a dedicated conda environment on your host machine.

### Software Requirements

- `conda`, preferably installed through [Miniforge](https://conda-forge.org/download/). It installs Python and the required package dependencies. Tested with conda 25.11.0.
- A GPU compute stack such as NVIDIA CUDA or AMD ROCm, as needed to run PyTorch on the GPU. Tested with NVIDIA CUDA 13.0 and driver version 580.119.02.

:::tip
We recommend installing conda through [Miniforge](https://conda-forge.org/download/) for a fast, community-maintained conda distribution that defaults to conda-forge.
:::

### 1. Download Model Weights

Download the pre-trained ETSAM Stage 1 and Stage 2 checkpoints from Zenodo:

```bash
wget -P checkpoints/ https://zenodo.org/records/17571925/files/etsam_stage1_v1.pt
wget -P checkpoints/ https://zenodo.org/records/17571925/files/etsam_stage2_v1.pt
```

The weights are saved in `checkpoints/`. ETSAM uses these paths by default.

### 2. Create the Conda Environment

Use the provided `environment.yml` to create a dedicated environment with the Python dependencies:

```bash
conda env create -f environment.yml
```

:::note
Environment creation usually takes 5-10 minutes. This installs PyTorch, SAM2-related dependencies, and the required cryo-ET processing packages. Installation time may vary based on your internet speed.
:::

### 3. Activate the Environment

```bash
conda activate etsam
```

You should see `(etsam)` at the beginning of your terminal prompt when the environment is active.

### 4. Verify the Installation

Run the help command to confirm ETSAM can be imported and executed:

```bash
python etsam.py --help
```

## Method 2: Installation Using Dev Containers

A [Dev Container](https://containers.dev/) is a reproducible way to run ETSAM. It builds the conda environment into a Docker image and downloads the model weights for you, so no manual environment setup or weight download is needed. This keeps the environment consistent across different systems.

### Software Requirements

- [Docker](https://docs.docker.com/get-docker/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for NVIDIA GPU access.
- Either VS Code with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), or the [`devcontainer` CLI](https://github.com/devcontainers/cli).

:::tip[Data Location]
When using Dev Containers, place your input tomogram data into the **data/** directory and store output segmentations in **results/**, as these directories are shared between the host and container.
:::

### Option A: VS Code

1. Open the ETSAM folder and choose **Reopen in Container** (command palette → *Dev Containers: Reopen in Container*).
   - The first run takes time to download the container image, install the conda environment, and download the checkpoints. Subsequent opens are instant.
2. The VS Code editor and its integrated terminal then run **inside** the container, giving you access to everything needed to run ETSAM.
3. Run `nvidia-smi` in the VS Code terminal to confirm the NVIDIA GPU is accessible inside the container.

Continue to the [Tutorial](/ETSAM/tutorial/) section to learn how to run ETSAM.

### Option B: Dev Container CLI

Start (and, on first run, build) the container:

```bash
devcontainer up
```

`devcontainer up` only starts the container — it does not give you a shell. Open one inside it with:

```bash
devcontainer exec bash
```

You are now inside the container, and everything including the `data/` folder is available. Continue to the [Tutorial](/ETSAM/tutorial/) section to learn how to run ETSAM.

## Repository Structure

After installation, the repository should look like this:

```text
ETSAM/
|-- etsam.py                  # Main inference script
|-- environment.yml           # Conda environment spec
|-- checkpoints/
|   |-- etsam_stage1_v1.pt    # Stage 1 model weights
|   |-- etsam_stage2_v1.pt    # Stage 2 model weights
|-- sam2/                     # Modified SAM2 source code
|-- training/                 # Training scripts
|-- evaluation/               # Evaluation scripts
|-- scripts/                  # Data collection utilities
|-- data/                     # Data directory created by the user
```

:::tip
Running into installation problems? See the [Troubleshooting](/ETSAM/troubleshooting/) page for common issues and fixes.
:::
