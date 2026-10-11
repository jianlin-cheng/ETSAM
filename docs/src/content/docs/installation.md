---
title: Installation
description: Install ETSAM with the installer, a conda environment or a Dev Container, get the model weights, and verify the setup.
prev:
  label: Overview
  link: /ETSAM/
next:
  label: Tutorial
  link: /ETSAM/tutorial/
---

ETSAM provides the `etsam` command. There are three ways to install it:

- **[ETSAM installer](#method-1-etsam-installer-recommended)** (recommended): a single download that sets up ETSAM, its dependencies and checkpoints in their own conda environment.
- **[Conda environment from source](#method-2-conda-environment-from-source)**: Clone git repo and install conda environment manually. Use this if you also want to modify, train or evaluate ETSAM.
- **[Dev Container](#method-3-dev-container)**: a reproducible Docker environment with everything preinstalled.

Installation typically takes 5–15 minutes, mostly spent downloading conda environment dependencies and model checkpoints.

## Requirements

### Hardware

- NVIDIA GPU with at least 2 GB VRAM; 4 GB VRAM is recommended.
- At least 16 GB CPU RAM; 32 GB is recommended. Peak memory grows with tomogram size: a 360 × 928 × 960 tomogram needs about 9 GB.
- About 6 GB of disk space for PyTorch and the 1.9 GB of model weights.

### Supported Platforms

ETSAM supports **Linux and Windows with an NVIDIA GPU** (tested on Fedora 43, Arch Linux and Windows 11).

:::note
Other configurations (macOS, AMD or Intel GPUs) may work, but require testing by users; contributions are welcome. For example, AMD GPUs may work with an [AMD specific PyTorch](https://pytorch.org/get-started/locally/) installed in the etsam environment by the conda constructor installer post-script: [installer/post_install.sh](installer/post_install.sh).
:::

### Tested Setup

- Intel Core i7-14700K CPU
- NVIDIA GeForce RTX 4070 with 12 GB VRAM
- 32 GB DDR5 CPU RAM
- NVIDIA CUDA 13.0 and driver version 580.119.02

## Method 1: ETSAM Installer (recommended)

It installs ETSAM in a separate python environment with dependencies and model checkpoints required to run it. Everything goes into one folder (Linux: `~/etsam_env`, Windows: `%USERPROFILE%\etsam_env`). It also makes the `etsam` and `etsam-postprocess` commands available in new terminals thereafter by adding them to the PATH.

### 1. Install ETSAM

On Linux, run in a terminal:

```bash
curl -fsSL https://jianlin-cheng.github.io/ETSAM/install.sh | sh
```

On Windows, run in PowerShell:

```powershell
irm https://jianlin-cheng.github.io/ETSAM/install.ps1 | iex
```

The script asks for the installation folder and shows the default one; press Enter to accept it. (Set `ETSAM_PREFIX` to choose the folder beforehand, or `ETSAM_YES=1` to skip all questions.)

You can also download the installer manually from the [latest release](https://github.com/jianlin-cheng/ETSAM/releases/latest) and run it with `bash ETSAM-<version>-Linux-x86_64.sh`, or double-click the `ETSAM-<version>-Windows-x86_64.exe` to install it.

### 2. Run ETSAM

**_Open a new terminal_** and run the following. 
```bash
etsam input_tomogram.mrc --output-dir results/
```

If `etsam` is not found, try opening new terminal or log out and log in again, then try once more.

Run `etsam --version` to see which version is installed (this works with every installation method).

### 3. Uninstall ETSAM

To uninstall:

- On Linux, delete the `etsam_env` folder (`rm -r ~/etsam_env`) and the lines marked "added by the ETSAM installer" in `~/.profile`, `~/.bashrc` and `~/.zshrc` (and `~/.config/fish/conf.d/etsam.fish`, if you use fish shell). 

- On Windows, use the uninstaller (`Uninstall-ETSAM.exe`) in the `%USERPROFILE%\etsam_env` installation folder.

### Upgrade or Reinstall ETSAM

Run the same install command again. The script finds the existing installation (through the conda environment list `~/.conda/environments.txt` and the `.etsam-installer` file in the installation folder), tells you which version is installed, and asks what to do:

- **Upgrade**, if the latest release is newer. On Linux the installation is updated in place; on Windows, whose installer cannot update a folder in place, the old installation is uninstalled and the new one installed.
- **Reinstall**, if you already have the latest version (to repair a broken installation, for example). The old installation is removed and installed again.

The downloaded model weights (`checkpoints/`) are kept in both cases, without copying them or using extra disk space. They are downloaded again only if the download URL of the weights changed in the new version. Close ChimeraX, napari and other programs that use ETSAM first. To skip the question (in scripts), set `ETSAM_YES=1`.

## Method 2: Conda Environment from Source

Use this method if you also want to modify, [train](/ETSAM/training/) or [evaluate](/ETSAM/evaluation/) ETSAM.

### Requirements

- `conda`, preferably installed through [Miniforge](https://conda-forge.org/download/). Tested with conda 25.11.0.

### 1. Get the Code

Download the project as a [ZIP archive](https://github.com/jianlin-cheng/ETSAM/archive/refs/heads/main.zip) and extract **ETSAM-main.zip** into a preferred folder, or git clone it from the terminal:

```bash
git clone https://github.com/jianlin-cheng/ETSAM
cd ETSAM
```

### 2. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate etsam
```

:::note
Environment creation usually takes 5–10 minutes. It installs PyTorch with CUDA support from conda-forge, matched to your NVIDIA driver.
:::

### 3. Install the ETSAM Package
```bash
pip install --no-deps -e .
```

:::note
The model weights download automatically on first use. You can also download them and place the checkpoints in `checkpoints/` folder inside the cloned/downloaded repository folder:
```bash
wget -P checkpoints/ https://zenodo.org/records/17571925/files/etsam_stage1_v1.pt
wget -P checkpoints/ https://zenodo.org/records/17571925/files/etsam_stage2_v1.pt
```
:::

You need to activate ETSAM conda environment each time with: `conda activate etsam` before using `etsam` command or running `python etsam.py <>` directly from the repository folder.

## Method 3: Dev Container

A [Dev Container](https://containers.dev/) builds the conda environment from Method 2 into a Docker image and downloads the model weights into `checkpoints/` for you. This keeps the environment identical across systems. Use this if you are finding it hard to get it working in your particular Linux distro or Windows.

### Software Requirements

- [Docker](https://docs.docker.com/get-docker/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for NVIDIA GPU access.
- Either VS Code with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), or the [`devcontainer` CLI](https://github.com/devcontainers/cli).

:::tip[Data Location]
Place your input tomograms in **data/** and write output segmentations to **results/**. These folders are shared between the host and the container.
:::

:::note[Running ETSAM in the container]
The container runs ETSAM from the source folder. Use `python etsam.py` wherever this documentation uses `etsam`, and `python postprocess.py` for `etsam-postprocess`. They take exactly the same arguments.
:::

### Option A: VS Code

1. Open the ETSAM folder and choose **Reopen in Container** (command palette → *Dev Containers: Reopen in Container*).
   - The first run takes time to download the container image, install the conda environment, and download the checkpoints. Subsequent opens are instant.
2. The VS Code editor and its integrated terminal now run **inside** the container.
3. Run `nvidia-smi` in the VS Code terminal to confirm the NVIDIA GPU is accessible inside the container.

### Option B: Dev Container CLI

Start (and, on first run, build) the container:

```bash
devcontainer up
```

`devcontainer up` only starts the container; it does not give you a shell. Open one inside it with:

```bash
devcontainer exec bash
```

You are now inside the container, with the `data/` folder available.

## Model Weights

ETSAM looks for the checkpoints `etsam_stage1_v1.pt` and `etsam_stage2_v1.pt` in this order:

1. `checkpoints/` next to `etsam.py`, i.e. in the ETSAM source folder,
2. `checkpoints/` in the environment's folder, where the [ETSAM installer](#method-1-etsam-installer-recommended) puts them (`~/etsam_env/checkpoints`),
3. the folder in the `ETSAM_CHECKPOINT_DIR` environment variable, or `~/.cache/etsam` if that variable is not set.

Missing weights are downloaded into the last location the first time ETSAM runs.

## Installed Commands

| Command | Purpose |
| --- | --- |
| `etsam` | Segment membranes in a tomogram. See the [Tutorial](/ETSAM/tutorial/). |
| `etsam-postprocess` | Clean up an existing mask. See [Advanced Usage](/ETSAM/advanced/). |

Run any command with `--help` for all its options.

:::tip
Running into installation problems? See the [Troubleshooting](/ETSAM/troubleshooting/) page for common issues and fixes.
:::

Continue to the [Tutorial](/ETSAM/tutorial/) to segment your first tomogram.
