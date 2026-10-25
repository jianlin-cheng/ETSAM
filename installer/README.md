<img src="../ETSAM_icon.svg" alt="ETSAM membrane icon" width="200" height="160" />

# ETSAM installer

A self-contained installer for ETSAM, built with [conda constructor](https://conda.github.io/constructor/).
It installs Python, ETSAM and its dependencies into one folder (by default `~/etsam_env`, or
`%USERPROFILE%\etsam_env` on Windows) and registers it with conda, so that `conda env list` shows it and the
ETSAM plugins for ChimeraX and napari can find it. It also makes the `etsam` and `etsam-postprocess` commands
available in new terminals. They are in a folder of their own, `commands/` in the installation folder, and
only that folder is added to the `PATH`: on Linux in `~/.profile`, `~/.bashrc`, `~/.zshrc` and fish's
`conf.d` (for the shells in use), on macOS also in `~/.zshrc` (created if needed: zsh is the default shell)
and `~/.bash_profile` (if it exists), on Windows in the user `PATH` (removed again by the uninstaller,
`pre_uninstall.bat`). The environment's Python, pip and conda are not put on the `PATH`. Each command runs
ETSAM with the environment's Python in isolated mode (`python -I`), so the user's Python settings and
packages are not used. Set `ETSAM_NO_PATH=1` to skip adding the folder to the `PATH`.

Currently supported: **NVIDIA GPUs on Linux and Windows (x86-64)**, and **Apple silicon Macs (arm64) with
macOS 14 or newer**, where PyTorch uses the GPU through Metal (MPS).

| File | Purpose |
| --- | --- |
| `recipe/recipe.yaml` | conda package of ETSAM (noarch), built with rattler-build |
| `construct.yaml` | installer definition: Python 3.12, conda, the etsam package |
| `pre_install.sh`, `pre_install.bat` | stop early, with a clear message, if no NVIDIA driver is found (macOS: if the Mac has no Apple silicon or macOS is older than 14) |
| `post_install.sh`, `post_install.bat` | install PyTorch from conda-forge for the GPU, clean the package cache, and download the model weights (1.9 GB) into `<prefix>/checkpoints` |
| `build.sh` | build the installer for the current platform |

PyTorch is installed by the post-install script rather than shipped in the installer: a CUDA build of
PyTorch is about 3 GB, more than GitHub allows per release file, and conda-forge's `pytorch-gpu` picks the
CUDA build that matches the installed NVIDIA driver. The PyTorch version is pinned (`PYTORCH_VERSION` in
the post-install scripts) to the version ETSAM is validated with.

## Building

```bash
conda create -n etsam-build -c conda-forge rattler-build constructor
conda activate etsam-build
installer/build.sh
```

The installer is written to `installer/build/installer/`: `ETSAM-<version>-Linux-x86_64.sh` on Linux,
`ETSAM-<version>-Windows-x86_64.exe` on Windows and `ETSAM-<version>-macOS-arm64.sh` on an Apple silicon Mac
(constructor builds installers only for the platform it runs on). The
version comes from `pyproject.toml`.

The GitHub Actions workflow `.github/workflows/installer.yml` builds and tests the three installers. On
macOS (an Apple silicon runner) it also tests ETSAM end to end on the GPU: the `etsam` command (compared with
a CPU run), the plugins' worker, the napari plugin's panel and the plugins' "Install ETSAM", on the 32 center
slices of a real tomogram (`tests/fetch_test_tomogram.py`, `tests/e2e/`). It runs on
demand, and as part of a release (`.github/workflows/release.yml`, run by a version tag `v*`), which
attaches them to the GitHub release; see `RELEASING.md`. The plugins download the
installer from the latest release: `releases/latest` redirects to the tag (`.../releases/tag/v1.2.3`), and the file is `releases/download/v1.2.3/ETSAM-1.2.3-<OS>-x86_64.<ext>` (the file names carry the version).

## Installing

```bash
bash ETSAM-<version>-Linux-x86_64.sh            # interactive
bash ETSAM-<version>-Linux-x86_64.sh -b -p ~/etsam_env   # silent
bash ETSAM-<version>-macOS-arm64.sh -b -p ~/etsam_env     # macOS, silent
```

On Windows, run `ETSAM-<version>-Windows-x86_64.exe`; for a silent install:
`ETSAM-<version>-Windows-x86_64.exe /S /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /D=%USERPROFILE%\etsam_env`.

## Upgrading

`docs/public/install.sh` and `install.ps1` (the one-line installers) find an existing installation through
`~/.conda/environments.txt` (`register_envs: true` in `construct.yaml`) and the `.etsam-installer` marker
(first word: the installer version) in the listed folder, and ask whether to upgrade (newer release) or
reinstall (same version); `ETSAM_YES=1` answers yes. The latest version is taken from the redirect of
`releases/latest`.

- **Linux and macOS:** an upgrade runs the installer with `-u`, which updates the folder in place (constructor
  re-links the packages, and the post-install script runs again). A reinstall deletes the folder and installs again.
- **Windows:** the NSIS installer refuses a non-empty folder, so both run the installer's uninstaller
  (`Uninstall-ETSAM.exe /S /_?=<folder>`) and then the new installer.
- **Model weights:** `checkpoints/` is renamed next to the installation folder (`<folder>.checkpoints-keep`; a rename,
  no copy) while the old installation is removed, and moved back into the new one. The scripts set
  `ETSAM_SKIP_WEIGHTS=1` for the installer in that case and then run `etsam.ensure_checkpoints()`, which keeps the
  files unless their download URL changed: a `<name>.url` file next to each checkpoint records the URL it came from.

The ChimeraX and napari plugins run the silent install themselves the first time they start, if no
ETSAM environment is found.

## Testing without a GPU

```bash
ETSAM_SKIP_GPU_CHECK=1 ETSAM_ACCELERATOR=cpu ETSAM_SKIP_WEIGHTS=1 bash ETSAM-<version>-Linux-x86_64.sh -b -p /tmp/etsam-test
```

`ETSAM_SKIP_GPU_CHECK=1` skips the NVIDIA check, and `ETSAM_ACCELERATOR=cpu` installs the CPU build of
PyTorch. These are for testing the installer on machines without a GPU only; running ETSAM without an
NVIDIA GPU is not supported. `ETSAM_SKIP_WEIGHTS=1` skips downloading the model weights; ETSAM then downloads them on first use
(into `~/.cache/etsam`). The CI workflow tests the installers this way.

## Supporting other GPUs

The scripts are written so that other accelerators can be added: a detection step in `pre_install.*` and
a new case in `post_install.*` that installs the matching PyTorch build. For example:

- **AMD GPUs (ROCm, Linux):** detect ROCm (`rocminfo`, `/opt/rocm`) and install a ROCm build of PyTorch.
- **Intel GPUs:** detect the GPU and install the XPU build of PyTorch.

These need testing on such hardware. Contributions are welcome: please open an issue or a pull request at
<https://github.com/jianlin-cheng/ETSAM>.
