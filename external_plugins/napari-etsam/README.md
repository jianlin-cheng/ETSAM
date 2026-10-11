# ETSAM Plugin for napari

The **napari-etsam** plugin runs ETSAM inside [napari](https://napari.org), a free image viewer for multi-dimensional data. You open a tomogram, install and activate plugin, click **Segment membranes**. The membrane mask appears as an new label layer that you can refine, inspect and save.

> **Note**
> On first start, the plugin installs ETSAM in its own environment (if not already installed using the [ETSAM installer](https://jianlin-cheng.github.io/ETSAM/installation)).

![ETSAM membrane segmentation overlaid on a tomogram in napari](https://raw.githubusercontent.com/jianlin-cheng/ETSAM/main/docs/public/napari/napari-result.png)

*ETSAM membrane mask (yellow) on EMD-4949 in napari, with the ETSAM panel on the right.*

## Requirements

| | Minimum | Recommended |
| --- | --- | --- |
| GPU | NVIDIA GPU with 2 GB VRAM | NVIDIA GPU with 4 GB+ VRAM |
| RAM | 16 GB | 32 GB |
| Operating system | **Linux or Windows, with an NVIDIA GPU** | |

## Installation

### 1. Install the plugin

**From napari (recommended):**

1. Install napari if you don't have it yet. The [napari bundle](https://napari.org/stable/tutorials/fundamentals/installation_bundle_conda.html) is the easiest option.
2. In napari, open **Plugins ▸ Install/Uninstall Plugins…**.
3. Search for **ETSAM**, click **Install** next to *napari-etsam*, and restart napari.

**With pip**, in napari's Python environment:

```bash
pip install napari-etsam
```

> **Tip**
> To install the plugin from a checkout of the [ETSAM repository](https://github.com/jianlin-cheng/ETSAM) instead, use the Python of napari's environment:
>
> ```bash
> /path/to/napari/python -m pip install ./ETSAM/external_plugins/napari-etsam
> ```

### 2. First start: ETSAM is installed for you

ETSAM itself runs in its own environment, separate from napari. When you first open the ETSAM panel, the plugin looks for it and, if it is not installed yet, runs the [ETSAM installer](https://jianlin-cheng.github.io/ETSAM/installation/#method-1-etsam-installer-recommended) in the background. The installer puts ETSAM into `~/etsam_env` (on Windows, `%USERPROFILE%\etsam_env`) and sets up the conda environment and the model weights (about 5 GB in total), so this takes a few minutes; the panel shows its progress. Subsequent use starts are immediate.

## Segmenting a Tomogram

1. **Open the panel**: **Plugins ▸ ETSAM Membrane Segmentation (napari-etsam)**.
2. **Open a tomogram**: drag and drop an `.mrc` file onto napari, or click **Open…** in the panel. The new layer is selected as the *Tomogram* automatically.
3. Optionally, tick **Clean up mask** or **Closely spaced membranes** (see [Options](#options)).
4. Click **Segment membranes**. The progress bar follows ETSAM through Stage 1 and Stage 2 slice by slice, then through clean-up. Click **Cancel** to stop at any time.

![ETSAM progress in napari](https://raw.githubusercontent.com/jianlin-cheng/ETSAM/main/docs/public/napari/napari-progress.png)

*Progress is reported per slice for each stage, with an estimate of the time remaining.*

5. When finished, a **`<tomogram> membranes`** label layer is added on top of the tomogram, and the panel shows the run time and the fraction of voxels classified as membrane.
6. **Refine**: move the **Threshold** slider. The mask updates instantly, and clean-up is re-applied in the background if enabled. Lower values find more (and fainter) membrane; higher values give fewer false positives. **Reset** returns to the default `-0.25`. You can also correct the mask by hand with napari's label tools (paint, erase, fill).
7. **Save**: click **Save mask…** (the mask at the current threshold, including manual edits) or **Save logits…** (raw membrane scores). Files are written as `.mrc` with the tomogram's voxel size, so they overlay directly in ChimeraX, IMOD or other tools. *File ▸ Save Selected Layers…* with a `.mrc` file name works too.

## Options

The main options are in the panel. The rest are under **Advanced settings**.

<img src="https://raw.githubusercontent.com/jianlin-cheng/ETSAM/main/docs/public/napari/napari-advanced.png" alt="ETSAM advanced settings in napari" width="420">

| Option | Command-line equivalent | Description |
| --- | --- | --- |
| Threshold | `--logit-threshold` | Binarization threshold of the Stage 2 logits (default `-0.25`). Adjustable live after a run. |
| Closely spaced membranes | `--split-processing` | Segments the four Y/X quadrants separately with a denser prompt grid; separates closely apposed membranes. About 4× slower. |
| Clean up mask | `--post-process` | Removes thin noise blobs and cuts short bridges between parallel membranes, with the default [post-processing](https://jianlin-cheng.github.io/ETSAM/advanced/#post-processing-with-etsam-postprocess) settings. Compare with the raw result, because it can remove small real membranes. |
| Normalization | `--normalize-method` | Try *Softplus* if membranes are missed with the default. |
| Stage 1 / Stage 2 prompt | `--stage1-prompt`, `--stage2-prompt` | Prompting strategies; see [Advanced Usage](https://jianlin-cheng.github.io/ETSAM/advanced/#prompt-methods). |

The plugin and the `etsam` command produce the same results for the same settings.

## Troubleshooting

When something goes wrong, the panel shows a red box with the problem and a suggested fix. **Show details** displays the full error, and **Copy error report** copies it with your system information (versions, GPU) for a [GitHub issue](https://github.com/jianlin-cheng/ETSAM/issues).

<img src="https://raw.githubusercontent.com/jianlin-cheng/ETSAM/main/docs/public/napari/napari-error.png" alt="ETSAM error panel in napari" width="420">

| Message | What to do |
| --- | --- |
| *No tomogram selected* | Open an `.mrc` file first (drag and drop, *File ▸ Open* or **Open…**). |
| *This layer cannot be segmented* | ETSAM needs a single 3D grayscale volume (Z, Y, X). Select the tomogram layer rather than a 2D image, RGB image or a time series. |
| *Installing ETSAM failed: … requires an NVIDIA GPU* | The installer supports NVIDIA GPUs only. Install ETSAM yourself as described in [Installation](https://jianlin-cheng.github.io/ETSAM/installation/#method-2-conda-environment-from-source). |
| *Installing ETSAM failed: … the folder already exists* | An earlier installation was interrupted. Delete the `etsam` folder in your home folder and reopen the panel. |
| *ETSAM is not installed* | This computer is not supported by the installer. Install ETSAM yourself (see [Installation](#installation)), or set `ETSAM_PYTHON`. |
| *ETSAM is not installed in the environment …* | That environment lacks ETSAM. Run step 3 of the conda installation (`pip install --no-deps -e .` in the ETSAM folder). |
| *Running on CPU (no GPU: this will be slow)* although the computer has an NVIDIA GPU | Update the NVIDIA driver, and check that PyTorch in the `etsam` environment can use the GPU (see [Verify the Installation](https://jianlin-cheng.github.io/ETSAM/installation/#4-verify-the-installation)). |
| *The GPU is busy* | Another program keeps ETSAM from using the GPU. Close other programs that use the GPU, or restart napari. |
| *Out of memory* | Close other programs (especially ones using the GPU) and try again, or bin the tomogram (fewer slices need less RAM). |
| *Could not download the ETSAM model weights* | Check your internet connection or proxy and retry. Offline machines: copy `etsam_stage1_v1.pt` and `etsam_stage2_v1.pt` into the weights folder (see [Model Weights](https://jianlin-cheng.github.io/ETSAM/installation/)). |
| Empty or incomplete mask | Lower the threshold, try *Softplus* normalization, a `grid` Stage 1 prompt, or *Closely spaced membranes*. See [Troubleshooting](https://jianlin-cheng.github.io/ETSAM/troubleshooting/#empty-or-incomplete-membrane-predictions). |
| *ETSAM stopped unexpectedly* | Report it with **Copy error report**. |

## Uninstalling

Remove the plugin with **Plugins ▸ Install/Uninstall Plugins ▸ napari-etsam ▸ Uninstall**, or `pip uninstall napari-etsam`. To remove ETSAM completely, refer uninstall section in [Installation](https://jianlin-cheng.github.io/ETSAM/installation/#3-uninstall-etsam) guide.
