import argparse
import hashlib
import http.client
import importlib.metadata
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from importlib.resources import files

import mrcfile
import numpy as np
import torch
from PIL import Image
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import inference
import postprocess
from sam2.utils.disk_volume import read_by_slice
from sam2.utils.volume import FusedTomogram, NormalizedTomogram, StageOneLogits

CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT_URL = "https://zenodo.org/records/17571925/files/{}"
CHECKPOINT_MD5 = {
    "etsam_stage1_v1.pt": "db2dc46d775a17619e0a657aabaf0636",
    "etsam_stage2_v1.pt": "170488377f1424be1c431bc7983c21b4",
}
# Zenodo limits the speed of each connection, so the weights are downloaded in
# parallel parts (measured: 1 connection 0.45 MB/s, 16 3.8 MB/s, 32 6.9 MB/s,
# 64 13.3 MB/s).
DOWNLOAD_CONNECTIONS = int(os.environ.get("ETSAM_DOWNLOAD_CONNECTIONS", "64"))


def min_max_normalize(array):
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    return array


def min_max_postive_values_normalize(array):
    array[array < 0] = 0
    array = array / np.max(array)
    return array


def softplus_minmax_normalize(array):
    """Normalize a float32 array in place and return it."""
    tensor = torch.from_numpy(array)
    torch.ops.aten.softplus.out(tensor, 1, 20, out=tensor)  # softplus(beta=1, threshold=20)
    minimum, maximum = np.min(array), np.max(array)
    array -= minimum
    array /= maximum - minimum
    return array


def save_mask(mask, voxel_size, output_map_file_path):
    with mrcfile.new(output_map_file_path, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.uint8))
        mrc.voxel_size = voxel_size


def save_logits(logits, voxel_size, output_map_file_path):
    with mrcfile.new_mmap(
        output_map_file_path, shape=logits.shape, mrc_mode=2, fill=0, overwrite=True
    ) as mrc:
        mrc.set_data(logits)
        mrc.voxel_size = voxel_size


def split_into_quadrants(tomogram_data):
    """
    Split a (Z, Y, X) tomogram into 4 sub-tomograms by halving Y and X.

    Returns a list of (sub_tomogram, (y_start, y_end, x_start, x_end)) tuples.
    """
    _, H, W = tomogram_data.shape
    y_mid = H // 2
    x_mid = W // 2

    quadrants = [
        (tomogram_data[:, :y_mid, :x_mid], (0, y_mid, 0, x_mid)),
        (tomogram_data[:, :y_mid, x_mid:], (0, y_mid, x_mid, W)),
        (tomogram_data[:, y_mid:, :x_mid], (y_mid, H, 0, x_mid)),
        (tomogram_data[:, y_mid:, x_mid:], (y_mid, H, x_mid, W)),
    ]
    return quadrants


def run_two_stage_inference_on_tomogram(
    tomogram_data,
    config,
    stage1_ckpt_path,
    stage2_ckpt_path,
    stage1_prompt,
    stage2_prompt,
    grid_interval,
    stage1_logit_threshold,
    device="cuda",
    progress=None,
    output=None,
):
    """
    Run the two-stage ETSAM pipeline on a (Z, Y, X) volume (full tomogram or a sub-region).

    Runs stage 1 on the normalized volume, fuses its logits with the volume, then runs stage 2.

    Writes the stage-2 logits into output, a float32 array matching the input
    spatial shape (allocated if None), and returns it. Stage 1 stores its
    logits in output too: stage 2 overwrites each slice after reading it.
    progress(stage, slices_done, total_slices) is called after each slice.
    """
    if output is None:
        output = np.empty(tomogram_data.shape, dtype=np.float32)
    stage1_logits = StageOneLogits(
        output,
        prompt_interval=grid_interval if stage2_prompt == "etsam_stage1_partial" else None,
        prompt_threshold=stage1_logit_threshold,
    )

    print(f"==> Stage 1 prediction (prompt: {stage1_prompt})")
    inference.etsam_inference(
        tomogram_data,
        config_path=config,
        ckpt_path=stage1_ckpt_path,
        prompt_method=stage1_prompt,
        grid_interval=grid_interval,
        device=device,
        progress=None if progress is None else lambda done, total: progress(1, done, total),
        output=stage1_logits,
    )

    # Fused slice by slice, when the stage-2 predictor reads them.
    combined_input = FusedTomogram(tomogram_data, stage1_logits)

    print(f"==> Stage 2 prediction (prompt: {stage2_prompt})")
    inference.etsam_inference(
        combined_input,
        stage1_logits.prompts,
        config_path=config,
        ckpt_path=stage2_ckpt_path,
        prompt_method=stage2_prompt,
        grid_interval=grid_interval,
        device=device,
        progress=None if progress is None else lambda done, total: progress(2, done, total),
        output=output,
    )

    return output


def ensure_checkpoints(progress=None):
    """
    Return the stage 1 and stage 2 checkpoint paths, downloading them on first use.

    Uses checkpoints/ next to this file if present, then checkpoints/ in the Python
    environment (where the ETSAM installer puts them), otherwise $ETSAM_CHECKPOINT_DIR
    (default ~/.cache/etsam). progress(fraction, message) reports the download.
    """
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    env_dir = os.path.join(sys.prefix, "checkpoints")
    cache_dir = os.environ.get(
        "ETSAM_CHECKPOINT_DIR", os.path.join(os.path.expanduser("~"), ".cache", "etsam")
    )
    paths = []
    for name, md5 in CHECKPOINT_MD5.items():
        url = CHECKPOINT_URL.format(name)
        path = os.path.join(repo_dir, name)
        if not os.path.exists(path):
            path = os.path.join(env_dir, name)
        if not os.path.exists(path):
            path = os.path.join(cache_dir, name)
        if os.path.exists(path) and not path.startswith(repo_dir):
            # Keep a downloaded checkpoint unless it came from another URL than
            # the one this version of ETSAM uses (e.g. after an upgrade).
            if _recorded_source(path) not in (None, url):
                print(f"==> {name} is outdated (new download URL), downloading it again", flush=True)
                os.remove(path)
            else:
                _record_source(path, url)
        if not os.path.exists(path):
            print(f"==> Downloading {name} to {cache_dir}", flush=True)
            os.makedirs(cache_dir, exist_ok=True)
            _download(url, path + ".part", name, progress)
            if _md5(path + ".part") != md5:
                os.remove(path + ".part")
                raise RuntimeError(f"Downloaded {name} is corrupt, please try again.")
            os.replace(path + ".part", path)
            _record_source(path, url)
        paths.append(path)
    return paths


def _recorded_source(path):
    """URL a checkpoint was downloaded from (saved next to it), or None if unknown."""
    try:
        with open(path + ".url") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _record_source(path, url):
    if _recorded_source(path) == url:
        return
    try:
        with open(path + ".url", "w") as f:
            f.write(url + "\n")
    except OSError:
        pass  # read-only folder: the checkpoint is still used


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, path, name, progress=None, connections=None, retries=5):
    """
    Download url to path in parallel parts (HTTP range requests), retrying each
    part from where it stopped. Falls back to one connection if the server does
    not support ranges. Reports progress(fraction, message), or prints it.
    """
    connections = max(1, connections or DOWNLOAD_CONNECTIONS)
    # Size, and whether ranges work: ask for the first byte.
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60
    ) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status == 206 and "/" in content_range:
            total = int(content_range.rsplit("/", 1)[1])
        else:
            total, connections = int(response.headers.get("Content-Length", 0)), 1
    if not total:
        connections = 1
    part_size = -(-total // connections) if total else 0
    parts = [(i * part_size, min(total, (i + 1) * part_size) - 1) for i in range(connections)]
    with open(path, "wb") as f:
        f.truncate(total)
    received = [0]
    lock = threading.Lock()
    failed = threading.Event()

    def fetch(start, end):
        position, attempt = start, 0
        while total == 0 or position <= end:
            headers = {} if total == 0 else {"Range": f"bytes={position}-{end}"}
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response, open(path, "r+b") as f:
                    if headers and response.status != 206:
                        raise RuntimeError(f"the server ignored the download range for {name}")
                    f.seek(position)
                    while chunk := response.read(1 << 20):
                        if failed.is_set():
                            return
                        f.write(chunk)
                        position += len(chunk)
                        with lock:
                            received[0] += len(chunk)
                if total == 0:
                    return
            except (OSError, urllib.error.URLError, http.client.HTTPException) as e:
                attempt += 1
                if attempt > retries or failed.is_set():
                    raise RuntimeError(f"Could not download {name}: {e}") from e
                time.sleep(min(30, 2 * attempt))

    started = time.monotonic()
    last_print = 0.0
    tty = sys.stdout.isatty()

    def report(final=False):
        nonlocal last_print
        done, elapsed = received[0], max(time.monotonic() - started, 1e-6)
        speed = done / elapsed / 1e6
        fraction = done / total if total else 0.0
        message = f"Downloading {name} ({done >> 20} / {total >> 20} MB, {speed:.1f} MB/s)"
        if progress is not None:
            progress(fraction, message)
        elif tty:
            print(f"\r    {done >> 20} / {total >> 20} MB ({100 * fraction:.0f}%), {speed:.1f} MB/s   ",
                  end="\n" if final else "", flush=True)
        elif final or time.monotonic() - last_print >= 10:
            # Not a terminal (e.g. the installer run by a plugin): a line every 10 s.
            last_print = time.monotonic()
            print(f"    {done >> 20} / {total >> 20} MB ({100 * fraction:.0f}%), {speed:.1f} MB/s", flush=True)

    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        futures = [pool.submit(fetch, start, end) for start, end in parts]
        try:
            while not all(future.done() for future in futures):
                time.sleep(0.5)
                report()
                for future in futures:
                    if future.done() and future.exception():
                        raise future.exception()
        except BaseException:
            failed.set()  # stop the other parts
            raise
    report(final=True)
    if total and received[0] != total:
        raise RuntimeError(f"Could not download {name}: incomplete ({received[0]} of {total} bytes).")


def segment(
    tomogram_data,
    normalize_method="min_max_positive_values",
    split_processing=False,
    stage1_prompt="grid_zero",
    stage2_prompt="zero",
    progress=None,
):
    """
    Run ETSAM on a (Z, Y, X) tomogram array and return the stage-2 logits.

    progress(fraction, message) is called after each slice; raise from it to cancel.
    """
    stage1_ckpt_path, stage2_ckpt_path = ensure_checkpoints(progress)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid_interval = 10 if split_processing else 50
    # A memory-mapped tomogram (from the CLI or a plugin) is read with file
    # reads, so that it does not accumulate in RAM as its slices are used.
    tomogram_data = read_by_slice(tomogram_data)

    print(f"==> Normalizing tomogram data with {normalize_method} method")
    if normalize_method == "min_max_positive_values":
        # Normalized slice by slice, when the predictor reads them.
        normalized_tomogram_data = NormalizedTomogram(tomogram_data)
    elif normalize_method == "softplus_minmax":
        normalized_tomogram_data = softplus_minmax_normalize(np.array(tomogram_data, dtype=np.float32))

    if split_processing:
        quadrants = split_into_quadrants(normalized_tomogram_data)
        quadrant_labels = ["top-left", "top-right", "bottom-left", "bottom-right"]
        print(
            f"==> Split tomogram into {len(quadrants)} sub-tomograms (split-processing mode, grid_interval={grid_interval})"
        )
        for label, (sub, (y0, y1, x0, x1)) in zip(quadrant_labels, quadrants):
            print(f"    {label}: shape {sub.shape}, Y[{y0}:{y1}] X[{x0}:{x1}]")
    else:
        quadrants = [(normalized_tomogram_data, None)]
        quadrant_labels = [""]

    logits = np.empty(tomogram_data.shape, dtype=np.float32)
    for i, (label, (sub_tomogram, coords)) in enumerate(zip(quadrant_labels, quadrants)):
        if split_processing:
            print(f"\n==> Processing quadrant: {label}")

        def slice_progress(stage, done, total):
            if progress is not None:
                prefix = f"Quadrant {i + 1}/4 · " if split_processing else ""
                fraction = (2 * i + stage - 1 + done / total) / (2 * len(quadrants))
                progress(fraction, f"{prefix}Stage {stage} · slice {done}/{total}")

        if coords is None:
            output = logits
        else:
            y_start, y_end, x_start, x_end = coords
            output = logits[:, y_start:y_end, x_start:x_end]
        run_two_stage_inference_on_tomogram(
            sub_tomogram,
            CONFIG,
            stage1_ckpt_path,
            stage2_ckpt_path,
            stage1_prompt,
            stage2_prompt,
            grid_interval=grid_interval,
            stage1_logit_threshold=-0.25,
            device=device,
            progress=slice_progress,
            output=output,
        )

    return logits


BLUE = "#29abe2"


@lru_cache(maxsize=1)
def _native_image_type():
    """Probe lazily, only for a real interactive CLI, never on module import."""
    if not all(stream is not None and stream.isatty()
               for stream in (sys.stdin, sys.stdout, sys.__stdin__, sys.__stdout__)):
        return None
    # A terminal that ignores graphics queries is an ordinary unsupported case.
    # Suppress the library's probe warnings while preserving its logging settings.
    logger = logging.getLogger("textual_image._terminal")
    was_disabled = logger.disabled
    try:
        logger.disabled = True
        from textual_image.renderable import Image, SixelImage, TGPImage
        return Image if Image in (SixelImage, TGPImage) else None
    except Exception:
        # Branding must not prevent segmentation when terminal probing fails.
        return None
    finally:
        logger.disabled = was_disabled


def version():
    """Read the same package version for installer, pip, and conda installations."""
    try:
        return importlib.metadata.version("etsam")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (ETSAM is not installed with pip)"


def banner(console=None, *, show_logo=True):
    """Show a native image only when supported; otherwise show just name and version."""
    console = console if console is not None else Console(highlight=False)
    text = Text("ETSAM", style=f"bold {BLUE}")
    text.append(f"\nVersion: {version()}\n", style="default")
    text.append("Segment membranes in cryo-electron tomograms", style="default")
    try:
        "▀╭╮╰╯".encode(console.encoding)
        unicode_ok = True
    except (UnicodeEncodeError, LookupError):
        unicode_ok = False

    content = text
    image_type = (
        _native_image_type()
        if show_logo and console.is_terminal and not console.no_color and unicode_ok else None
    )
    if image_type is not None:
        # Pass the original transparent PNG to the native renderer. No block,
        # braille or Unicode-image fallback is used on unsupported terminals.
        stacked = console.width < 64
        logo_width = min(12, max(6, console.width - (6 if stacked else 34)))
        try:
            with files("etsam_assets").joinpath("etsam-logo.png").open("rb") as image_file:
                with Image.open(image_file) as image:
                    logo = image_type(image.copy(), width=logo_width, height="auto")
            if stacked:
                content = Group(logo, text)
            else:
                layout = Table.grid(padding=(0, 1))
                layout.add_column(width=logo_width)
                layout.add_column()
                layout.add_row(logo, text)
                content = layout
        except (OSError, ValueError):
            content = text

    return Panel(
        content, box=box.ROUNDED if unicode_ok else box.ASCII,
        border_style=BLUE, padding=(0, 1), expand=False,
    )


def _print_banner():
    console = Console(file=sys.stdout, highlight=False)
    try:
        console.print(banner(console))
    except Exception:
        # A failed graphics operation must never stop an ETSAM command.
        console.print(banner(console, show_logo=False))


def print_banner():
    """Print once at command startup; argparse handles the version-only case."""
    if not any(arg in ("-V", "--version") for arg in sys.argv[1:]):
        _print_banner()


class VersionAction(argparse.Action):
    """Render directly so argparse does not reflow the banner's whitespace."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        _print_banner()
        parser.exit()


def main():
    print_banner()
    parser = argparse.ArgumentParser(
        prog="etsam",
        usage="etsam [options] TOMOGRAM",
        epilog="Documentation: https://jianlin-cheng.github.io/ETSAM/",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=24),
    )
    parser.add_argument(
        "input_tomogram_file_path", metavar="TOMOGRAM", help="input tomogram (.mrc)"
    )
    parser.add_argument(
        "-V", "--version", action=VersionAction, help="show the version and exit"
    )
    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir", metavar="DIR", default="./",
        help="folder for the results (default: current folder)",
    )  # fmt: skip
    output.add_argument(
        "--logit-threshold", metavar="T", type=float, default=-0.25,
        help="logit threshold for the mask (default: -0.25)",
    )  # fmt: skip
    output.add_argument(
        "--store-logits", action="store_true",
        help="also save the logits (membrane scores)",
    )  # fmt: skip
    segmentation = parser.add_argument_group("segmentation")
    segmentation.add_argument(
        "--split-processing", action="store_true",
        help="segment the 4 Y/X quadrants separately",
    )  # fmt: skip
    segmentation.add_argument(
        "--normalize-method", metavar="M", default="min_max_positive_values",
        choices=["min_max_positive_values", "softplus_minmax"],
        help="min_max_positive_values (default), softplus_minmax",
    )  # fmt: skip
    segmentation.add_argument(
        "--stage1-prompt", metavar="P", default="grid_zero",
        choices=["zero", "grid", "grid_zero"],
        help="zero, grid, grid_zero (default)",
    )  # fmt: skip
    segmentation.add_argument(
        "--stage2-prompt", metavar="P", default="zero",
        choices=["zero", "grid", "grid_zero", "etsam_stage1_partial"],
        help="zero (default), grid, grid_zero, etsam_stage1_partial",
    )  # fmt: skip
    postprocess.add_postprocess_arguments(parser.add_argument_group("post-processing"))

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    stage1_prompt = args.stage1_prompt
    stage2_prompt = args.stage2_prompt
    store_logits = args.store_logits
    stage2_logit_threshold = args.logit_threshold
    split_processing = args.split_processing
    input_tomogram_file_path = os.path.abspath(args.input_tomogram_file_path)
    output_dir = os.path.abspath(args.output_dir)
    normalize_method = args.normalize_method

    try:
        print("==> Reading input tomogram:", input_tomogram_file_path)
        with mrcfile.mmap(input_tomogram_file_path, mode="r") as mrc:
            voxel_size = mrc.voxel_size.copy()
            input_tomogram_shape = mrc.data.shape
            if not torch.cuda.is_available():
                print("Warning: no CUDA GPU found, running on the CPU (very slow)")
            print(f"Input tomogram shape: {input_tomogram_shape}, voxel size: {voxel_size}")

            # The tomogram stays memory-mapped: its slices are read as needed.
            logits = segment(
                mrc.data,
                normalize_method,
                split_processing,
                stage1_prompt,
                stage2_prompt,
            )

        if split_processing:
            output_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_split_processing_mask.mrc",
            )
            output_logits_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_split_processing_logits.mrc",
            )
            output_post_processed_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_split_processing_post_processed_mask.mrc",
            )
        else:
            output_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_mask.mrc",
            )
            output_logits_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_logits.mrc",
            )
            output_post_processed_mask_file_path = os.path.join(
                output_dir,
                f"{os.path.basename(input_tomogram_file_path).split('.')[0]}_etsam_predicted_post_processed_mask.mrc",
            )

        mask = (logits > stage2_logit_threshold).astype(np.uint8)

        print("==> Saving predicted mask")
        os.makedirs(output_dir, exist_ok=True)
        save_mask(mask, voxel_size, output_mask_file_path)

        if store_logits:
            print("==> Saving segmentation logits")
            os.makedirs(os.path.dirname(output_logits_file_path), exist_ok=True)
            save_logits(logits, voxel_size, output_logits_file_path)

        if postprocess.postprocess_requested(args):
            post_processed_mask = postprocess.postprocess_cli(mask, args)
            print("==> Saving post-processed mask")
            os.makedirs(
                os.path.dirname(output_post_processed_mask_file_path), exist_ok=True
            )
            save_mask(
                post_processed_mask, voxel_size, output_post_processed_mask_file_path
            )

        print(
            "-------------------------------------------------------------------------------------------------"
        )

    except Exception as e:
        print("Error: " + str(e))


if __name__ == "__main__":
    main()
