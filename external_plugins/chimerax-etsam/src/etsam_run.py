"""
Run ETSAM in its own environment, on arrays.

Shared, identical copy in the ChimeraX plugin (chimerax.etsam) and the napari
plugin (napari_etsam); keep them in sync.

ETSAM is never imported into the host application: it bundles a modified copy of
SAM 2 as the top-level package ``sam2``, which could clash with the official
``sam2`` used by other plugins. Each call runs ``etsam_worker.py`` with the ETSAM
environment's Python (found by ``etsam_env.py``). Arrays are passed as ``.npy``
files and progress comes back as JSON lines.
"""

from __future__ import annotations

import collections
import errno
import json
import os
import shutil
import subprocess
import tempfile
import threading

import numpy as np

from . import etsam_env

# Temporary .npy files go here, not the system temp dir: it may be in RAM, and
# logits can be several GB.
TEMP_DIR = os.path.join(os.path.expanduser("~"), ".cache", "etsam-plugins")


class Cancelled(Exception):
    pass


class ETSAMError(Exception):
    """ETSAM failed; ``details`` holds its log."""

    def __init__(self, message, error_type="", details=""):
        super().__init__(message)
        self.error_type = error_type
        self.details = details


def _exit_reason(code: int) -> str:
    if code in (-9, 137):
        return (
            "ETSAM was stopped by the operating system, most likely because the computer "
            "ran out of memory (RAM). Close other programs, or bin the tomogram, and try again."
        )
    if code < 0:
        return f"ETSAM crashed (signal {-code})."
    return f"ETSAM stopped unexpectedly (exit code {code})."


class WorkerRun:
    """
    One call of ``etsam_worker.py <mode>``. ``run()`` blocks; ``cancel()`` may be
    called from any thread. ``progress(fraction, message)`` receives updates.
    """

    def __init__(
        self, mode: str, array: np.ndarray | None = None, options: list[str] | None = None
    ):
        self.mode = mode
        self.array = array
        self.options = options or []
        self.cancelled = False
        self.device_name = ""
        self._proc: subprocess.Popen | None = None
        self.log: collections.deque[str] = collections.deque(maxlen=200)

    def cancel(self) -> None:
        self.cancelled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def run(self, progress=None):
        python = etsam_env.find_python()
        if python is None:
            raise ETSAMError(etsam_env.NOT_FOUND, "NotFound")
        cmd = [python, etsam_env.WORKER, self.mode, *self.options]
        work = None
        if self.array is not None:
            os.makedirs(TEMP_DIR, exist_ok=True)
            work = tempfile.mkdtemp(prefix="run-", dir=TEMP_DIR)
            src, dst = os.path.join(work, "input.npy"), os.path.join(work, "output.npy")
            cmd += ["--input", src, "--output", dst]
        try:
            if work:
                np.save(src, self.array)
            self._execute(cmd, progress)
            return np.load(dst) if work else None
        finally:
            if work:
                shutil.rmtree(work, ignore_errors=True)

    def _execute(self, cmd, progress):
        self.log.append("$ " + " ".join(cmd))
        if self.cancelled:
            raise Cancelled()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **etsam_env.subprocess_options(),
        )
        reader = threading.Thread(target=self._read_stderr, daemon=True)
        reader.start()
        done, fatal = None, None
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    if line:
                        self.log.append(line)
                    continue
                event = json.loads(line)
                kind = event.get("event")
                if kind == "start":
                    self.device_name = event.get("device_name", "")
                    if progress is not None and self.device_name:
                        slow = "" if event.get("accelerated") else " (no GPU: this will be slow)"
                        progress(0.0, f"Running on {self.device_name}{slow}")
                elif kind == "progress" and progress is not None:
                    progress(event.get("fraction", 0.0), event.get("message", ""))
                elif kind == "done":
                    done = event
                elif kind == "fatal":
                    fatal = event
            code = self._proc.wait()
            reader.join(5)
        finally:
            self._proc = None
        if self.cancelled:
            raise Cancelled()
        details = "\n".join(self.log)
        if fatal is not None:
            if fatal.get("traceback"):
                details += "\n\n" + fatal["traceback"]
            raise ETSAMError(
                fatal.get("error", "ETSAM failed."), fatal.get("error_type", ""), details
            )
        if done is None or code != 0:
            raise ETSAMError(_exit_reason(code), "Crash", details)

    def _read_stderr(self):
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip().split("\r")[-1]  # tqdm progress bars
            if line:
                self.log.append(line)


def segment_run(
    volume, normalize_method, split_processing, stage1_prompt, stage2_prompt
) -> WorkerRun:
    """A run that segments ``volume`` (Z, Y, X) and returns the stage 2 logits."""
    options = [
        "--normalize-method", normalize_method,
        "--stage1-prompt", stage1_prompt,
        "--stage2-prompt", stage2_prompt,
    ]  # fmt: skip
    if split_processing:
        options.append("--split-processing")
    return WorkerRun("segment-array", np.asarray(volume), options)


def postprocess(mask: np.ndarray) -> np.ndarray:
    """Clean up a binary mask with ETSAM's post-processing."""
    return WorkerRun("postprocess", np.asarray(mask, dtype=np.uint8)).run()


def explain(exc: BaseException) -> tuple[str, str]:
    """A title and a message, with what to do, for an error from a run."""
    import traceback

    text = str(exc)
    details = getattr(exc, "details", "") or "".join(traceback.format_tb(exc.__traceback__))
    error_type = getattr(exc, "error_type", type(exc).__name__)
    if error_type == "NotFound":
        return "ETSAM is not installed", text
    # numpy's message when a write fails, e.g. "56534016 requested and 0 written"
    if getattr(exc, "errno", None) == errno.ENOSPC or (
        "requested and" in text and "written" in text
    ):
        return "Not enough disk space", (
            f"The disk with {TEMP_DIR} is full. ETSAM passes the tomogram and its result "
            "through this folder, which needs about twice the tomogram's size (as float32). "
            "Free some disk space and try again."
        )
    if "GPU is busy" in text:
        return "The GPU is busy", text
    if error_type in ("MemoryError", "OutOfMemoryError") or "out of memory" in text.lower():
        return "Out of memory", (
            "The GPU or system RAM ran out of memory. Close other programs (especially ones using "
            "the GPU) and try again, or bin the tomogram. A ~1000 × 1000 × 400 tomogram needs about "
            "2 GB of GPU memory and 10-16 GB of RAM."
        )
    if "ensure_checkpoints" in details:
        return "Could not download the ETSAM model weights", (
            f"{text}\n\nCheck your internet connection and try again, or copy etsam_stage1_v1.pt and "
            "etsam_stage2_v1.pt into ~/.cache/etsam (or the folder set in ETSAM_CHECKPOINT_DIR)."
        )
    if "cuda" in text.lower():
        return "GPU error", (
            f"{text}\n\nETSAM's PyTorch may not support this GPU or driver. Updating the GPU "
            "driver usually helps."
        )
    return "Segmentation failed", text or type(exc).__name__
