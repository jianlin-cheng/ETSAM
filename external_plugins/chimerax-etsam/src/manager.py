"""
Per-session ETSAM state shared by the tool window and the ``etsam`` command.

The manager segments open volumes with ETSAM in its own environment (see
etsam_env.py and etsam_run.py) and adds the membrane logits as a new volume,
shown as a surface at the threshold level. The tool's threshold and the volume's
surface level (Volume Viewer, "volume" command) are the same value. Results are
saved only when asked to. Listeners (the tool window) are told about every
change with ``listener(event, *args)``:

* ``"status"``                   busy state changed
* ``"setup", state, message``    ETSAM environment: checking, installing, ready,
                                 missing (no installer for this computer) or failed
* ``"progress", fraction, text`` segmentation progress
* ``"done", result``             a volume was segmented (a Result)
* ``"failed", exc``              segmentation failed (etsam_run.Cancelled if cancelled)
* ``"level", result``             a result's threshold (surface level) changed
* ``"threshold", result, text``  statistics (and clean-up) for the new threshold are ready
"""

from __future__ import annotations

import html
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
from chimerax.core.settings import Settings

from . import etsam_env, etsam_run

RESULT_COLOR = "#e8a33d"  # membranes: warm gold, visible on grey tomograms
DEFAULT_THRESHOLD = -0.25
NORMALIZATIONS = ["min_max_positive_values", "softplus_minmax"]
STAGE1_PROMPTS = ["grid_zero", "grid", "zero"]
STAGE2_PROMPTS = ["zero", "grid", "grid_zero", "etsam_stage1_partial"]
# Shown value for voxels that clean-up removes: below any threshold.
REMOVED = np.float32(-100.0)


class ETSAMSettings(Settings):
    AUTO_SAVE = {
        "threshold": DEFAULT_THRESHOLD,
        "split_processing": False,
        "cleanup": False,
        "normalize_method": NORMALIZATIONS[0],
        "stage1_prompt": STAGE1_PROMPTS[0],
        "stage2_prompt": STAGE2_PROMPTS[0],
        # See apply_web_gpu_workaround()
        "web_panels_without_gpu": True,
    }


def nvidia_linux() -> bool:
    import sys

    return sys.platform.startswith("linux") and os.path.exists("/proc/driver/nvidia/version")


def apply_web_gpu_workaround(session) -> None:
    """
    Let ETSAM use an NVIDIA GPU while ChimeraX is open (Linux).

    With some NVIDIA drivers, the GPU process of QtWebEngine (the embedded web
    browser that draws ChimeraX's Log and Help panels) keeps any other program
    from creating a CUDA context ("CUDA-capable device(s) is/are busy or
    unavailable"). Rendering those HTML panels in software avoids that and does
    not affect ChimeraX's 3D graphics. Must run before the ChimeraX UI is built.
    """
    if not nvidia_linux():
        return
    settings = ETSAMSettings(session, "ETSAM")
    if not settings.web_panels_without_gpu:
        return
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    if "--disable-gpu" not in flags.split():
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (flags + " --disable-gpu").strip()


def get_manager(session) -> "ETSAMManager":
    manager = getattr(session, "_etsam_manager", None)
    if manager is None:
        manager = ETSAMManager(session)
        session._etsam_manager = manager
    return manager


@dataclass
class Result:
    """A segmented volume: its logits, and the volume that shows them."""

    source: object  # the segmented Volume
    logits: np.ndarray  # float32 (Z, Y, X) stage 2 logits, as computed
    model: object  # Volume of the logits (cleaned up if ``cleanup``), surface at ``threshold``
    cleanup: bool
    split: bool
    threshold: float
    seconds: float
    cleaned_at: float | None = None  # threshold of the clean-up shown in the volume

    def mask(self) -> np.ndarray:
        """The membrane mask at the current threshold (after clean-up, if enabled)."""
        if self.cleanup and self.cleaned_at != self.threshold:  # clean-up still running
            return etsam_run.postprocess((self.logits > self.threshold).astype(np.uint8))
        return (self.model.data.array > self.threshold).astype(np.uint8)


def duration(seconds: float) -> str:
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes} min {seconds:02d} s" if minutes else f"{seconds} s"


class ETSAMManager:
    def __init__(self, session):
        self.session = session
        self.settings = ETSAMSettings(session, "ETSAM")
        self.running = False  # segmenting
        self.run_process = None  # etsam_run.WorkerRun in progress
        self.setup_state = "unknown"  # see the "setup" event above
        self.setup_message = ""
        self.results: dict[int, Result] = {}  # id(source volume) -> Result
        self._threshold_generation = 0
        self._listeners = []
        from chimerax.core.models import REMOVE_MODELS

        session.triggers.add_handler(REMOVE_MODELS, self._models_removed)

    # -- listeners ---------------------------------------------------------------

    def add_listener(self, listener):
        self._listeners.append(listener)

    def remove_listener(self, listener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, event, *args):
        for listener in list(self._listeners):
            try:
                listener(event, *args)
            except Exception:
                self.session.logger.report_exception(f"ETSAM: error in {event} listener")

    def _on_main(self, func, *args):
        if threading.current_thread() is threading.main_thread():
            func(*args)
        else:
            self.session.ui.thread_safe(func, *args)

    @property
    def busy(self) -> bool:
        return self.running or self.setup_state in ("checking", "installing")

    # -- ETSAM environment ----------------------------------------------------

    def ensure_etsam(self, install=False):
        """Find ETSAM; install only after the user clicks Install ETSAM."""
        if self.busy or self.setup_state == "ready":
            return
        self._set_setup("checking", "")

        def work():
            try:
                if etsam_env.find_python():
                    self._on_main(self._set_setup, "ready", "")
                    return
                if not etsam_env.can_install():
                    self._on_main(self._set_setup, "missing", etsam_env.NOT_FOUND)
                    return
                if not install:
                    self._on_main(self._set_setup, "missing", etsam_env.INSTALL_MESSAGE)
                    return
                report = lambda message: self._on_main(self._set_setup, "installing", message)
                report("Installing ETSAM (first start)…")
                etsam_env.install(log=report)
                self._on_main(self._set_setup, "ready", "✓ ETSAM is installed.")
            except Exception as e:
                self._on_main(self._set_setup, "failed", f"Installing ETSAM failed: {e}")

        threading.Thread(target=work, name="ETSAM setup", daemon=True).start()

    def _set_setup(self, state, message):
        self.setup_state, self.setup_message = state, message
        logger = self.session.logger
        first = message.splitlines()[0] if message else ""
        if state == "failed":
            logger.info(_red(html.escape(f"ETSAM: {message}")), is_html=True)
        elif first and state in ("installing", "ready"):
            logger.status(f"ETSAM: {first}", secondary=True)
            if state == "ready" or first.startswith("Installing ETSAM"):
                logger.info(f"ETSAM: {first}")
        self._notify("setup", state, message)
        self._notify("status")

    # -- segmentation ---------------------------------------------------------

    def options(self) -> dict:
        s = self.settings
        return dict(
            threshold=s.threshold,
            split_processing=s.split_processing,
            cleanup=s.cleanup,
            normalize_method=s.normalize_method,
            stage1_prompt=s.stage1_prompt,
            stage2_prompt=s.stage2_prompt,
        )

    def segment(self, volume, options=None, blocking=False):
        """Segment one open volume; the result is added as new volumes in ChimeraX."""
        if self.busy:
            raise RuntimeError("ETSAM is busy; wait for it to finish or cancel it.")
        options = options or self.options()
        array = np.asarray(volume.full_matrix())
        if array.ndim != 3 or min(array.shape) < 2:
            raise ValueError(
                f"{volume.name} has shape {array.shape}; ETSAM needs a 3D tomogram (Z, Y, X)."
            )
        self.running = True
        self._notify("status")
        self.session.logger.info(f"ETSAM: segmenting #{volume.id_string} {volume.name}")
        started = time.monotonic()

        def progress(fraction, message):
            self._on_main(self._notify, "progress", fraction, message)

        def work():
            try:
                self.run_process = etsam_run.segment_run(
                    array,
                    options["normalize_method"],
                    options["split_processing"],
                    options["stage1_prompt"],
                    options["stage2_prompt"],
                )
                logits = self.run_process.run(progress=progress)
                if self.run_process.device_name:
                    self._on_main(
                        self.session.logger.info, f"ETSAM: ran on {self.run_process.device_name}"
                    )
                mask = None
                if options["cleanup"]:
                    progress(1.0, "Cleaning up the mask…")
                    mask = (logits > options["threshold"]).astype(np.uint8)
                    self.run_process = etsam_run.WorkerRun("postprocess", mask)
                    mask = self.run_process.run()
                elapsed = time.monotonic() - started
                self._on_main(self._segmented, volume, logits, mask, options, elapsed)
            except BaseException as e:
                self._on_main(self._segment_failed, volume, e)

        if blocking:
            work()
        else:
            threading.Thread(target=work, name="ETSAM segment", daemon=True).start()

    def cancel(self):
        if self.running and self.run_process is not None:
            self.run_process.cancel()

    def _segment_failed(self, volume, exc):
        self.running = False
        self.run_process = None
        if isinstance(exc, etsam_run.Cancelled):
            self.session.logger.info("ETSAM: cancelled.")
        else:
            title, message = etsam_run.explain(exc)
            self.session.logger.info(
                _red(html.escape(f"ETSAM: {volume.name}: {title}. {message.splitlines()[0]}")),
                is_html=True,
            )
        self._notify("failed", exc)
        self._notify("status")

    def _segmented(self, volume, logits, mask, options, seconds):
        self.running = False
        self.run_process = None
        old = self.results.pop(id(volume), None)
        if old is not None and not old.model.deleted:  # segmenting again replaces it
            self.session.models.close([old.model])
        shown = logits if mask is None else np.where(mask > 0, logits, REMOVED)
        model = _add_volume(self.session, shown, volume, f"{volume.name} membranes")
        result = Result(
            volume, logits, model, options["cleanup"], options["split_processing"],
            options["threshold"], seconds,
            cleaned_at=options["threshold"] if mask is not None else None,
        )  # fmt: skip
        _style(self.session, model, result.threshold)
        model.add_volume_change_callback(
            lambda v, change: self._volume_changed(result, change)
        )
        self.results[id(volume)] = result
        percent = 100 * np.count_nonzero(shown > result.threshold) / shown.size
        self.session.logger.info(
            f"ETSAM: segmented {volume.name} in {duration(seconds)}; "
            f"{percent:.1f}% of voxels are membrane at threshold {result.threshold:.2f}."
        )
        self._notify("done", result)
        self._notify("status")

    @staticmethod
    def is_result(model) -> bool:
        """Whether ``model`` is one of ETSAM's result volumes (mask or logits)."""
        return getattr(model, "etsam_result", False)

    def result_for(self, volume) -> Result | None:
        result = self.results.get(id(volume)) if volume is not None else None
        if result is None or result.model.deleted:
            return None
        return result

    def _models_removed(self, trigger_name, models):
        removed = {id(m) for m in models}
        for key, r in list(self.results.items()):
            if id(r.source) in removed or id(r.model) in removed:
                del self.results[key]  # frees the logits
        self._notify("status")

    # -- threshold (surface level) --------------------------------------------

    def set_threshold(self, result: Result, threshold: float):
        """Set the result's threshold: the surface level of its volume."""
        if not result.model.deleted:
            # Calls _volume_changed, like a change in the Volume Viewer or a command.
            result.model.set_parameters(surface_levels=[threshold])

    def _volume_changed(self, result: Result, change):
        if change != "thresholds changed" or result.model.deleted or not result.model.surfaces:
            return
        level = float(result.model.surfaces[0].level)
        if level == result.threshold:
            return
        result.threshold = level
        self.settings.threshold = level
        self._notify("level", result)
        self._update_for_threshold(result)

    def _update_for_threshold(self, result: Result):
        """Re-run clean-up (if enabled) for the new threshold, and report statistics."""
        self._threshold_generation += 1
        generation = self._threshold_generation
        threshold = result.threshold

        def report(shown, cleanup_pending):
            if generation != self._threshold_generation or result.model.deleted:
                return  # a newer threshold is being applied, or the result was closed
            if shown is not None:
                result.model.data.array[...] = shown
                result.model.data.values_changed()
                result.cleaned_at = threshold
            source = result.logits if shown is None else shown
            percent = 100 * np.count_nonzero(source > threshold) / source.size
            text = f"Threshold {threshold:.2f} · {percent:.1f}% of voxels are membrane"
            if cleanup_pending:
                text += " · cleaning up…"
            elif result.cleanup:
                text += " (cleaned)"
            self._notify("threshold", result, text)

        def work():
            try:
                if not result.cleanup:
                    self._on_main(report, None, False)
                    return
                self._on_main(report, None, True)
                cleaned = etsam_run.postprocess((result.logits > threshold).astype(np.uint8))
                self._on_main(report, np.where(cleaned > 0, result.logits, REMOVED), False)
            except Exception as e:
                self._on_main(self._notify, "failed", e)

        threading.Thread(target=work, name="ETSAM threshold", daemon=True).start()

    # -- saving -----------------------------------------------------------------

    def default_path(self, result: Result, kind: str) -> str:
        """``kind``: "mask" or "logits". Same names as the etsam command line."""
        source = getattr(result.source.data, "path", None)
        if isinstance(source, str) and os.path.isfile(source):
            folder, name = os.path.dirname(source), os.path.basename(source).split(".")[0]
        else:
            folder, name = os.path.expanduser("~"), result.source.name.split(".")[0]
        tag = "_etsam_predicted_split_processing" if result.split else "_etsam_predicted"
        suffix = "logits" if kind == "logits" else (
            "post_processed_mask" if result.cleanup else "mask"
        )  # fmt: skip
        return os.path.join(folder, f"{name}{tag}_{suffix}.mrc")

    def save(self, result: Result, kind: str, path: str) -> str:
        """Save the mask at the current threshold, or the logits, as MRC."""
        from chimerax.map_data import ArrayGridData
        from chimerax.map_data.fileformats import save_grid_data

        if not path.lower().endswith((".mrc", ".map", ".rec")):
            path += ".mrc"
        # A separate grid object (sharing the array), so that saving does not rename
        # the volume shown in ChimeraX.
        d = result.source.data
        array = result.logits if kind == "logits" else result.mask()
        grid = ArrayGridData(
            array, origin=d.origin, step=d.step, cell_angles=d.cell_angles,
            rotation=d.rotation, name=os.path.basename(path),
        )  # fmt: skip
        save_grid_data([grid], path, self.session, format="mrc")
        self.session.logger.info(f"ETSAM: saved {path}")
        return path


# --------------------------------------------------------------------------- #
# Result volumes
# --------------------------------------------------------------------------- #


def _add_volume(session, array, source, name):
    """Open ``array`` as a new volume aligned with ``source``."""
    from chimerax.map import volume_from_grid_data
    from chimerax.map_data import ArrayGridData

    d = source.data
    grid = ArrayGridData(
        array, origin=d.origin, step=d.step, cell_angles=d.cell_angles,
        rotation=d.rotation, name=name,
    )  # fmt: skip
    model = volume_from_grid_data(
        grid, session, style="surface", open_model=False, show_dialog=False
    )
    model.etsam_result = True  # tag before adding, so the tool never lists it as a tomogram
    model.name = name
    session.models.add([model])
    model.position = source.position
    return model


def _style(session, model, threshold):
    from chimerax.core.commands import run

    # The whole volume as a surface at the threshold, at half resolution for very
    # large tomograms to keep it responsive.
    size = model.data.size
    step = 2 if size[0] * size[1] * size[2] > 2e8 else 1
    run(
        session,
        f"volume #{model.id_string} region all step {step} style surface "
        f"level {threshold} color {RESULT_COLOR}",
        log=False,
    )


def _red(text: str) -> str:
    return f'<span style="color:#d03030">{text}</span>'
