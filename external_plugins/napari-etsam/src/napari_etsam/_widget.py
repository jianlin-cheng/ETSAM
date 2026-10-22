"""The ETSAM membrane segmentation dock widget."""

import html
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from napari.layers import Image
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QGuiApplication, QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from napari_etsam import etsam_env, etsam_run
from napari_etsam._writer import save_mrc

DOCS_URL = "https://jianlin-cheng.github.io/ETSAM/napari/"
DEFAULT_THRESHOLD = -0.25
OK, WARN, ERR, MUTED = "#3fb950", "#d29922", "#f85149", "#8b949e"
SETUP_BLUE = "#58a6ff"


Cancelled = etsam_run.Cancelled


class _Signals(QObject):
    progress = Signal(float, str)
    setup = Signal(str)  # ETSAM installation progress (first start)


def _duration(seconds):
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes} min {seconds:02d} s" if minutes else f"{seconds} s"


class ETSAMWidget(QWidget):
    """Segment cell membranes in cryo-ET tomograms with ETSAM."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.signals = _Signals()
        self.signals.progress.connect(self._on_progress)
        self.worker = None
        self.etsam_ready = False
        self.run_process = None  # etsam_run.WorkerRun in progress
        # id(image layer) -> dict(image, labels, logits, cleanup, split, threshold), where
        # labels shows the logits thresholded at "threshold" (and cleaned up, if enabled)
        self.results = {}
        # id(image layer) -> threshold the user chose for it; others use DEFAULT_THRESHOLD.
        self.thresholds = {}
        self.started = 0.0
        self.eta_anchor = None
        self.threshold_timer = QTimer(self, singleShot=True, interval=200)
        self.threshold_timer.timeout.connect(self._apply_threshold)
        self.threshold_generation = 0

        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self._refresh_layers()
        self.signals.setup.connect(self._on_setup_progress)
        self._ensure_etsam()

    def showEvent(self, event):
        super().showEvent(event)
        self._bold_title()

    def _bold_title(self):
        """Show the dock widget's title in bold (napari draws it with a QLabel)."""
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, QDockWidget):
            parent = parent.parentWidget()
        if parent is None:
            return
        parent.setWindowIcon(self.window_icon)
        # napari replaces the title bar when the dock is floated or moved.
        if not parent.property("etsamTitleIconConnected"):
            parent.setProperty("etsamTitleIconConnected", True)
            refresh = lambda *_: QTimer.singleShot(0, self._bold_title)
            parent.topLevelChanged.connect(refresh)
            parent.dockLocationChanged.connect(refresh)
        bar = parent.titleBarWidget()
        title = getattr(bar, "title", None)
        if isinstance(title, QLabel):
            layout = bar.layout()
            if layout is not None and bar.findChild(QLabel, "etsamTitleIcon") is None:
                icon_label = QLabel(bar)
                icon_label.setObjectName("etsamTitleIcon")
                icon_label.setPixmap(self.window_icon.pixmap(18, 18))
                icon_label.setFixedSize(18, 18)
                icon_label.setAccessibleName("ETSAM membrane icon")
                icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                index = layout.indexOf(title)
                layout.insertWidget(index if index >= 0 else layout.count(), icon_label)
        if isinstance(title, QLabel):
            font = title.font()
            font.setBold(True)
            title.setFont(font)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        root = QVBoxLayout(self)
        self.window_icon = QIcon(str(Path(__file__).with_name("etsam-icon.svg")))
        self.setWindowIcon(self.window_icon)

        # Check for ETSAM and offer installation before downloading anything.
        self.setup_row = QWidget()
        setup_layout = QHBoxLayout(self.setup_row)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(6)
        self.setup_spinner = QLabel()
        self.setup_spinner.setFixedWidth(20)
        self.setup_spinner.setStyleSheet(f"color: {SETUP_BLUE}; font-size: 18px;")
        self.setup_spinner.hide()
        self.setup_frame = 0
        self.setup_timer = QTimer(self.setup_row, interval=90)
        self.setup_timer.timeout.connect(self._advance_setup_spinner)
        setup_layout.addWidget(self.setup_spinner, 0, Qt.AlignmentFlag.AlignTop)
        self.setup_label = QLabel()
        self.setup_label.setWordWrap(True)
        self.setup_label.setStyleSheet(f"color: {MUTED};")
        self.setup_label.setOpenExternalLinks(True)
        setup_layout.addWidget(self.setup_label, 1)
        root.addWidget(self.setup_row)
        self.install_btn = QPushButton("Install ETSAM")
        self.install_btn.clicked.connect(lambda: self._ensure_etsam(install=True))
        self.install_btn.hide()
        root.addWidget(self.install_btn)

        form = QFormLayout()
        row = QHBoxLayout()
        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self._layer_selected)
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._open_file)
        row.addWidget(self.layer_combo, 1)
        row.addWidget(open_btn)
        form.addRow("Tomogram:", row)

        th_box = QVBoxLayout()
        th_row = QHBoxLayout()
        self.th_slider = QSlider(Qt.Horizontal)
        self.th_slider.setRange(-300, 300)  # threshold x 100
        self.th_spin = QDoubleSpinBox()
        self.th_spin.setRange(-20, 20)
        self.th_spin.setSingleStep(0.05)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.th_spin.setValue(DEFAULT_THRESHOLD))
        self.th_slider.valueChanged.connect(lambda v: self.th_spin.setValue(v / 100))
        self.th_spin.valueChanged.connect(self._threshold_changed)
        self._show_threshold(DEFAULT_THRESHOLD)
        th_row.addWidget(self.th_slider)
        th_row.addWidget(self.th_spin)
        th_row.addWidget(reset)
        th_box.addLayout(th_row)
        self.th_hint = QLabel()
        self.th_hint.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        th_box.addWidget(self.th_hint)
        form.addRow("Threshold:", th_box)

        self.split_cb = QCheckBox("Closely spaced membranes (split processing, ~4× slower)")
        form.addRow("", self.split_cb)
        self.cleanup_cb = QCheckBox("Clean up mask (remove noise, separate touching membranes)")
        form.addRow("", self.cleanup_cb)
        root.addLayout(form)

        # "Advanced settings" toggles the options below it; Help sits on the same row.
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        advanced_row = QHBoxLayout()
        advanced_row.addWidget(self.advanced_toggle)
        advanced_row.addStretch()
        help_btn = QPushButton("Help")
        help_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        advanced_row.addWidget(help_btn)
        root.addLayout(advanced_row)
        self.advanced = QWidget()
        adv_form = QFormLayout(self.advanced)
        adv_form.setContentsMargins(16, 0, 0, 0)
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["min_max_positive_values", "softplus_minmax"])
        adv_form.addRow("Normalization:", self.norm_combo)
        self.s1_combo = QComboBox()
        self.s1_combo.addItems(["grid_zero", "grid", "zero"])
        adv_form.addRow("Stage 1 prompt:", self.s1_combo)
        self.s2_combo = QComboBox()
        self.s2_combo.addItems(["zero", "grid", "grid_zero", "etsam_stage1_partial"])
        adv_form.addRow("Stage 2 prompt:", self.s2_combo)
        self.advanced.setVisible(False)
        root.addWidget(self.advanced)

        def toggle(checked):
            self.advanced.setVisible(checked)
            self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

        self.advanced_toggle.toggled.connect(toggle)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Segment membranes")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        run_row.addWidget(self.run_btn, 3)
        run_row.addWidget(self.cancel_btn, 1)
        root.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        root.addWidget(self.progress)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        save_row = QHBoxLayout()
        self.save_mask_btn = QPushButton("Save mask…")
        self.save_mask_btn.setToolTip("Save the mask at the current threshold.")
        self.save_mask_btn.clicked.connect(self._save_mask)
        self.save_logits_btn = QPushButton("Save logits…")
        self.save_logits_btn.clicked.connect(self._save_logits)
        save_row.addWidget(self.save_mask_btn)
        save_row.addWidget(self.save_logits_btn)
        root.addLayout(save_row)

        # Error panel: what went wrong, how to fix it, and the full report.
        self.error_panel = QFrame()
        self.error_panel.setObjectName("etsamError")
        self.error_panel.setStyleSheet(
            f"#etsamError {{ border: 1px solid {ERR}; border-radius: 4px; }}"
        )
        err = QVBoxLayout(self.error_panel)
        self.error_title = QLabel()
        self.error_title.setWordWrap(True)
        self.error_title.setStyleSheet(f"color: {ERR}; font-weight: bold;")
        self.error_message = QLabel()
        self.error_message.setWordWrap(True)
        self.error_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.error_details = QPlainTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.hide()
        buttons = QHBoxLayout()
        details_btn = QPushButton("Show details")
        details_btn.clicked.connect(
            lambda: self.error_details.setVisible(self.error_details.isHidden())
        )
        copy_btn = QPushButton("Copy error report")
        copy_btn.clicked.connect(
            lambda: QGuiApplication.clipboard().setText(self.error_details.toPlainText())
        )
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.clicked.connect(self.error_panel.hide)
        for b in (details_btn, copy_btn, dismiss_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        err.addWidget(self.error_title)
        err.addWidget(self.error_message)
        err.addLayout(buttons)
        err.addWidget(self.error_details)
        self.error_panel.hide()
        root.addWidget(self.error_panel)
        root.addStretch()
        self._update_result_buttons()

    def _advance_setup_spinner(self):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.setup_spinner.setText(frames[self.setup_frame % len(frames)])
        self.setup_frame += 1

    def _set_setup(self, text, busy=False):
        self.setup_label.setText(text)
        self.setup_label.setStyleSheet(f"color: {SETUP_BLUE if busy else MUTED};")
        self.setup_row.setVisible(bool(text))
        self.setup_spinner.setVisible(busy and bool(text))
        if busy and text:
            if not self.setup_timer.isActive():
                self.setup_frame = 0
                self._advance_setup_spinner()
                self.setup_timer.start()
        else:
            self.setup_timer.stop()

    def _set_status(self, text, color=""):
        self.status_label.setStyleSheet(f"color: {color};" if color else "")
        # Word wrap breaks lines only at spaces; let long file paths break after
        # their separators too, instead of being cut off.
        self.status_label.setText(text.replace("\\", "\\​").replace("/", "/​"))
        self.status_label.setToolTip(text)

    def _show_error(self, title, message, exc=None):
        report = message
        if exc is not None:
            lines = [
                f"Error: {type(exc).__name__}: {exc}",
                f"Python {sys.version.split()[0]}, {platform.platform()}",
            ]
            lines.append(f"ETSAM environment: {etsam_env.find_python() or 'not found'}")
            if isinstance(exc, etsam_run.ETSAMError):
                lines += ["", "ETSAM log:", exc.details]
            else:
                lines += [
                    "",
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                ]
            report = "\n".join(lines)
        self.error_title.setText(title)
        self.error_message.setText(message)
        self.error_details.setPlainText(report)
        self.error_details.hide()
        self.error_panel.show()

    # -------------------------------------------------------------- layers --
    def _images(self):
        return [layer for layer in self.viewer.layers if isinstance(layer, Image)]

    def _current_image(self):
        for layer in self._images():
            if id(layer) == self.layer_combo.currentData():
                return layer
        return None

    def _current_result(self):
        result = self.results.get(id(self._current_image()))
        if result and result["labels"] in self.viewer.layers:
            return result
        return None

    def _refresh_layers(self, select=None):
        previous = self._current_image()
        select = select or previous
        images = self._images()
        self.thresholds = {key: value for key, value in self.thresholds.items()
                           if key in {id(layer) for layer in images}}
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for layer in images:
            self.layer_combo.addItem(layer.name, id(layer))
        if not images:
            self.layer_combo.addItem("— open a tomogram (.mrc) first —", None)
        elif select in images:
            self.layer_combo.setCurrentIndex(images.index(select))
        self.layer_combo.blockSignals(False)
        if self._current_image() is not previous:
            self._layer_selected()
        self._update_result_buttons()

    def _layer_selected(self):
        # A tomogram shows the default threshold unless the user chose another for it.
        self._show_threshold(self.thresholds.get(id(self._current_image()), DEFAULT_THRESHOLD))
        self._update_result_buttons()

    def _on_layer_inserted(self, event):
        layer = event.value
        self._refresh_layers(select=layer if isinstance(layer, Image) else None)

    def _on_layer_removed(self, event):
        for key, result in list(self.results.items()):
            if event.value in (result["image"], result["labels"]):
                del self.results[key]  # frees the logits
        self._refresh_layers()

    def _update_result_buttons(self):
        has_result = self._current_result() is not None
        self.save_mask_btn.setVisible(has_result)
        self.save_logits_btn.setVisible(has_result)
        self.th_hint.setText(
            "◀ more membrane · fewer false positives ▶"
            + ("  (updates the result instantly)" if has_result else "")
        )

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open tomogram",
            str(Path.home()),
            "MRC files (*.mrc *.rec *.map *.mrcs *.st *.ali)",
        )
        if path:
            self.viewer.open(path, plugin="napari-etsam")

    # --------------------------------------------------------------- setup --
    def _ensure_etsam(self, install=False):
        """Find ETSAM; install only after the user clicks Install ETSAM."""
        if self.worker is not None:
            return
        self.install_btn.hide()
        signals = self.signals

        def job():
            if etsam_env.find_python():
                return True
            if not install or not etsam_env.can_install():
                return False
            signals.setup.emit("Installing ETSAM (first start)…")
            etsam_env.install(log=signals.setup.emit)
            return True

        self._set_setup("Looking for ETSAM…", busy=True)
        self.run_btn.setEnabled(False)
        self.worker = create_worker(job, _ignore_errors=True)
        self.worker.returned.connect(self._on_setup_result)
        self.worker.errored.connect(self._on_setup_error)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_setup_progress(self, text):
        self._set_setup(html.escape(text), busy=True)

    def _on_setup_result(self, ready):
        self.etsam_ready = ready
        self.install_btn.setVisible(not ready and etsam_env.can_install())
        message = etsam_env.INSTALL_MESSAGE if etsam_env.can_install() else etsam_env.NOT_FOUND
        self._set_setup(
            f"<span style='color:{SETUP_BLUE}'>✓ ETSAM is installed.</span>"
            if ready else f"<span style='color:{WARN}'>{message}</span>"
        )

    def _on_setup_error(self, exc):
        self.etsam_ready = False
        self.install_btn.setVisible(etsam_env.can_install())
        first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        self._set_setup(
            f"<span style='color:{ERR}'>{first}</span> "
            f"<a href='{etsam_env.INSTALL_DOCS}'>Installation guide</a>"
        )
        if getattr(exc, "error_type", "") != "NotFound":
            self._show_error("Installing ETSAM failed", str(exc), exc)

    # ----------------------------------------------------------------- run --
    def _run(self):
        self.error_panel.hide()
        image = self._current_image()
        if image is None:
            self._show_error(
                "No tomogram selected",
                "Open a tomogram first: click 'Open…', use File ▸ Open, or drag and drop an .mrc file onto napari.",
            )
            return
        volume = image.data[0] if image.multiscale else image.data
        if image.rgb or volume.ndim != 3:
            self._show_error(
                "This layer cannot be segmented",
                f"'{image.name}' has shape {tuple(volume.shape)}; ETSAM needs a single 3D grayscale tomogram (Z, Y, X).",
            )
            return

        settings = {
            "normalize_method": self.norm_combo.currentText(),
            "split_processing": self.split_cb.isChecked(),
            "stage1_prompt": self.s1_combo.currentText(),
            "stage2_prompt": self.s2_combo.currentText(),
        }
        threshold = self.th_spin.value()
        cleanup = self.cleanup_cb.isChecked()
        signals = self.signals
        self.run_process = etsam_run.segment_run(volume, **settings)

        def job():
            logits = self.run_process.run(progress=signals.progress.emit)
            mask = (logits > threshold).astype(np.uint8)
            if cleanup:
                signals.progress.emit(1.0, "Cleaning up the mask…")
                self.run_process = etsam_run.WorkerRun("postprocess", mask)
                mask = self.run_process.run()
            return logits, mask

        self.started = time.perf_counter()
        self.eta_anchor = None
        self.progress.setValue(0)
        self.progress.show()
        self._set_status("Loading ETSAM…")
        self.worker = create_worker(job, _ignore_errors=True)
        self.worker.returned.connect(
            lambda result: self._on_done(
                image, *result, threshold, cleanup, settings["split_processing"]
            )
        )
        self.worker.errored.connect(self._on_error)
        self.worker.finished.connect(self._on_finished)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.worker.start()

    def _cancel(self):
        self.cancel_btn.setEnabled(False)
        if self.run_process is not None:
            self.run_process.cancel()
        self._set_status("Cancelling…")

    def _on_progress(self, fraction, message):
        self.progress.setValue(int(fraction * 1000))
        eta = ""
        if "slice" in message:
            now = time.perf_counter()
            if self.eta_anchor is None:
                self.eta_anchor = (now, fraction)
            t0, f0 = self.eta_anchor
            if fraction - f0 > 0.02:
                eta = f" · about {_duration((1 - fraction) * (now - t0) / (fraction - f0))} left"
        self._set_status(message + eta)

    def _on_done(self, image, logits, mask, threshold, cleanup, split):
        result = self.results.get(id(image))
        if result and result["labels"] in self.viewer.layers:
            labels = result["labels"]
            labels.data = mask
        else:
            labels = self.viewer.add_labels(
                mask,
                name=f"{image.name} membranes",
                scale=image.scale,
                translate=image.translate,
                opacity=0.6,
                metadata=dict(image.metadata),
            )
            labels.colormap = DirectLabelColormap(
                color_dict={None: (0, 0, 0, 0), 1: (1.0, 0.85, 0.1, 1.0)}
            )
        self.results[id(image)] = {
            "image": image,
            "labels": labels,
            "logits": logits,
            "cleanup": cleanup,
            "split": split,
            "threshold": threshold,
        }
        self.viewer.layers.selection.active = labels
        self.progress.setValue(1000)
        self.progress.hide()
        percent = 100 * np.count_nonzero(mask) / mask.size
        elapsed = _duration(time.perf_counter() - self.started)
        self._set_status(
            f"✓ Segmented '{image.name}' in {elapsed} · {percent:.1f}% of voxels are membrane. "
            "Adjust the threshold to refine.",
            OK,
        )
        show_info(f"ETSAM finished '{image.name}' in {elapsed}")

    def _on_error(self, exc):
        self.progress.hide()
        if isinstance(exc, Cancelled):
            self._set_status("Cancelled.", WARN)
            return
        self._set_status("✗ Segmentation failed, see below.", ERR)
        title, message = etsam_run.explain(exc)
        self._show_error(title, message, exc)

    def _on_finished(self):
        self.worker = None
        self.run_process = None
        self.run_btn.setEnabled(self.etsam_ready)
        self.cancel_btn.setEnabled(False)
        self._update_result_buttons()

    # ----------------------------------------------------------- threshold --
    def _threshold_changed(self, value):
        image = self._current_image()
        if image is not None:
            self.thresholds[id(image)] = value
        self.th_slider.blockSignals(True)
        self.th_slider.setValue(int(round(max(-3, min(3, value)) * 100)))
        self.th_slider.blockSignals(False)
        if self._current_result() is not None and self.worker is None:
            self.threshold_timer.start()

    def _show_threshold(self, value):
        """Show the threshold of the selected tomogram, without applying it to its result."""
        self.th_spin.blockSignals(True)
        self.th_spin.setValue(value)
        self.th_spin.blockSignals(False)
        self.th_slider.blockSignals(True)
        self.th_slider.setValue(int(round(max(-3, min(3, value)) * 100)))
        self.th_slider.blockSignals(False)

    def _apply_threshold(self):
        """Re-binarize the stored logits (and re-run clean-up) without re-running ETSAM."""
        result = self._current_result()
        if result is None:
            return
        self.threshold_generation += 1
        generation = self.threshold_generation
        threshold = self.th_spin.value()

        def job():
            mask = (result["logits"] > threshold).astype(np.uint8)
            yield mask, result["cleanup"]
            if result["cleanup"]:
                yield etsam_run.postprocess(mask), False

        def show(item):
            mask, cleanup_pending = item
            if generation != self.threshold_generation:
                return  # a newer threshold is being applied
            result["labels"].data = mask
            result["threshold"] = None if cleanup_pending else threshold
            percent = 100 * np.count_nonzero(mask) / mask.size
            text = f"Threshold {threshold:.2f} · {percent:.1f}% of voxels are membrane"
            if cleanup_pending:
                self._set_status(text + " · cleaning up…")
            else:
                self._set_status(text + (" (cleaned)" if result["cleanup"] else ""), OK)

        worker = create_worker(job, _ignore_errors=True)
        worker.yielded.connect(show)
        worker.errored.connect(
            lambda exc: self._show_error("Re-thresholding failed", str(exc), exc)
        )
        worker.start()

    # ---------------------------------------------------------------- save --
    def _save(self, title, suffix, data):
        result = self._current_result()
        source = result["image"].metadata.get("source_path")
        folder = Path(source).parent if source else Path.home()
        name = Path(source).name.split(".")[0] if source else result["image"].name
        tag = "_etsam_predicted_split_processing" if result["split"] else "_etsam_predicted"
        default = str(folder / f"{name}{tag}_{suffix}.mrc")
        path, _ = QFileDialog.getSaveFileName(self, title, default, "MRC (*.mrc)")
        if not path:
            return
        try:
            path = save_mrc(path, data, result["image"].metadata, result["image"].scale)
            self._set_status(f"✓ Saved {path}", OK)
        except Exception as e:  # noqa: BLE001 - any failure is shown in the error panel
            self._show_error(f"Could not save {path}", str(e), e)

    def _save_mask(self):
        """Save the mask at the current threshold, even if the layer is not updated yet."""
        result = self._current_result()
        threshold = self.th_spin.value()
        if result["threshold"] == threshold:
            mask = np.asarray(result["labels"].data).astype(np.uint8)
        else:
            mask = (result["logits"] > threshold).astype(np.uint8)
            if result["cleanup"]:
                self._set_status("Cleaning up the mask…")
                mask = etsam_run.postprocess(mask)
        suffix = "post_processed_mask" if result["cleanup"] else "mask"
        self._save("Save membrane mask", suffix, mask)

    def _save_logits(self):
        self._save("Save membrane logits", "logits", self._current_result()["logits"])
