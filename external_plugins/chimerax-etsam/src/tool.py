"""
The ETSAM tool window (Tools ▸ Volume Data ▸ ETSAM Membrane Segmentation).

Laid out like the napari plugin's panel, with the same labels and options.
"""

from __future__ import annotations

import html
import os
import platform
import sys
import time
import traceback

import numpy as np

from chimerax.core.tools import ToolInstance
from Qt.QtCore import QRect, Qt, QTimer
from Qt.QtGui import QGuiApplication, QIcon
from Qt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QProxyStyle,
    QStyle,
    QStyleOptionDockWidget,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import etsam_env, etsam_run
from .manager import (
    DEFAULT_THRESHOLD,
    NORMALIZATIONS,
    STAGE1_PROMPTS,
    STAGE2_PROMPTS,
    duration,
    get_manager,
)

OK, WARN, ERR, MUTED = "#2e8b57", "#b8860b", "#d03030", "#808080"
SETUP_BLUE = "#2474c4"
MRC_FILTER = "MRC files (*.mrc *.rec *.map *.mrcs *.st *.ali)"


class _TitleIconStyle(QProxyStyle):
    """Add an icon to the native dock title while retaining its buttons and dragging."""

    def __init__(self, icon, dock):
        super().__init__()
        self.setParent(dock)
        self.icon = icon

    def drawControl(self, element, option, painter, widget=None):
        if element == QStyle.ControlElement.CE_DockWidgetTitle and not option.verticalTitleBar:
            background = QStyleOptionDockWidget(option)
            background.title = ""
            super().drawControl(element, background, painter, widget)
            title = QStyleOptionDockWidget(option)
            title.rect = option.rect.adjusted(24, 0, 0, 0)
            super().drawControl(element, title, painter, widget)
            self.icon.paint(painter, QRect(option.rect.left() + 3,
                                          option.rect.center().y() - 8, 18, 18))
        else:
            super().drawControl(element, option, painter, widget)


class ETSAMTool(ToolInstance):
    SESSION_ENDURING = True  # keep the tool (and a running job) across session closes
    SESSION_SAVE = False
    help = None  # set in __init__ to the bundled help page

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "ETSAM Membrane Segmentation"
        self.help = "file://" + os.path.join(os.path.dirname(__file__), "docs", "etsam.html")
        self.manager = get_manager(session)
        self.started = 0.0
        self.eta_anchor = None
        self.threshold_timer = QTimer(singleShot=True, interval=200)
        self.threshold_timer.timeout.connect(self._apply_threshold)

        from chimerax.ui import MainToolWindow

        self.tool_window = MainToolWindow(self)
        self._build_ui(self.tool_window.ui_area)

        from chimerax.core.models import ADD_MODELS, MODEL_NAME_CHANGED, REMOVE_MODELS

        triggers = session.triggers
        self._handlers = [
            triggers.add_handler(ADD_MODELS, self._models_added),
            triggers.add_handler(REMOVE_MODELS, lambda *_: self._refresh_volumes()),
            triggers.add_handler(MODEL_NAME_CHANGED, lambda *_: self._refresh_volumes()),
        ]
        self.manager.add_listener(self._on_manager_event)
        self.tool_window.manage(placement="side")
        self._bold_title()
        self._refresh_volumes()
        self._show_setup(self.manager.setup_state, self.manager.setup_message)
        self.manager.ensure_etsam()  # check only; installation needs a button click

    def delete(self):
        self.setup_timer.stop()
        for handler in self._handlers:
            self.session.triggers.remove_handler(handler)
        self.manager.remove_listener(self._on_manager_event)
        super().delete()

    # ------------------------------------------------------------------ UI --

    def _bold_title(self):
        """Show the panel's title in bold (when docked; a floating panel has a window title)."""
        try:
            dock = self.tool_window._dock_widget  # no public API for the title's font
        except Exception:
            return
        dock.setWindowIcon(self.window_icon)
        if not isinstance(dock.style(), _TitleIconStyle):
            dock.setStyle(_TitleIconStyle(self.window_icon, dock))
        bold = dock.font()
        bold.setBold(True)
        dock.setFont(bold)
        normal = dock.font()
        normal.setBold(False)  # explicitly, so that the panel does not inherit bold
        dock.widget().setFont(normal)

    def _build_ui(self, parent):
        s = self.manager.settings
        root = QVBoxLayout(parent)
        root.setContentsMargins(6, 6, 6, 6)
        self.window_icon = QIcon(os.path.join(os.path.dirname(__file__), "docs", "etsam-icon.svg"))
        parent.setWindowIcon(self.window_icon)

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
        self.install_btn.clicked.connect(lambda: self.manager.ensure_etsam(install=True))
        self.install_btn.hide()
        root.addWidget(self.install_btn)

        form = QFormLayout()
        row = QHBoxLayout()
        self.volume_combo = QComboBox()
        self.volume_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.volume_combo.currentIndexChanged.connect(lambda *_: self._volume_selected())
        self.no_volume_label = QLabel(
            "<i>No tomogram is open. Open one with Open… or File ▸ Open.</i>"
        )
        self.no_volume_label.setWordWrap(True)
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._open_file)
        row.addWidget(self.volume_combo, 1)
        row.addWidget(self.no_volume_label, 1)
        row.addWidget(open_btn)
        form.addRow("Tomogram:", row)

        th_box = QVBoxLayout()
        th_row = QHBoxLayout()
        self.th_slider = QSlider(Qt.Orientation.Horizontal)
        self.th_slider.setRange(-300, 300)  # threshold x 100
        self.th_spin = QDoubleSpinBox()
        self.th_spin.setRange(-20, 20)
        self.th_spin.setSingleStep(0.05)
        self.th_spin.setMinimumWidth(80)
        reset = QPushButton("Reset")
        reset.clicked.connect(lambda: self.th_spin.setValue(DEFAULT_THRESHOLD))
        self.th_slider.valueChanged.connect(lambda v: self.th_spin.setValue(v / 100))
        self.th_spin.valueChanged.connect(self._threshold_changed)
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

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        advanced_row = QHBoxLayout()
        advanced_row.addWidget(self.advanced_toggle)
        advanced_row.addStretch()
        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self._show_help)
        advanced_row.addWidget(help_btn)
        root.addLayout(advanced_row)
        self.advanced = QWidget()
        adv_form = QFormLayout(self.advanced)
        adv_form.setContentsMargins(16, 0, 0, 0)
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(NORMALIZATIONS)
        adv_form.addRow("Normalization:", self.norm_combo)
        self.s1_combo = QComboBox()
        self.s1_combo.addItems(STAGE1_PROMPTS)
        adv_form.addRow("Stage 1 prompt:", self.s1_combo)
        self.s2_combo = QComboBox()
        self.s2_combo.addItems(STAGE2_PROMPTS)
        adv_form.addRow("Stage 2 prompt:", self.s2_combo)
        self.advanced.setVisible(False)
        root.addWidget(self.advanced)

        def toggle(checked):
            self.advanced.setVisible(checked)
            self.advanced_toggle.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
            )

        self.advanced_toggle.toggled.connect(toggle)

        # Options are remembered between sessions.
        self.th_spin.setValue(s.threshold)
        for checkbox, key in (
            (self.split_cb, "split_processing"),
            (self.cleanup_cb, "cleanup"),
        ):
            checkbox.setChecked(bool(getattr(s, key)))
            checkbox.toggled.connect(lambda value, key=key: setattr(s, key, bool(value)))
        for combo, key in (
            (self.norm_combo, "normalize_method"),
            (self.s1_combo, "stage1_prompt"),
            (self.s2_combo, "stage2_prompt"),
        ):
            combo.setCurrentIndex(max(0, combo.findText(getattr(s, key))))
            combo.currentTextChanged.connect(lambda value, key=key: setattr(s, key, value))

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
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        save_row = QHBoxLayout()
        self.save_mask_btn = QPushButton("Save mask…")
        self.save_mask_btn.setToolTip("Save the mask at the current threshold (surface level).")
        self.save_mask_btn.clicked.connect(lambda: self._save("mask"))
        self.save_logits_btn = QPushButton("Save logits…")
        self.save_logits_btn.clicked.connect(lambda: self._save("logits"))
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
        self.error_message.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
                f"ETSAM environment: {etsam_env.find_python() or 'not found'}",
                "",
            ]
            if getattr(exc, "details", ""):
                lines += ["ETSAM log:", exc.details]
            else:
                lines.append(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                )
            report = "\n".join(lines)
        self.error_title.setText(title)
        self.error_message.setText(message)
        self.error_details.setPlainText(report)
        self.error_details.hide()
        self.error_panel.show()

    # ------------------------------------------------------------- volumes --

    def _is_tomogram(self, model) -> bool:
        """Volumes that can be segmented: not ETSAM's own results."""
        from chimerax.map import Volume

        return isinstance(model, Volume) and model.data is not None and not self.manager.is_result(model)

    def _volumes(self):
        from chimerax.map import Volume

        return [m for m in self.session.models.list(type=Volume) if self._is_tomogram(m)]

    def _current_volume(self):
        wanted = self.volume_combo.currentData()
        return next((v for v in self._volumes() if id(v) == wanted), None)

    def _current_result(self):
        return self.manager.result_for(self._current_volume())

    def _volume_selected(self):
        result = self._current_result()
        if result is not None:
            self._show_threshold(result.threshold)
        self._update_buttons()

    def _models_added(self, trigger_name, models):
        added = [m for m in models if self._is_tomogram(m)]
        self._refresh_volumes(select=added[-1] if added else None)

    def _refresh_volumes(self, select=None):
        volumes = self._volumes()
        select = select or self._current_volume()
        self.volume_combo.blockSignals(True)
        self.volume_combo.clear()
        for v in volumes:
            self.volume_combo.addItem(f"#{v.id_string} {v.name}", id(v))
        if select in volumes:
            self.volume_combo.setCurrentIndex(volumes.index(select))
        self.volume_combo.blockSignals(False)
        self.volume_combo.setVisible(bool(volumes))
        self.no_volume_label.setVisible(not volumes)
        self._update_buttons()

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self.tool_window.ui_area, "Open tomogram", os.path.expanduser("~"), MRC_FILTER
        )
        if path:
            from chimerax.core.commands import run
            from chimerax.core.commands.cli import quote_path_if_necessary

            run(self.session, f"open {quote_path_if_necessary(path)}")

    # --------------------------------------------------------------- setup --

    def _show_setup(self, state, message):
        self.install_btn.setVisible(state in ("missing", "failed") and etsam_env.can_install())
        self.install_btn.setEnabled(not self.manager.busy)
        first = html.escape(message.splitlines()[0]) if message else ""
        if state == "missing":
            self._set_setup(f"<span style='color:{WARN}'>{first}</span>")
        elif state == "failed":
            self._set_setup(
                f"<span style='color:{ERR}'>{first}</span> "
                f"<a href='{etsam_env.INSTALL_DOCS}'>Installation guide</a>"
            )
            if state == "failed":
                self._show_error("Installing ETSAM failed", message)
        elif state == "checking":
            self._set_setup("Looking for ETSAM…", busy=True)
        elif state == "installing":
            self._set_setup(first, busy=True)
        else:  # ready: blue confirmation without an active spinner
            self._set_setup(f"<span style='color:{SETUP_BLUE}'>{first}</span>" if first else "")
        self.setup_label.setToolTip(message)

    # ----------------------------------------------------------------- run --

    def _run(self):
        self.error_panel.hide()
        volume = self._current_volume()
        if volume is None:
            self._show_error(
                "No tomogram selected",
                "Open a tomogram first: click 'Open…', or use File ▸ Open.",
            )
            return
        self.started = time.perf_counter()
        self.eta_anchor = None
        self.progress.setValue(0)
        self.progress.show()
        self._set_status("Loading ETSAM…")
        # Run the "etsam segment" command with the panel's settings, so that the run is
        # in the Log and the command history, and can be repeated or put in a script.
        from chimerax.core.commands import run
        from chimerax.core.errors import UserError

        from .cmd import NotSegmentable, segment_command

        try:
            run(self.session, segment_command(volume, self.manager.options()))
        except Exception as e:
            self.progress.hide()
            self._set_status("✗ Segmentation failed, see below.", ERR)
            if isinstance(e, NotSegmentable):
                self._show_error("This volume cannot be segmented", str(e))
            elif isinstance(e, UserError):
                self._show_error("ETSAM could not start", str(e))
            else:
                self._show_error("ETSAM could not start", str(e), e)

    def _cancel(self):
        self.cancel_btn.setEnabled(False)
        self.manager.cancel()
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
                eta = f" · about {duration((1 - fraction) * (now - t0) / (fraction - f0))} left"
        self._set_status(message + eta)

    def _on_done(self, result):
        self.progress.setValue(1000)
        self.progress.hide()
        self._show_threshold(result.threshold)
        percent = 100 * np.count_nonzero(result.mask()) / result.logits.size
        self._set_status(
            f"✓ Segmented '{result.source.name}' in {duration(result.seconds)} · "
            f"{percent:.1f}% of voxels are membrane. Adjust the threshold to refine.",
            OK,
        )

    def _on_failed(self, exc):
        self.progress.hide()
        if isinstance(exc, etsam_run.Cancelled):
            self._set_status("Cancelled.", WARN)
            return
        self._set_status("✗ Segmentation failed, see below.", ERR)
        title, message = etsam_run.explain(exc)
        self._show_error(title, message, exc)

    # ----------------------------------------------------------- threshold --

    # The threshold is the surface level of the result volume: changing it here moves
    # the level, and changing the level (Volume Viewer, "volume" command) moves this.

    def _threshold_changed(self, value):
        self.manager.settings.threshold = float(value)
        self.th_slider.blockSignals(True)
        self.th_slider.setValue(int(round(max(-3, min(3, value)) * 100)))
        self.th_slider.blockSignals(False)
        if self._current_result() is not None:
            self.threshold_timer.start()

    def _apply_threshold(self):
        result = self._current_result()
        if result is not None:
            self.manager.set_threshold(result, self.th_spin.value())

    def _show_threshold(self, value):
        """Show a threshold set elsewhere (the volume's level), without applying it again."""
        self.th_spin.blockSignals(True)
        self.th_spin.setValue(value)
        self.th_spin.blockSignals(False)
        self.th_slider.blockSignals(True)
        self.th_slider.setValue(int(round(max(-3, min(3, value)) * 100)))
        self.th_slider.blockSignals(False)

    # ---------------------------------------------------------------- save --

    def _save(self, kind):
        result = self._current_result()
        if result is None:
            return
        title = "Save membrane logits" if kind == "logits" else "Save membrane mask"
        path, _ = QFileDialog.getSaveFileName(
            self.tool_window.ui_area, title, self.manager.default_path(result, kind),
            "MRC (*.mrc)",
        )  # fmt: skip
        if not path:
            return
        from chimerax.core.commands import run
        from chimerax.core.commands.cli import quote_path_if_necessary

        command = f"etsam save #{result.model.id_string} {quote_path_if_necessary(path)}"
        if kind == "logits":
            command += " type logits"
        try:
            path = run(self.session, command) or path
            self._set_status(f"✓ Saved {path}", OK)
        except Exception as e:
            self._show_error(f"Could not save {path}", str(e), e)

    def _show_help(self):
        from chimerax.help_viewer import show_url

        show_url(self.session, self.help)

    # ------------------------------------------------------------- updates --

    def _on_manager_event(self, event, *args):
        if event == "setup":
            self._show_setup(*args)
        elif event == "level":
            if args[0] is self._current_result():
                self._show_threshold(args[0].threshold)
        elif event == "progress":
            self._on_progress(*args)
        elif event == "done":
            self._on_done(args[0])
        elif event == "failed":
            self._on_failed(args[0])
        elif event == "threshold":
            result, text = args
            if result is self._current_result():
                self._set_status(text, "" if text.endswith("…") else OK)
        self._update_buttons()

    def _update_buttons(self):
        busy = self.manager.busy
        has_volume = self._current_volume() is not None
        has_result = self._current_result() is not None
        self.run_btn.setEnabled(not busy and has_volume and self.manager.setup_state == "ready")
        self.volume_combo.setEnabled(not busy)
        self.cancel_btn.setEnabled(self.manager.running)
        self.save_mask_btn.setVisible(has_result)
        self.save_logits_btn.setVisible(has_result)
        self.th_hint.setText(
            "◀ more membrane · fewer false positives ▶"
            + ("  (updates the result instantly)" if has_result else "")
        )
