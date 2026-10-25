"""Drive the napari plugin's panel end to end, as a user would (CI).

    python tests/e2e/napari_plugin.py TOMOGRAM.mrc OUTPUT_DIR

(On Linux without a display, set QT_QPA_PLATFORM=offscreen; on macOS, napari
needs the default platform, which has OpenGL.)

Run with a Python that has napari, a Qt binding and napari-etsam, not with the
ETSAM environment: the plugin finds that environment itself (as installed by
the ETSAM installer) and runs ETSAM there. Opens the tomogram with the plugin's
reader, waits for the panel to find ETSAM, clicks "Segment membranes" with
mask clean-up, changes the threshold, and saves the mask and the logits with
the panel's buttons into OUTPUT_DIR (napari_mask.mrc, napari_logits.mrc).
"""

import os
import sys
import time
from unittest.mock import patch

import napari
import numpy as np
from qtpy.QtWidgets import QApplication

from napari_etsam import etsam_env
from napari_etsam._widget import ETSAMWidget


def wait_for(condition, timeout, what, widget):
    end = time.monotonic() + timeout
    while not condition():
        QApplication.processEvents()
        time.sleep(0.05)
        if time.monotonic() > end:
            fail(f"timed out after {timeout} s waiting for {what}", widget)


def fail(message, widget):
    print(f"FAIL: {message}")
    print("  setup:  ", widget.setup_label.text())
    print("  status: ", widget.status_label.text())
    if widget.error_panel.isVisibleTo(widget):
        print("  error:  ", widget.error_title.text(), "-", widget.error_message.text())
        print(widget.error_details.toPlainText())
    sys.exit(1)


def idle(widget):
    return widget.worker is None


def main():
    tomogram, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    print("installer for this computer:", etsam_env.installer_platform())
    print("ETSAM environment:", etsam_env.find_python())

    viewer = napari.Viewer(show=False)
    image = viewer.open(tomogram, plugin="napari-etsam")[0]
    print(f"opened {image.name}: shape {image.data.shape}, metadata {image.metadata.get('voxel_size')}")
    widget = ETSAMWidget(viewer)
    viewer.window.add_dock_widget(widget)

    wait_for(lambda: idle(widget), 600, "the panel to find ETSAM", widget)
    if not widget.etsam_ready:
        fail("ETSAM is not ready", widget)
    print("setup:", widget.setup_label.text())

    progress = []
    widget.signals.progress.connect(lambda fraction, message: progress.append(message))
    widget.cleanup_cb.setChecked(True)
    widget.run_btn.click()
    started = time.monotonic()
    wait_for(lambda: idle(widget), 3600, "the segmentation", widget)
    result = widget.results.get(id(image))
    if result is None:
        fail("no result after segmenting", widget)
    print(f"segmented in {time.monotonic() - started:.0f} s:", widget.status_label.toolTip())
    device = next((m for m in progress if m.startswith("Running on")), "(no device reported)")
    print("device:", device)
    if os.environ.get("ETSAM_EXPECT_DEVICE") and os.environ["ETSAM_EXPECT_DEVICE"] not in device:
        fail(f"expected to run on {os.environ['ETSAM_EXPECT_DEVICE']}: {device}", widget)

    labels, logits = result["labels"], result["logits"]
    if labels not in viewer.layers or labels.data.shape != image.data.shape:
        fail("the membranes layer is missing or has the wrong shape", widget)
    if logits.shape != image.data.shape or logits.dtype != np.float32:
        fail(f"logits {logits.shape} {logits.dtype} do not match the tomogram", widget)
    # Clean-up only removes voxels (it may remove all of them in a small tomogram).
    if not result["cleanup"] or np.count_nonzero(labels.data & ~(logits > result["threshold"])):
        fail("the mask was not cleaned up, or has voxels below the threshold", widget)

    # A new threshold re-binarizes the stored logits (and cleans the mask up
    # again) without running ETSAM again.
    widget.th_spin.setValue(0.5)
    wait_for(lambda: result["threshold"] == 0.5, 600, "the new threshold", widget)
    print("threshold:", widget.status_label.toolTip())
    if np.count_nonzero(labels.data) > np.count_nonzero(logits > 0.5):
        fail("the cleaned-up mask at threshold 0.5 has more voxels than the logits above 0.5", widget)

    for button, name in ((widget.save_mask_btn, "napari_mask.mrc"),
                         (widget.save_logits_btn, "napari_logits.mrc")):
        path = os.path.join(output_dir, name)
        with patch("napari_etsam._widget.QFileDialog.getSaveFileName", return_value=(path, "")):
            button.click()
        if not os.path.isfile(path):
            fail(f"{name} was not saved", widget)
        print("saved", path)
    viewer.close()
    print("napari plugin: all checks passed.")


if __name__ == "__main__":
    main()
