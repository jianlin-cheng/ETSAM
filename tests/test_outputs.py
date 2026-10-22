"""Results written to files: the CLI's MRC files and segment(output_path=...)."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mrcfile
import numpy as np

import etsam


def fake_two_stages(volume, *args, output, mask_threshold=None, **kwargs):
    """Stand-in for both stages: the logits are the normalized input minus 0.5."""
    for index in range(len(volume)):
        logits = volume[index] - np.float32(0.5)
        output[index] = logits if mask_threshold is None else logits > mask_threshold
    return output


class OutputTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.tomogram = np.random.default_rng(4).standard_normal((6, 10, 12)).astype(np.float32)
        self.logits = etsam.min_max_postive_values_normalize(self.tomogram.copy()) - np.float32(0.5)
        # Plain functions rather than mocks, which would keep the input file open.
        for name, stand_in in (
            ("ensure_checkpoints", lambda progress=None: ("a", "b")),
            ("run_two_stage_inference_on_tomogram", fake_two_stages),
        ):
            patcher = patch.object(etsam, name, new=stand_in)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_cli(self, *options):
        path = self.directory / "tomo.mrc"
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(self.tomogram)
            mrc.voxel_size = (1.5, 2.5, 3.5)
        with patch.object(sys, "argv", ["etsam", str(path), "--output-dir", str(self.directory), *options]):
            etsam.main()

    def read(self, name):
        with mrcfile.open(self.directory / name) as mrc:
            return mrc.data.copy(), mrc.voxel_size.copy(), mrc.header.copy()

    def test_cli_mask(self):
        self.run_cli("--logit-threshold", "0.1")
        mask, voxel_size, _ = self.read("tomo_etsam_predicted_mask.mrc")
        np.testing.assert_array_equal(mask, (self.logits > 0.1).astype(np.uint8))
        self.assertEqual(tuple(voxel_size.tolist()), (1.5, 2.5, 3.5))
        self.assertFalse((self.directory / "tomo_etsam_predicted_logits.mrc").exists())

    def test_cli_logits_are_written_to_their_mrc_file(self):
        self.run_cli("--store-logits", "--split-processing")
        logits, voxel_size, header = self.read("tomo_etsam_predicted_split_processing_logits.mrc")
        mask, _, _ = self.read("tomo_etsam_predicted_split_processing_mask.mrc")
        self.assertEqual(logits.tobytes(), self.logits.tobytes())
        np.testing.assert_array_equal(mask, (self.logits > -0.25).astype(np.uint8))
        self.assertEqual(tuple(voxel_size.tolist()), (1.5, 2.5, 3.5))
        np.testing.assert_allclose(
            [header.dmin, header.dmax, header.dmean, header.rms],
            [self.logits.min(), self.logits.max(), self.logits.mean(), self.logits.std()],
            rtol=1e-5,
        )
        self.assertEqual(sorted(p.name for p in self.directory.iterdir() if p.name.startswith(".")), [])

    def test_segment_to_npy_file(self):
        path = self.directory / "logits.npy"
        for split_processing in (False, True):
            logits = etsam.segment(self.tomogram, split_processing=split_processing, output_path=path)
            self.assertEqual(np.asarray(logits).tobytes(), self.logits.tobytes())
            del logits

    def test_failed_segmentation_keeps_the_previous_file(self):
        path = self.directory / "logits.npy"
        np.save(path, np.zeros(3))

        def cancel(fraction, message):
            raise KeyboardInterrupt

        with patch.object(etsam, "run_two_stage_inference_on_tomogram", new=lambda *a, progress, **k: progress(1, 1, 6)):
            with self.assertRaises(KeyboardInterrupt):
                etsam.segment(self.tomogram, output_path=path, progress=cancel)
        np.testing.assert_array_equal(np.load(path), np.zeros(3))
        self.assertEqual([p.name for p in self.directory.iterdir()], ["logits.npy"])


if __name__ == "__main__":
    unittest.main()
