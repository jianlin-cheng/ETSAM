"""File-backed volumes must read and write exactly what an array would."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mrcfile
import numpy as np

from sam2.utils.disk_volume import DiskVolume, read_by_slice, scratch_volume, writable_array


def stack(volume):
    return np.stack([volume[index] for index in range(len(volume))])


class DiskVolumeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.rng = np.random.default_rng(11)

    def assertBytesEqual(self, actual, expected):
        self.assertEqual((actual.dtype, actual.shape), (expected.dtype, expected.shape))
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_reads_slices_and_crops_of_mapped_files(self):
        for number, dtype in enumerate((np.int8, np.int16, np.float32, np.dtype(">f4"))):
            source = (self.rng.standard_normal((13, 17, 21)) * 20).astype(dtype)
            path = self.directory / f"{number}.npy"
            np.save(path, source)
            mapped = np.load(path, mmap_mode="r")
            for array, expected in ((mapped, source), (mapped[2:9], source[2:9])):
                volume = read_by_slice(array)
                self.assertIsInstance(volume, DiskVolume)
                expected = expected.astype(np.float32)
                self.assertBytesEqual(stack(volume), expected)
                self.assertBytesEqual(stack(volume[:, 3:11, 4:19]), expected[:, 3:11, 4:19])
                self.assertBytesEqual(np.array(volume), expected)
                self.assertBytesEqual(volume[-1], expected[-1])
                volume.close()
            del mapped

    def test_reads_mrc_files(self):
        source = self.rng.standard_normal((5, 8, 9)).astype(np.float32)
        path = self.directory / "tomogram.mrc"
        with mrcfile.new(path) as mrc:
            mrc.set_data(source)
        with mrcfile.mmap(path, mode="r") as mrc:
            volume = read_by_slice(mrc.data)
            self.assertBytesEqual(stack(volume), source)
            volume.close()

    def test_other_layouts_are_left_as_they_are(self):
        path = self.directory / "fortran.npy"
        np.save(path, np.asfortranarray(self.rng.standard_normal((4, 5, 6))))
        for array in (
            np.load(path, mmap_mode="r"),  # slices are interleaved in the file
            np.load(path, mmap_mode="c"),  # copy-on-write: may differ from the file
            np.zeros((4, 5, 6)),
        ):
            self.assertIs(read_by_slice(array), array)

    def test_writes_slices_and_crops(self):
        expected = self.rng.standard_normal((6, 10, 12)).astype(np.float32)
        with patch.dict(os.environ, {"ETSAM_SCRATCH": "disk"}):
            volume = scratch_volume(expected.shape)
        self.assertIsInstance(volume, DiskVolume)
        crops = [volume[:, :5, :7], volume[:, :5, 7:], volume[:, 5:, :]]
        for index in range(6):
            crops[0][index] = expected[index, :5, :7]
            crops[1][index] = expected[index, :5, 7:]
            crops[2][index] = expected[index, 5:]
        self.assertBytesEqual(stack(volume), expected)
        with writable_array(volume) as array:
            self.assertBytesEqual(np.asarray(array), expected)
            array += 1
        self.assertBytesEqual(stack(volume), expected + 1)
        volume.close()

    def test_scratch_storage_settings(self):
        for setting, shape, on_disk in (
            ("memory", (300, 512, 512), False),
            ("auto", (300, 512, 512), True),
            ("auto", (3, 8, 8), False),
            ("disk", (3, 8, 8), True),
        ):
            with patch.dict(os.environ, {"ETSAM_SCRATCH": setting}):
                volume = scratch_volume(shape)
            self.assertEqual((volume.shape, isinstance(volume, DiskVolume)), (shape, on_disk))
            if on_disk:
                volume.close()
        # Without enough free disk space, scratch volumes stay in RAM.
        full_disk = SimpleNamespace(free=0)
        with patch.dict(os.environ, {"ETSAM_SCRATCH": "disk"}), patch("shutil.disk_usage", return_value=full_disk):
            self.assertIsInstance(scratch_volume((3, 8, 8)), np.ndarray)


if __name__ == "__main__":
    unittest.main()
