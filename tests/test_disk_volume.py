"""File-backed volumes must read and write exactly what an array would."""

from pathlib import Path
import tempfile
import unittest

import mrcfile
import numpy as np

from sam2.utils.disk_volume import DiskVolume, read_by_slice


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


if __name__ == "__main__":
    unittest.main()
