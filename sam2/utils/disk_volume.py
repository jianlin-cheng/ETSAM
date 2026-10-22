"""Volumes stored in files, read and written one slice at a time.

Reading a memory-mapped volume keeps every page that was touched in the
process's memory (its working set) until the mapping is closed, so a whole
tomogram accumulates in RAM as it is processed. Reading each slice with a file
read instead leaves the data to the operating system's file cache, which can
drop it whenever memory is needed. The same goes for writes, which lets
large intermediate volumes live in temporary files (see scratch_volume).

Like a NumPy array, a DiskVolume has a shape and a length and is indexed by
slice; volume[i] returns a new float32 array, volume[i] = frame writes a slice,
and volume[:, y0:y1, x0:x1] is a crop.
"""

from contextlib import contextmanager
import os
import shutil
import tempfile
import threading

import numpy as np

# Smaller scratch volumes stay in RAM (unless ETSAM_SCRATCH=disk).
SCRATCH_DISK_MIN_BYTES = 256 * 2**20


class DiskVolume:
    """A (Z, Y, X) volume at `offset` in an open binary file, with byte `strides`."""

    def __init__(self, file, shape, dtype=np.float32, offset=0, strides=None, lock=None):
        self.file = file
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.offset = offset
        self.strides = strides or (
            self.shape[1] * self.shape[2] * self.dtype.itemsize,
            self.shape[2] * self.dtype.itemsize,
            self.dtype.itemsize,
        )
        # The prefetch thread reads while the main thread uses the same file.
        self.lock = lock or threading.Lock()

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        array = np.empty(self.shape, dtype=dtype or np.float32)
        for index in range(len(self)):
            array[index] = self[index]
        return array

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._crop(*index)
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        # Read the byte range of the slice, then pick its voxels out of it.
        height, width = self.shape[1:]
        span = (height - 1) * self.strides[1] + (width - 1) * self.strides[2] + self.dtype.itemsize
        data = np.empty(span, dtype=np.uint8)
        with self.lock:
            self.file.seek(self.offset + index * self.strides[0])
            if self.file.readinto(data) != span:
                raise EOFError(f"{self.file.name} is shorter than its volume")
        frame = np.ndarray((height, width), self.dtype, data, strides=self.strides[1:])
        return np.ascontiguousarray(frame, dtype=np.float32)

    def __setitem__(self, index, frame):
        if not -len(self) <= index < len(self):
            raise IndexError(index)
        index %= len(self)
        frame = np.ascontiguousarray(frame, dtype=self.dtype)
        height, width = self.shape[1:]
        if self.strides[2] != self.dtype.itemsize:
            raise ValueError("only volumes with contiguous rows can be written")
        start = self.offset + index * self.strides[0]
        with self.lock:
            if self.strides[1] == width * self.dtype.itemsize:
                self.file.seek(start)
                self.file.write(frame.data)
            else:  # a crop: write row by row
                for row in range(height):
                    self.file.seek(start + row * self.strides[1])
                    self.file.write(frame[row].data)

    def close(self):
        self.file.close()

    def _crop(self, z, y, x):
        if z != slice(None):
            raise IndexError("only [:, y0:y1, x0:x1] crops are supported")
        (y0, y1, ystep), (x0, x1, xstep) = y.indices(self.shape[1]), x.indices(self.shape[2])
        if ystep != 1 or xstep != 1:
            raise IndexError("crops must be contiguous")
        offset = self.offset + y0 * self.strides[1] + x0 * self.strides[2]
        shape = (self.shape[0], y1 - y0, x1 - x0)
        return DiskVolume(self.file, shape, self.dtype, offset, self.strides, self.lock)


def read_by_slice(array):
    """Read a read-only memory-mapped (Z, Y, X) array as a DiskVolume.

    Other arrays, and mapped layouts that do not keep each slice in its own
    byte range, are returned as they are.
    """
    if not (isinstance(array, np.memmap) and array.mode == "r" and array.ndim == 3):
        return array
    span = sum((n - 1) * stride for n, stride in zip(array.shape[1:], array.strides[1:]))
    if min(array.strides) <= 0 or span + array.itemsize > array.strides[0] or 0 in array.shape:
        return array
    # The array may be a view: find its offset from the start of the mapped data.
    root = array
    while isinstance(root.base, np.memmap):
        root = root.base
    offset = root.offset + array.ctypes.data - root.ctypes.data
    return DiskVolume(open(array.filename, "rb"), array.shape, array.dtype, offset, array.strides)


def scratch_volume(shape):
    """A temporary float32 (Z, Y, X) volume.

    Large volumes are stored in a temporary file that the operating system
    deletes when it is closed (also if the process is killed); smaller ones,
    and any that do not fit on the disk, are NumPy arrays in RAM. Set
    ETSAM_SCRATCH to "disk" or "memory" to choose either (default "auto"), and
    TMPDIR (TEMP on Windows) to choose the folder.
    """
    setting = os.environ.get("ETSAM_SCRATCH", "auto").lower()
    if setting not in ("auto", "disk", "memory"):
        raise ValueError("ETSAM_SCRATCH must be auto, disk or memory")
    nbytes = int(np.prod(shape)) * 4
    if setting == "memory" or (setting == "auto" and nbytes < SCRATCH_DISK_MIN_BYTES):
        return np.empty(shape, dtype=np.float32)
    try:
        if shutil.disk_usage(tempfile.gettempdir()).free < nbytes:
            return np.empty(shape, dtype=np.float32)
        file = tempfile.TemporaryFile(prefix="etsam-")
        file.truncate(nbytes)
    except OSError:
        return np.empty(shape, dtype=np.float32)
    return DiskVolume(file, shape)


@contextmanager
def writable_array(volume):
    """A volume as one writable NumPy array, for whole-volume computations.

    A DiskVolume is memory-mapped for the duration of the with block.
    """
    if isinstance(volume, np.ndarray):
        yield volume
        return
    array = np.memmap(volume.file, volume.dtype, "r+", offset=volume.offset, shape=volume.shape)
    try:
        yield array
    finally:
        array.flush()
        array._mmap.close()
