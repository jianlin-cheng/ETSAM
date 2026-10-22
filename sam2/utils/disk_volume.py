"""Volumes stored in files, read one slice at a time with ordinary file reads.

Reading a memory-mapped volume keeps every page that was touched in the
process's memory (its working set) until the mapping is closed, so a whole
tomogram accumulates in RAM as it is processed. Reading each slice with a file
read instead leaves the data to the operating system's file cache, which can
drop it whenever memory is needed.

Like a NumPy array, a DiskVolume has a shape and a length and is indexed by
slice; volume[i] returns a new float32 array, and volume[:, y0:y1, x0:x1] a crop.
"""

import threading

import numpy as np


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
