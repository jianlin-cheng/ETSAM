"""ETSAM volumes whose slices are computed when they are read.

Apart from a few volume-wide values (a maximum, or a minimum and maximum), ETSAM's
normalization and stage fusion work voxel by voxel. These classes compute the
volume-wide values once and then produce each slice on demand, so the predictor
never needs a whole normalized volume in memory. The results are identical to
normalizing the whole volume at once.

Like a NumPy array, a volume has a shape and a length and is indexed by slice;
each slice is returned as a new float32 array.
"""

import numpy as np


class NormalizedTomogram:
    """A tomogram with min_max_positive_values normalization.

    Same as etsam.min_max_postive_values_normalize on a float32 copy of the whole
    tomogram: negative values become 0, then everything is divided by the maximum.
    The tomogram itself is not modified.
    """

    def __init__(self, tomogram, maximum=None):
        self.tomogram = tomogram
        self.shape = tomogram.shape
        if maximum is None:
            maximum = np.max([np.max(self._positive(index)) for index in range(len(self))])
        self.maximum = maximum

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            # A crop, e.g. a split-processing quadrant, normalized like the whole.
            return NormalizedTomogram(self.tomogram[index], self.maximum)
        frame = self._positive(index)
        frame /= self.maximum
        return frame

    def _positive(self, index):
        frame = np.array(self.tomogram[index], dtype=np.float32)
        frame[frame < 0] = 0
        return frame


class StageOneLogits:
    """Stage-one logits, stored for fusion into the stage-two input.

    The predictor writes the logits one slice at a time. Each slice is clipped
    to [-5, 5] as it is stored in `storage`, a float32 array of the volume's
    shape, and the volume's minimum and maximum are updated along the way. The
    etsam_stage1_partial prompts (the thresholded logits of every
    `prompt_interval`-th slice) are kept as well.
    """

    def __init__(self, storage, prompt_interval=None, prompt_threshold=None):
        self.storage = storage
        self.shape = storage.shape
        self.prompt_interval = prompt_interval
        self.prompt_threshold = prompt_threshold
        self.prompts = {}
        self.minimum = np.float32(np.inf)
        self.maximum = np.float32(-np.inf)

    def __len__(self):
        return self.shape[0]

    def __setitem__(self, index, logits):
        if self.prompt_interval is not None and index % self.prompt_interval == 0:
            self.prompts[index] = (logits > self.prompt_threshold).astype(np.uint8)
        clipped = np.clip(logits, -5, 5)
        self.minimum = np.minimum(self.minimum, np.min(clipped))
        self.maximum = np.maximum(self.maximum, np.max(clipped))
        self.storage[index] = clipped


class FusedTomogram:
    """The stage-two input: a normalized tomogram plus its normalized stage-one logits.

    Same as tomogram + etsam.min_max_normalize(clipped stage-one logits).
    """

    def __init__(self, tomogram, logits):
        self.tomogram = tomogram
        self.logits = logits
        self.shape = tomogram.shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        logits = self.logits
        frame = logits.storage[index] - logits.minimum
        frame /= logits.maximum - logits.minimum
        frame += self.tomogram[index]
        return frame
