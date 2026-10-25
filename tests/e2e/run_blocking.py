"""Run the etsam command with every Tensor.to copy blocking (diagnostics).

    python tests/e2e/run_blocking.py TOMOGRAM.mrc [etsam options]

Tests whether non-blocking copies between the GPU and the CPU change the result.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

to = torch.Tensor.to


def blocking_to(self, *args, **kwargs):
    kwargs.pop("non_blocking", None)
    return to(self, *args, **kwargs)


torch.Tensor.to = blocking_to

import etsam  # noqa: E402

sys.argv[0] = "etsam"
etsam.main()
