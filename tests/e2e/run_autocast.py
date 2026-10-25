"""Run the etsam command with another precision (diagnostics).

    python tests/e2e/run_autocast.py {bfloat16,float16} TOMOGRAM.mrc [etsam options]

Runs the model under autocast to the given type on its device (ETSAM uses
bfloat16 on CUDA and float32 elsewhere), to compare precisions on MPS.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

import etsam  # noqa: E402
import inference  # noqa: E402

dtype = getattr(torch, sys.argv.pop(1))
inference.autocast = lambda device: torch.autocast(device_type=torch.device(device).type, dtype=dtype)
sys.argv[0] = "etsam"
etsam.main()
