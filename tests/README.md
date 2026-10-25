# Tests

Run the unit tests from the repository root (no GPU or checkpoints needed):

```bash
python -m unittest discover -s tests
```

They check that the slice-by-slice pipeline (normalization, stage fusion,
prompts, split processing, file-backed volumes and outputs) produces exactly the
bytes of the original whole-volume computations.

End-to-end tests on a real tomogram run in CI on an Apple silicon Mac (see
`.github/workflows/installer.yml`); they can also be run by hand on any computer.
`fetch_test_tomogram.py` downloads the center slices (32 by default; CI uses 16) of a public tomogram
(only those bytes), `e2e/check_segmentation.py` checks a result (and compares it
with a reference run, such as the CPU), and `e2e/napari_plugin.py` drives the
napari plugin's panel:

```bash
python tests/fetch_test_tomogram.py small.mrc
etsam small.mrc --output-dir gpu --store-logits
ETSAM_DEVICE=cpu etsam small.mrc --output-dir cpu --store-logits
python tests/e2e/check_segmentation.py small.mrc --logits gpu/small_etsam_predicted_logits.mrc \
    --reference cpu/small_etsam_predicted_logits.mrc
```

`ETSAM_DEVICE` (`cuda`, `mps` or `cpu`) overrides the device ETSAM picks.

If results on a device depart from the CPU, `e2e/replay_ops.py` finds the
operations to blame: it runs the model on the CPU and replays every PyTorch
operation on the device with the CPU's exact inputs, ranking each call site by
its own error (a correct float32 operation is within about 1e-6). Data moved
between devices is not replayed; on MPS, asynchronous copies to the CPU were
the problem (see `to_device` in `sam2/modeling/sam2_utils.py`).

```bash
python tests/e2e/replay_ops.py small.mrc --device mps --slices 3
```

To measure time and memory on a real tomogram (needs a GPU and the installed
checkpoints), and to check that another revision gives an identical result:

```bash
cp tests/benchmark_segmentation.py ../benchmark_segmentation.py
git switch --detach <reference-revision>
python ../benchmark_segmentation.py tomogram.mrc --save reference.npy
git switch -
python ../benchmark_segmentation.py tomogram.mrc --compare reference.npy
```

Run it from the repository root: it measures the ETSAM in the current folder.

`--output-type mask`, `--split`, `--normalize-method` and the prompt options
select the same settings as `etsam.segment`.
