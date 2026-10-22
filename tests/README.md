# Tests

Run the unit tests from the repository root (no GPU or checkpoints needed):

```bash
python -m unittest discover -s tests
```

They check that the slice-by-slice pipeline (normalization, stage fusion,
prompts, split processing, file-backed volumes and outputs) produces exactly the
bytes of the original whole-volume computations.

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
