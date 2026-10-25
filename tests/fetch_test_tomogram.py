"""Download a few center slices of a real tomogram, for quick end-to-end tests.

    python tests/fetch_test_tomogram.py OUTPUT.mrc [--slices 32] [--url URL]

Only the MRC header and the requested slices are downloaded (HTTP range
requests): an MRC file stores its slices one after another after the header, so
the center slices are one contiguous byte range. The default tomogram is from
EMPIAR-13056 (mouse embryonic fibroblast lamella, 9.98 Å/px, 682 × 960 × 334);
its 32 center slices are 84 MB. ETSAM finds membranes only after tracking for
about 10 slices, so fewer slices give an empty mask. Needs only numpy and
mrcfile.
"""

import argparse
import http.client
import os
import time
import urllib.request

import mrcfile
import numpy as np

DEFAULT_URL = (
    "https://ftp.ebi.ac.uk/empiar/world_availability/13056/data/morphometrics_thickness/"
    "Tomogram_bin6/MIM019_2_lam1_ts_002.mrc_9.98Apx.mrc"
)
MODES = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16}


def fetch(url, start, end, retries=5):
    """Bytes start..end (inclusive) of url."""
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"{url} does not support range requests")
                data = response.read()
            if len(data) != end - start + 1:
                raise OSError(f"incomplete download ({len(data)} of {end - start + 1} bytes)")
            return data
        except (OSError, http.client.HTTPException):  # e.g. a dropped connection (IncompleteRead)
            if attempt == retries:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    header = np.frombuffer(fetch(args.url, 0, 1023), dtype="<i4")
    nx, ny, nz, mode = (int(value) for value in header[:4])
    extended = int(header[23])
    if tuple(header[16:19]) != (1, 2, 3) or mode not in MODES:
        raise RuntimeError(f"unsupported MRC file (mode {mode}, axes {tuple(header[16:19])})")
    cell = np.frombuffer(header[10:13].tobytes(), dtype="<f4")
    voxel_size = tuple(float(c) / n for c, n in zip(cell, (nx, ny, nz)))
    dtype = np.dtype(MODES[mode]).newbyteorder("<")

    count = min(args.slices, nz)
    first = (nz - count) // 2
    slice_bytes = nx * ny * dtype.itemsize
    offset = 1024 + extended + first * slice_bytes
    print(f"{args.url}: {nx} x {ny} x {nz} (mode {mode}); slices {first}..{first + count - 1}")
    data = np.frombuffer(fetch(args.url, offset, offset + count * slice_bytes - 1), dtype=dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with mrcfile.new(args.output, overwrite=True) as mrc:
        mrc.set_data(data.reshape(count, ny, nx).astype(np.float32))
        mrc.voxel_size = voxel_size
    print(f"Wrote {args.output}: shape {(count, ny, nx)}, voxel size {voxel_size[0]:.2f} Å")


if __name__ == "__main__":
    main()
