"""Load one day of Mojito L1 data into enchilada and decimate it.

Needs a Mojito L1 file:

    uv run --no-sync python examples/load_mojito.py /path/to/mojito_light_..._L1.h5

or set ENCHILADA_MOJITO_L1 to the file path. This example only reads data;
blocks and the Wheel live in the other examples.
"""

import os
import sys

import numpy as np

from enchilada import decimate, load_mojito_l1


def main(path, *, duration_s=86400.0, factor=10):
    """Load A, E, T for `duration_s` seconds, decimate by `factor`, print a summary."""
    native = load_mojito_l1(path, duration_s=duration_s)
    data = decimate(native, factor)
    print(f"native grid: {native.grid}")
    print(f"decimated grid: {data.grid} (t0 = {data.t0} s on the file's TCB clock)")
    for name in data.channel_names:
        native_rms = float(np.sqrt(np.mean(native.channel_data[name] ** 2)))
        rms = float(np.sqrt(np.mean(data.channel_data[name] ** 2)))
        print(
            f"  {name}: rms {native_rms:.3e} native, {rms:.3e} decimated "
            f"(fractional frequency; the decimated record keeps its edge samples)"
        )
    first, last = data.orbit_ephemeris.time_range_gps
    print(f"orbit span: {first:.3f} to {last:.3f} s")
    return data


if __name__ == "__main__":
    file_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ENCHILADA_MOJITO_L1")
    )
    if not file_path:
        sys.exit("usage: load_mojito.py PATH  (or set ENCHILADA_MOJITO_L1)")
    main(file_path)
