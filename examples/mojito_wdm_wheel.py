"""Load a Mojito L1 window, decimate it, store it as WDM and run a Wheel on it.

Needs the `mojito` extra (h5py) and a Mojito L1 file:

    uv run --no-sync python examples/mojito_wdm_wheel.py /path/to/mojito_light_..._L1.h5

or set ENCHILADA_MOJITO_L1 to the file path. `load_mojito.py` is the data side
of the Mojito story; this is the Wheel side. A WDM observation needs exactly
`num_frequency_divisions * num_time_divisions` samples, both counts even, and
neither `transform` nor the loader pads or trims, so `fit_window` sizes the
window first: the longest one of at most `duration_s` whose decimated record
fits the grid.
"""

import os
import sys

import numpy as np

from enchilada import Wheel, decimate, load_mojito_l1, transform
from enchilada.testing import EchoBlock

SECONDS_PER_DAY = 86400.0


def fit_window(duration_s, *, dt, factor, num_frequency_divisions):
    """Longest window of at most `duration_s` whose record, decimated by `factor`,
    has `num_frequency_divisions * num_time_divisions` samples with
    `num_time_divisions` even. Returns `(native_duration_s, num_time_divisions)`.

    `dt` is the file's native sampling interval; one WDM time column spans
    `factor * num_frequency_divisions` native samples.
    """
    samples_per_column = factor * num_frequency_divisions
    num_time_divisions = int(duration_s / (samples_per_column * dt))
    num_time_divisions -= num_time_divisions % 2
    if num_time_divisions < 2:
        raise ValueError(
            f"duration_s={duration_s} holds fewer than two WDM time columns of "
            f"{samples_per_column * dt} s"
        )
    return num_time_divisions * samples_per_column * dt, num_time_divisions


def main(
    path, *, num_frequency_divisions=512, factor=10, duration_s=7 * SECONDS_PER_DAY
):
    """Fit a window, load and decimate it, translate to WDM, run one Wheel cycle.

    `factor` must be at least 2 (`decimate` refuses 1).
    """
    # A cheap probe for the file's sampling interval: one minute of one channel.
    probe = load_mojito_l1(path, channels=("A",), duration_s=60.0, orbit=False)
    native_duration_s, num_time_divisions = fit_window(
        duration_s,
        dt=probe.dt,
        factor=factor,
        num_frequency_divisions=num_frequency_divisions,
    )
    native = load_mojito_l1(path, duration_s=native_duration_s)
    # decimate leaves edge_samples(factor) = 40 filter-transient samples at each end
    # (enchilada.resample). WDM atoms overlap neighbouring columns and the transform
    # is periodic, so the transient leaks several columns inward at both ends; trim
    # or down-weight those columns if a campaign is sensitive to them.
    data = decimate(native, factor)  # exactly Nf * Nt samples
    observed = transform(data, "wdm", num_frequency_divisions=num_frequency_divisions)
    grid = observed.wdm_grid
    print(
        f"window: {native_duration_s} s (factor {factor} "
        f"x Nf {num_frequency_divisions} x Nt {num_time_divisions} "
        f"= {native.num_time_samples} native samples at dt = {probe.dt} s)"
    )
    band_spacing_hz = 1.0 / (2.0 * grid.num_frequency_divisions * data.dt)
    column_duration_s = grid.num_frequency_divisions * data.dt
    print(
        f"WDM grid: Nf = {grid.num_frequency_divisions}, "
        f"Nt = {grid.num_time_divisions}, dF = {band_spacing_hz:.3e} Hz, "
        f"dT = {column_duration_s} s, shape {grid.array_shape}"
    )
    for name in observed.channel_names:
        rms = float(np.sqrt(np.mean(observed.channel_data[name] ** 2)))
        print(f"  {name}: rms {rms:.3e} per WDM coefficient")
    # WDM observations: their grid is the campaign's home grid, no count needed.
    wheel = Wheel(observed, random_seed=0)
    wheel.add(EchoBlock(name="echo"))
    # EchoBlock adds no signal, so its "[echo] ... residual RMS" line repeats the A
    # rms above: the block was served the WDM coefficients.
    wheel.run(num_cycles=1)
    residual = wheel.residual()
    print(f"residual: domain {residual.data_domain} on {residual.wdm_grid}")
    return wheel


if __name__ == "__main__":
    file_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ENCHILADA_MOJITO_L1")
    )
    if not file_path:
        sys.exit("usage: mojito_wdm_wheel.py PATH  (or set ENCHILADA_MOJITO_L1)")
    main(file_path)
