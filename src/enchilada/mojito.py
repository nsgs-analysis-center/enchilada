"""Read DDPC Mojito L1 HDF5 files into `L1Data`.

Mojito is the LISA DDPC Common Dataset 1 (LISA-DDPC-SEG-TN-008). An L1 file
stores second-generation TDI time series under ``tdis/`` as ``X2``, ``Y2``,
``Z2``, ``A2``, ``E2`` and ``T2`` in total-frequency units (Hz); the time grid
in the attributes ``t0``, ``dt``, ``fs`` and ``size`` of ``tdis/sampling``;
quality flags in ``tdis/tdi_flags`` (0 usable, 1 not); the laser frequency in
the root attribute ``laser_frequency``; and a spacecraft ephemeris under
``orbits/`` in BCRS (equatorial) metres, in the layout
:meth:`enchilada.orbits.NumericOrbit.from_hdf5` reads by default.

Times are TCB seconds after the LISA reference epoch. The orbit table shares
that clock, and the loader carries them in ``start_time_gps`` unchanged. The
stored ``A2``/``E2``/``T2`` equal the orthonormal combination of
``X2``/``Y2``/``Z2`` that :meth:`enchilada.Covariance.in_basis` uses, so a
campaign may load either triple.

Reads with h5py, a core dependency.
"""

from __future__ import annotations

import math
import os

import h5py
import numpy as np

from enchilada.data import L1Data
from enchilada.orbits import NumericOrbit

TDI_CHANNELS = ("A", "E", "T", "X", "Y", "Z")
"""Channel names the loader accepts; dataset ``tdis/<name>2`` holds each."""

TDI_GENERATION = "2.0"
"""Every Mojito TDI dataset is a second-generation combination."""

_RATE_TOLERANCE = 1e-9


def _resolve_window(
    t0: float, dt: float, size: int, start_offset_s: float, duration_s: float | None
) -> tuple[int, int]:
    """First sample index and sample count of the requested window.

    The start rounds to the nearest sample and the count rounds down, so a
    window never reaches past what was asked for.
    """
    if not math.isfinite(start_offset_s):
        raise ValueError(f"start_offset_s must be finite, got {start_offset_s!r}")
    if duration_s is not None and not math.isfinite(duration_s):
        raise ValueError(f"duration_s must be finite, got {duration_s!r}")
    first = int(round(start_offset_s / dt))
    count = size - first if duration_s is None else int(math.floor(duration_s / dt))
    if first < 0 or first >= size or count < 1 or first + count > size:
        last_time = t0 + (size - 1) * dt
        raise ValueError(
            f"requested window start_offset_s={start_offset_s}, "
            f"duration_s={duration_s} (samples {first} to {first + count - 1}) "
            f"does not fit the file, which holds {size} samples at dt = {dt} s "
            f"spanning [{t0}, {last_time}] s; offsets count from the first sample"
        )
    return first, count


def _check_length(file: h5py.File, dataset: str, size: int) -> None:
    """Refuse a dataset whose length disagrees with the ``size`` attribute.

    h5py clips a slice that runs past a dataset's end, so an unchecked short
    dataset would return fewer samples than asked for, or leave samples of the
    window unflagged. Reads only the shape metadata, no data.
    """
    shape = file[dataset].shape
    if shape != (size,):
        raise ValueError(
            f"{dataset} has shape {shape} but tdis/sampling says size = {size}; "
            f"the file is inconsistent"
        )


def load_mojito_l1(
    path: str | os.PathLike[str],
    *,
    channels: tuple[str, ...] = ("A", "E", "T"),
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
    orbit: bool = True,
) -> L1Data:
    """Load a window of TDI channels from a Mojito L1 file.

    Args:
        path: The L1 HDF5 file.
        channels: Names from ``A E T X Y Z``, in the order the `L1Data` keeps;
            each reads dataset ``tdis/<name>2``.
        start_offset_s: Seconds after the file's first sample; rounds to the
            nearest sample.
        duration_s: Window length in seconds, rounded down to whole samples;
            None reads to the end of the file. On the full Mojito Light record
            that is about 252 million float64 samples per channel, 2 GB each;
            prefer a window.
        orbit: Attach the file's ``orbits`` table as a `NumericOrbit`
            (BCRS positions rotated to the ecliptic frame).

    Returns:
        Time-domain `L1Data` in fractional frequency (the stored Hz divided by
        the root attribute ``laser_frequency``), ``tdi_generation="2.0"``,
        ``sample_rate_hz`` from the ``fs`` attribute, and ``start_time_gps =
        t0 + first_sample * dt`` on the file's TCB clock.

    Raises:
        ValueError: unknown or missing channel, a file without the Mojito
            layout (no ``tdis/sampling`` group, ``tdis/tdi_flags`` dataset or
            root attribute ``laser_frequency``), a window outside the file,
            ``dt`` and ``fs`` attributes that disagree, a ``tdi_flags`` or
            channel dataset whose length differs from the ``size`` attribute,
            a ``laser_frequency`` that is not positive and finite, or any
            nonzero ``tdi_flags`` value in the window (enchilada has no gap
            mask).
    """
    channels = tuple(channels)
    unknown = [name for name in channels if name not in TDI_CHANNELS]
    if unknown:
        raise ValueError(
            f"unknown Mojito TDI channel(s) {unknown}; choose from "
            f"{', '.join(TDI_CHANNELS)}"
        )
    file_path = os.fspath(path)
    with h5py.File(file_path, "r") as file:
        if "tdis/sampling" not in file:
            raise ValueError(
                f"{file_path} has no tdis/sampling group; is it a Mojito L1 file? "
                f"Top-level members: {sorted(file)}"
            )
        sampling = file["tdis/sampling"].attrs
        t0 = float(sampling["t0"])
        dt = float(sampling["dt"])
        fs = float(sampling["fs"])
        size = int(sampling["size"])
        if abs(dt * fs - 1.0) > _RATE_TOLERANCE:
            raise ValueError(
                f"tdis/sampling attributes disagree: dt = {dt} s and fs = {fs} Hz "
                f"give dt * fs = {dt * fs}, expected 1"
            )
        first, count = _resolve_window(t0, dt, size, start_offset_s, duration_s)
        if "laser_frequency" not in file.attrs:
            raise ValueError(
                f"{file_path} has no root attribute laser_frequency, needed to "
                f"convert Hz to fractional frequency; root attributes: "
                f"{sorted(file.attrs)}"
            )
        laser_frequency_hz = float(file.attrs["laser_frequency"])
        if not (math.isfinite(laser_frequency_hz) and laser_frequency_hz > 0.0):
            raise ValueError(
                "root attribute laser_frequency must be a positive finite "
                f"frequency in Hz, got {laser_frequency_hz}"
            )
        window = slice(first, first + count)
        if "tdis/tdi_flags" not in file:
            raise ValueError(
                f"{file_path} has no dataset tdis/tdi_flags; the tdis group holds "
                f"{sorted(file['tdis'])}"
            )
        _check_length(file, "tdis/tdi_flags", size)
        flagged = np.flatnonzero(np.asarray(file["tdis/tdi_flags"][window]))
        if flagged.size:
            raise ValueError(
                f"{flagged.size} of {count} samples in the window carry a nonzero "
                f"tdi_flags value (first at file sample {first + int(flagged[0])}); "
                f"enchilada has no gap mask, so choose a window of usable samples"
            )
        channel_data: dict[str, np.ndarray] = {}
        for name in channels:
            dataset = f"tdis/{name}2"
            if dataset not in file:
                raise ValueError(
                    f"{file_path} has no dataset {dataset}; the tdis group holds "
                    f"{sorted(file['tdis'])}"
                )
            _check_length(file, dataset, size)
            values = np.asarray(file[dataset][window], dtype=np.float64)
            values /= laser_frequency_hz  # Hz -> fractional frequency
            channel_data[name] = values
    ephemeris = (
        NumericOrbit.from_hdf5(
            file_path, group_path="orbits", coordinate_frame="equatorial"
        )
        if orbit
        else None
    )
    return L1Data(
        channel_data,
        sample_rate_hz=fs,
        tdi_generation=TDI_GENERATION,
        physical_observable="fractional_frequency",
        channel_names=channels,
        start_time_gps=t0 + first * dt,
        orbit_ephemeris=ephemeris,
    )
