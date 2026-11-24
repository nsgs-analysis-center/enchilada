"""Rate reduction of time-domain `L1Data` behind one fixed anti-alias policy.

`decimate` keeps the lower 80 % of the new band clean: a Kaiser-window FIR
low-pass with 120 dB stopband attenuation, passband edge at 0.8 times the new
Nyquist frequency and stopband edge at the new Nyquist, applied zero-phase by
polyphase filtering. The policy is fixed rather than exposed as parameters so
every campaign decimates the same way; a different policy is a different
function.

SciPy is a core dependency, so ``decimate`` needs no extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import firwin, kaiserord, resample_poly

from enchilada.data import L1Data

if TYPE_CHECKING:
    from numpy.typing import NDArray

STOPBAND_ATTENUATION_DB = 120.0
"""Minimum attenuation above the new Nyquist frequency, in dB."""

PASSBAND_FRACTION = 0.8
"""Passband edge as a fraction of the new Nyquist frequency."""


def anti_alias_taps(factor: int) -> NDArray[np.float64]:
    """FIR coefficients :func:`decimate` uses for ``factor``.

    A Kaiser-window low-pass from :func:`scipy.signal.kaiserord` and
    :func:`scipy.signal.firwin`: transition band from
    ``PASSBAND_FRACTION * fs_new / 2`` to ``fs_new / 2`` with
    ``fs_new = fs / factor``, ``STOPBAND_ATTENUATION_DB`` attenuation, odd
    length so the polyphase filter is exactly zero-phase, unit DC gain. About
    ``78 * factor`` taps.
    """
    transition_width = (1.0 - PASSBAND_FRACTION) / factor  # of the input Nyquist
    numtaps, beta = kaiserord(STOPBAND_ATTENUATION_DB, transition_width)
    numtaps = int(numtaps) | 1
    cutoff = (1.0 + PASSBAND_FRACTION) / (2.0 * factor)  # transition-band centre
    return np.asarray(
        firwin(numtaps, cutoff, window=("kaiser", beta)), dtype=np.float64
    )


def edge_samples(factor: int) -> int:
    """Output samples at each end of a decimated record touched by the filter's
    zero padding: half the filter length, counted in output samples, plus one.
    It is 40 for every ``factor``."""
    return (anti_alias_taps(factor).size // 2) // factor + 1


def decimate(data: L1Data, factor: int) -> L1Data:
    """Return ``data`` at ``sample_rate_hz / factor`` on a new, validated grid.

    Args:
        data: Time-domain `L1Data`.
        factor: Integer rate reduction, at least 2.

    Returns:
        A new `L1Data` with ``num_time_samples = ceil(N / factor)``,
        ``sample_rate_hz = fs / factor``, the same ``start_time_gps`` (output
        sample ``k`` sits at input sample ``k * factor`` because the polyphase
        filter is zero-phase), the same channel order, conventions and
        ``orbit_ephemeris`` object.

    The first and last :func:`edge_samples` output samples carry the filter's
    edge transient from zero padding. Nothing is trimmed, so the grid stays
    predictable; trim and re-slice if a campaign cares.

    Raises:
        ValueError: for frequency or WDM input, ``factor < 2``, or a record
            shorter than the anti-alias filter.
        TypeError: for a non-integer ``factor``.
    """
    if data.data_domain != "time":
        raise ValueError(
            f"decimate takes time-domain L1Data, got data_domain="
            f"{data.data_domain!r}; transform to 'time' first"
        )
    if isinstance(factor, bool) or not isinstance(factor, (int, np.integer)):
        raise TypeError(f"factor must be an integer, got {factor!r}")
    if factor < 2:
        raise ValueError(f"factor must be at least 2, got {factor}")

    factor = int(factor)
    taps = anti_alias_taps(factor)
    if data.num_time_samples < taps.size:
        raise ValueError(
            f"record of {data.num_time_samples} samples is shorter than the "
            f"{taps.size}-tap anti-alias filter for factor {factor}; decimate a "
            f"longer record or use a smaller factor"
        )
    channel_data = {
        name: np.asarray(
            resample_poly(data.channel_data[name], 1, factor, window=taps),
            dtype=np.float64,
        )
        for name in data.channel_names
    }
    return L1Data(
        channel_data,
        sample_rate_hz=data.sample_rate_hz / factor,
        tdi_generation=data.tdi_generation,
        physical_observable=data.physical_observable,
        channel_names=data.channel_names,
        start_time_gps=data.start_time_gps,
        orbit_ephemeris=data.orbit_ephemeris,
    )
