from collections.abc import Mapping

import numpy as np
import pytest

from enchilada import Covariance, L1Data


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_observed(
    rng, *, num_time_samples=64, channel_names=("A", "E", "T"), **overrides
):
    """A small, valid time-domain L1Data to build tests on."""
    fields = dict(
        channel_data={
            ch: rng.standard_normal(num_time_samples) for ch in channel_names
        },
        sample_rate_hz=0.5,
        num_time_samples=num_time_samples,
        channel_names=channel_names,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=0.0,
    )
    fields.update(overrides)
    return L1Data(**fields)


@pytest.fixture
def observed(rng):
    return make_observed(rng)


def const_block_state(residual, value):
    """A block state holding `value` in every sample of every channel."""
    return residual.block_state(
        {ch: np.full_like(arr, value) for ch, arr in residual.channel_data.items()}
    )


def white_covariance(data, variance):
    """White noise of `variance` per time sample, on `data`'s grid.

    `variance` is one number for every channel or a channel -> number mapping.
    With X = dt * rfft(x) a stationary one-sided PSD S gives bin variance
    C = (Tobs / 2) S at every rfft bin; white noise has S = 2 sigma^2 / fs, so
    C = N sigma^2 / fs^2. On a WDM grid an active pixel has variance
    sigma^2 / fs and inactive pixels are exactly zero.
    """
    grid = data.grid
    if grid.data_domain == "time":
        scale = 1.0
    elif grid.data_domain == "frequency":
        scale = grid.num_time_samples / grid.sample_rate_hz**2
    else:
        scale = 1.0 / grid.sample_rate_hz
    levels = (
        variance
        if isinstance(variance, Mapping)
        else {ch: variance for ch in data.channel_names}
    )
    return Covariance.like(
        data,
        {
            ch: np.where(grid.active_mask, scale * levels[ch], 0.0)
            for ch in data.channel_names
        },
    )
