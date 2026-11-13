"""Grid: the four numbers that fix a channel array's layout, and L1Data.grid."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from enchilada import Grid, L1Data, WDMGrid
from enchilada.domains import DOMAINS


def test_domains_are_defined_once_and_re_exported():
    assert DOMAINS == ("time", "frequency", "wdm")
    assert L1Data.DOMAINS is DOMAINS


class TestMembers:
    def test_time_grid(self):
        grid = Grid("time", 64, 0.5)
        assert grid.channel_shape == (64,)
        assert grid.active_mask.shape == (64,) and grid.active_mask.all()
        assert grid.observation_duration_s == 128.0
        assert grid.wdm_grid is None

    @pytest.mark.parametrize(("n", "bins"), [(63, 32), (64, 33), (1, 1)])
    def test_frequency_grid(self, n, bins):
        grid = Grid("frequency", n, 1.0)
        assert grid.channel_shape == (bins,)
        assert grid.active_mask.shape == (bins,) and grid.active_mask.all()

    def test_wdm_grid(self):
        wdm = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
        grid = Grid("wdm", 64, 1.0, wdm)
        assert grid.channel_shape == (9, 8)
        np.testing.assert_array_equal(grid.active_mask, wdm.active_mask)
        assert not grid.active_mask[0, 1] and grid.active_mask[1, 1]

    def test_counts_and_rates_are_normalised(self):
        grid = Grid("time", np.int64(64), np.float32(0.5))
        assert type(grid.num_time_samples) is int
        assert type(grid.sample_rate_hz) is float
        assert grid == Grid("time", 64, 0.5)

    def test_equality_and_same_record(self):
        time = Grid("time", 64, 0.5)
        frequency = Grid("frequency", 64, 0.5)
        assert time != frequency
        assert time.same_record(frequency)
        assert not time.same_record(Grid("time", 32, 0.5))
        assert not time.same_record(Grid("time", 64, 1.0))
        assert hash(time) == hash(Grid("time", 64, 0.5))


class TestValidation:
    @pytest.mark.parametrize(
        ("args", "error", "match"),
        [
            (("wavelet", 64, 1.0), ValueError, "data_domain"),
            (("time", 0, 1.0), ValueError, "num_time_samples"),
            (("time", -4, 1.0), ValueError, "num_time_samples"),
            (("time", True, 1.0), ValueError, "num_time_samples"),
            (("time", 64.0, 1.0), ValueError, "num_time_samples"),
            (("time", 64, 0.0), ValueError, "sample_rate_hz"),
            (("time", 64, -1.0), ValueError, "sample_rate_hz"),
            (("time", 64, float("inf")), ValueError, "sample_rate_hz"),
            (("time", 64, "1.0"), TypeError, "sample_rate_hz"),
            (("time", 64, True), TypeError, "sample_rate_hz"),
            (("wdm", 64, 1.0), TypeError, "wdm_grid"),
            (("wdm", 64, 1.0, (8, 8)), TypeError, "wdm_grid"),
            (
                (
                    "wdm",
                    32,
                    1.0,
                    WDMGrid(num_frequency_divisions=8, num_time_divisions=8),
                ),
                ValueError,
                "num_time_samples",
            ),
            (
                (
                    "time",
                    64,
                    1.0,
                    WDMGrid(num_frequency_divisions=8, num_time_divisions=8),
                ),
                ValueError,
                "wdm_grid",
            ),
        ],
    )
    def test_rejects(self, args, error, match):
        with pytest.raises(error, match=match):
            Grid(*args)

    def test_is_frozen(self):
        grid = Grid("time", 64, 1.0)
        with pytest.raises(AttributeError):
            grid.num_time_samples = 32  # type: ignore[misc]


class TestL1DataGrid:
    def test_time_data(self, rng):
        data = make_observed(rng)
        assert data.grid == Grid("time", 64, 0.5)

    def test_frequency_data(self, rng):
        spectrum = make_observed(rng).to_frequency()
        assert spectrum.grid == Grid("frequency", 64, 0.5)
        assert spectrum.grid.channel_shape == spectrum.channel_data["A"].shape

    def test_wdm_data(self, rng):
        wdm = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
        data = replace(
            make_observed(rng),
            data_domain="wdm",
            channel_data={
                ch: np.zeros(wdm.array_shape) for ch in make_observed(rng).channel_names
            },
        )
        assert data.grid == Grid("wdm", 64, 0.5, wdm)
        assert data.grid.wdm_grid is data.wdm_grid
