"""Covariance translation between grids: the diagonal spectral-model rule.

Every covariance is read as a piecewise-constant one-sided spectral model
S(f, t) and the target grid takes weighted means of it with the Meyer
window's band weights and time weights. The kernels are pinned against
domania's dense oracles on a small grid (``build_W``, ``build_C_time``), so
the exact cases are exact and the smoothed cases preserve the trace.
"""

import domania
import numpy as np
import pytest
from domania.wilson import matrix

from enchilada import Covariance, Grid, L1Data, WDMGrid, transform
from enchilada._spectral import _layout, _layout_of, translate_blocks

NF, NT, FS = 8, 32, 2.0
N = NF * NT
DT = 1.0 / FS
NUM_BINS = N // 2 + 1


def grid(domain, num_frequency_divisions=NF):
    wdm = (
        WDMGrid(
            num_frequency_divisions=num_frequency_divisions,
            num_time_divisions=N // num_frequency_divisions,
        )
        if domain == "wdm"
        else None
    )
    return Grid(domain, N, FS, wdm)


TIME, FREQUENCY, WDM = grid("time"), grid("frequency"), grid("wdm")


def one(values):
    """A single channel pair's blocks: shape (1, 1, *channel_shape)."""
    return np.asarray(values)[None, None]


def psd(frequency_hz):
    """A smooth coloured one-sided PSD, finite at f = 0."""
    f = np.asarray(frequency_hz, dtype=float)
    return 1e-2 * (1 + 1 / (1 + (f / (0.1 / DT)) ** 4)) * (1 + 0.5 * np.cos(40 * f))


@pytest.fixture(scope="module")
def dense():
    """The dense WDM matrix W (wdm = W x), its inverse and the dt * rfft matrix."""
    W = matrix.build_W(NF, NT, DT)
    k = np.arange(NUM_BINS)[:, None]
    j = np.arange(N)[None, :]
    F = DT * np.exp(-2j * np.pi * k * j / N)
    return W, W.T / DT, F


def relative(actual, expected):
    return np.abs(actual / expected - 1).max()


class TestWhiteRoundTrips:
    @pytest.mark.parametrize(
        "source, via, level",
        [
            (TIME, WDM, 2.0),
            (TIME, FREQUENCY, 2.0),
            (FREQUENCY, WDM, 3.0),
            (FREQUENCY, TIME, 3.0),
            (WDM, TIME, 0.7),
            (WDM, FREQUENCY, 0.7),
        ],
        ids=["t-w-t", "t-f-t", "f-w-f", "f-t-f", "w-t-w", "w-f-w"],
    )
    def test_white_noise_comes_back_exactly(self, source, via, level):
        values = np.full(source.channel_shape, level)
        if source.data_domain == "frequency":
            values = values + 0j
        there = translate_blocks(one(values), source, via)
        back = translate_blocks(there, via, source)
        assert back.shape == one(values).shape
        assert relative(back.real, level) < 1e-14


class TestExactCases:
    def test_stationary_psd_into_wdm_is_domanias_band_mean(self):
        f = np.fft.rfftfreq(N, DT)
        bins = one((N * DT / 2) * psd(f) + 0j)
        mine = translate_blocks(bins, FREQUENCY, WDM)[0, 0]
        theirs = domania.wdm_covariance(
            lambda ff, t: psd(ff)[None, None], Nf=NF, Nt=NT, dt=DT
        )[0, 0]
        assert relative(mine, theirs) < 1e-12  # every cell, DC and Nyquist too

    def test_stationary_psd_into_wdm_is_the_exact_cell_variance(self, dense):
        W, _, _ = dense
        f = np.fft.rfftfreq(N, DT)
        bins = one((N * DT / 2) * psd(f) + 0j)
        time_covariance = matrix.build_C_time(N, DT, psd)
        exact = np.diag(W @ time_covariance @ W.T).reshape(NF, NT)
        assert relative(translate_blocks(bins, FREQUENCY, WDM)[0, 0], exact) < 1e-12

    def test_frequency_to_time_is_the_diagonal_of_the_time_covariance(self):
        f = np.fft.rfftfreq(N, DT)
        bins = one((N * DT / 2) * psd(f) + 0j)
        exact = np.diag(matrix.build_C_time(N, DT, psd))
        assert relative(translate_blocks(bins, FREQUENCY, TIME)[0, 0], exact) < 1e-12

    def test_wdm_to_frequency_is_the_exact_diagonal(self, dense):
        _, W_inverse, F = dense
        rng = np.random.default_rng(3)
        cells = np.exp(0.3 * rng.normal(size=(NF, NT)))
        exact = (np.abs(F @ W_inverse) ** 2) @ cells.ravel()
        mine = translate_blocks(one(cells), WDM, FREQUENCY)[0, 0]
        assert mine.dtype == np.complex128
        assert relative(mine.real, exact) < 1e-12

    def test_a_slowly_varying_white_variance_into_wdm_is_the_exact_cell_variance(
        self, dense
    ):
        W, _, _ = dense
        variance = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(N) / N)
        exact = ((W**2) @ variance).reshape(NF, NT)
        mine = translate_blocks(one(variance), TIME, WDM)[0, 0]
        assert relative(mine, exact) < 1e-12

    def test_i_i_d_variance_keeps_the_dc_and_nyquist_rows_exact(self, dense):
        W, _, _ = dense
        variance = np.random.default_rng(7).uniform(0.5, 3.0, N)
        exact = ((W**2) @ variance).reshape(NF, NT)
        mine = translate_blocks(one(variance), TIME, WDM)[0, 0]
        edge = np.isin(_layout(NF, NT, DT).band, (0, NF))
        assert edge.any()
        assert relative(mine[edge], exact[edge]) < 1e-12

    def test_a_step_variance_into_wdm_is_smoothed(self, dense):
        W, _, _ = dense
        variance = np.where(np.arange(N) < N // 2, 1.0, 3.0)
        exact = ((W**2) @ variance).reshape(NF, NT)
        mine = translate_blocks(one(variance), TIME, WDM)[0, 0]
        # The time -> WDM mean does not conserve the trace exactly (WDM -> time does).
        assert np.median(np.abs(mine / exact - 1)) < 0.1

    @pytest.mark.parametrize("n", [255, 256])
    def test_frequency_to_time_weights_nyquist_only_for_even_n(self, n):
        rng = np.random.default_rng(8)
        bins = rng.uniform(0.5, 2.0, n // 2 + 1)
        # The full two-sided spectrum repeats bins 1.. once as the conjugate
        # mirror; DC (and Nyquist, for even N) appear once. The time variance is
        # (N dt)^-2 times the sum of all N per-bin variances.
        full = np.concatenate([bins, bins[1 : (n + 1) // 2][::-1]])
        assert full.size == n
        expected = full.sum() / (n * DT) ** 2
        source, target = Grid("frequency", n, FS), Grid("time", n, FS)
        mine = translate_blocks(one(bins + 0j), source, target)[0, 0]
        assert relative(mine, expected) < 1e-12

    def test_the_record_guard_and_the_wdm_requirement(self):
        with pytest.raises(ValueError, match="two different grids of one record"):
            translate_blocks(one(np.ones(N)), TIME, TIME)
        with pytest.raises(ValueError, match="two different grids of one record"):
            translate_blocks(one(np.ones(N)), TIME, Grid("frequency", N, 4.0))
        with pytest.raises(ValueError, match="is not a WDM grid"):
            _layout_of(TIME)

    def test_time_to_frequency_is_the_mean_power(self):
        variance = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(N) / N)
        mine = translate_blocks(one(variance), TIME, FREQUENCY)[0, 0]
        assert mine.dtype == np.complex128
        assert relative(mine.real, DT**2 * variance.sum()) < 1e-14

    def test_wdm_to_time_preserves_the_trace_and_flattens_a_stationary_input(
        self, dense
    ):
        _, W_inverse, _ = dense
        rng = np.random.default_rng(4)
        cells = np.exp(0.3 * rng.normal(size=(NF, NT)))
        exact = (W_inverse**2) @ cells.ravel()
        mine = translate_blocks(one(cells), WDM, TIME)[0, 0]
        assert relative(mine.mean(), exact.mean()) < 1e-12  # trace preserved
        assert np.median(np.abs(mine / exact - 1)) < 0.05  # column-smoothed
        stationary = np.repeat(np.exp(0.5 * rng.normal(size=(NF, 1))), NT, axis=1)
        flat = translate_blocks(one(stationary), WDM, TIME)[0, 0]
        assert relative(flat, flat.mean()) < 1e-12
        assert (
            relative(flat.mean(), ((W_inverse**2) @ stationary.ravel()).mean()) < 1e-12
        )

    def test_wdm_to_wdm_is_the_band_mean_at_the_target_columns(self):
        rng = np.random.default_rng(5)
        cells = np.exp(0.3 * rng.normal(size=(NF, NT)))
        target = grid("wdm", 16)
        mine = translate_blocks(one(cells), WDM, target)[0, 0]
        source_layout, target_layout = _layout(NF, NT, DT), _layout(16, 16, DT)
        model = 2.0 * source_layout.padded(cells)
        by_hand = np.zeros((17, 16))
        for column in range(16):
            source_column = source_layout.column_of_sample[column * 16]
            by_hand[:, column] = target_layout.bands_from_bins(
                source_layout.bins_from_bands(model[:, source_column])
            )
        assert relative(mine, target_layout.folded(by_hand) / 2.0) < 1e-12
        white = translate_blocks(one(np.full((NF, NT), 0.7)), WDM, target)[0, 0]
        assert relative(white, 0.7) < 1e-14

    def test_time_varying_white_noise_cell_variances_match_realisations(self):
        rng = np.random.default_rng(6)
        variance = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(N) / N)
        draws = rng.standard_normal((4000, N)) * np.sqrt(variance)
        empirical = domania.transform(
            draws, "time", "wdm", N=N, dt=DT, target_wdm=(NF, NT)
        ).var(axis=0)
        predicted = translate_blocks(one(variance), TIME, WDM)[0, 0]
        rms = np.sqrt(np.mean((empirical / predicted - 1) ** 2))
        assert rms < 0.05  # per-cell Monte-Carlo noise is sqrt(2 / 4000) = 0.022


def three_channel_spectrum():
    """A 3-channel frequency covariance with a complex A-E cross term."""
    f = np.fft.rfftfreq(N, DT)
    blocks = np.zeros((3, 3, NUM_BINS), dtype=np.complex128)
    for index in range(3):
        blocks[index, index] = (index + 1) * (N * DT / 2) * psd(f)
    cross = 0.3 * np.sqrt(blocks[0, 0].real * blocks[1, 1].real)
    cross = cross * np.exp(1j * 10 * f)
    cross[0], cross[-1] = cross[0].real, cross[-1].real  # real DC and Nyquist
    blocks[0, 1], blocks[1, 0] = cross, cross.conj()
    return Covariance(blocks, channel_names=("A", "E", "T"), grid=FREQUENCY)


class TestTransformOverload:
    @pytest.mark.parametrize(
        "path",
        [
            ("time",),
            ("wdm",),
            ("wdm", "time"),
            ("time", "wdm"),
            ("wdm", "frequency"),
            ("wdm", "wdm16"),
        ],
        ids=["f-t", "f-w", "f-w-t", "f-t-w", "f-w-f", "f-w-w16"],
    )
    def test_a_correlated_covariance_stays_hermitian_and_positive_definite(self, path):
        translated = three_channel_spectrum()
        for step in path:
            if step == "wdm16":
                translated = transform(translated, "wdm", num_frequency_divisions=16)
                expected = grid("wdm", 16)
            else:
                translated = transform(
                    translated,
                    step,
                    num_frequency_divisions=NF if step == "wdm" else None,
                )
                expected = grid(step)
            assert type(translated) is Covariance
            assert translated.channel_names == ("A", "E", "T")
            assert translated.grid == expected
            assert translated.blocks.dtype == (
                np.complex128 if step == "frequency" else np.float64
            )
            assert np.all(translated.A > 0)
        back = transform(translated, "frequency")
        assert back.grid == FREQUENCY
        assert back.blocks.dtype == np.complex128
        assert np.all(back.blocks[0, 1].imag == 0)  # the diagonal model keeps Re
        assert np.all(back.A > 0)

    def test_reference_data_is_accepted_and_ignored(self):
        source = three_channel_spectrum()
        reference = L1Data(
            channel_data={"A": np.zeros(N)},
            channel_names=("A",),
            sample_rate_hz=FS,
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
            start_time_gps=0.0,
        )
        with_reference = transform(source, "time", reference_data=reference)
        np.testing.assert_array_equal(
            with_reference.blocks, transform(source, "time").blocks
        )

    def test_pairs_are_carried_elementwise(self):
        data_grid = TIME
        blocks = np.zeros((2, 2, N))
        blocks[0, 0], blocks[1, 1], blocks[0, 1], blocks[1, 0] = 1.0, 4.0, 0.5, 0.5
        source = Covariance(blocks, channel_names=("A", "E"), grid=data_grid)
        cells = transform(source, "wdm", num_frequency_divisions=NF)
        np.testing.assert_allclose(cells.A, DT * 1.0, rtol=1e-14)
        np.testing.assert_allclose(cells.E, DT * 4.0, rtol=1e-14)
        np.testing.assert_allclose(cells.AE, DT * 0.5, rtol=1e-14)
        spectrum = transform(source, "frequency")
        np.testing.assert_allclose(spectrum.AE.real, N * DT**2 * 0.5, rtol=1e-14)

    def test_the_same_grid_returns_an_independent_equal_copy(self):
        source = three_channel_spectrum()
        copy = transform(source, "frequency")
        assert copy is not source
        assert copy.grid == source.grid
        np.testing.assert_array_equal(copy.blocks, source.blocks)
        assert not np.shares_memory(copy.blocks, source.blocks)
        cells = transform(source, "wdm", num_frequency_divisions=NF)
        again = transform(cells, "wdm")  # WDM to WDM without counts: its own grid
        assert again.grid == cells.grid
        np.testing.assert_array_equal(again.blocks, cells.blocks)
        same_counts = transform(cells, "wdm", num_frequency_divisions=NF)
        np.testing.assert_array_equal(same_counts.blocks, cells.blocks)

    def test_division_keywords_follow_the_data_rules(self):
        source = three_channel_spectrum()
        with pytest.raises(ValueError, match="division counts apply only"):
            transform(source, "time", num_frequency_divisions=NF)
        with pytest.raises(ValueError, match="num_frequency_divisions"):
            transform(source, "wdm")  # a WDM target from frequency needs a count
        with pytest.raises(ValueError, match="target_domain must be one of"):
            transform(source, "wavelet")
        by_columns = transform(source, "wdm", num_time_divisions=NT)
        assert by_columns.grid == WDM
