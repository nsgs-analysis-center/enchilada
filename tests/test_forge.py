"""The forge: data helpers, the skeleton block and the two workbench notebooks
run against a simulated Mojito-layout file."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    MOJITO_DT,
    MOJITO_T0,
    white_covariance,
    write_mojito_file,
)
from enchilada import Covariance, L1Data, decimate, load_mojito_l1, transform

FORGE = Path(__file__).resolve().parents[1] / "forge"
FILE_SAMPLES = 4096  # 1024 s at the fixture's 4 Hz
FACTOR = 8  # 0.5 Hz: a slipped power of dt or fs cannot hide behind unit rates
DURATION_S = 600.0  # 2400 native samples -> 300 decimated samples at 0.5 Hz
WINDOW_SAMPLES = 300
NF, NT = 8, 16  # 1024 native samples, longer than the 627-tap filter for factor 8
HOME_ROWS = 10  # FORGE_ROWS for the tests: 300 samples = 10 rows x 30 columns


def load_forge(name):
    """Import forge/<name>.py the way test_examples.py imports example scripts."""
    spec = importlib.util.spec_from_file_location(name, FORGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coloured_psd(f):
    """A red one-sided PSD at Mojito scale."""
    return 1e-36 * (1.0 + (1e-3 / f) ** 2)


def steep_psd(f):
    """A red PSD whose wall sits inside the test grid's first bands."""
    return 1e-36 * (1.0 + (0.1 / f) ** 2)


@pytest.fixture
def simulated_file(tmp_path, monkeypatch):
    """A tiny Mojito-layout file standing in for the simulation's output, with
    the forge's environment variables pointing at it."""
    path = tmp_path / "simulated.h5"
    write_mojito_file(path, n=FILE_SAMPLES)
    monkeypatch.setenv("ENCHILADA_MOJITO_L1", str(path))
    monkeypatch.setenv("FORGE_DURATION_S", str(DURATION_S))
    monkeypatch.setenv("FORGE_FACTOR", str(FACTOR))
    monkeypatch.setenv("FORGE_ROWS", str(HOME_ROWS))
    return path


@pytest.fixture
def forge_data():
    return load_forge("data")


def wdm_options():
    return dict(domain="wdm", num_frequency_divisions=NF, num_time_divisions=NT)


def load_grid(forge_data, path, domain):
    """The simulated window on `domain`, all three channels."""
    options = (
        wdm_options() if domain == "wdm" else dict(domain=domain, duration_s=DURATION_S)
    )
    return forge_data.load_window(path, factor=FACTOR, **options)


# ---- load_window -----------------------------------------------------------


def test_time_window_is_decimated_from_the_file(simulated_file, forge_data):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S
    )
    assert window.data_domain == "time"
    assert window.channel_names == ("A", "E", "T")
    assert window.num_time_samples == WINDOW_SAMPLES
    assert window.sample_rate_hz == 1.0 / (MOJITO_DT * FACTOR)
    assert window.start_time_gps == MOJITO_T0
    assert window.orbit_ephemeris is not None


def test_frequency_window_is_the_time_window_transformed(simulated_file, forge_data):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S, channels=("A", "E")
    )
    spectrum = forge_data.load_window(
        simulated_file,
        factor=FACTOR,
        duration_s=DURATION_S,
        channels=("A", "E"),
        domain="frequency",
    )
    expected = transform(window, "frequency")
    assert spectrum.data_domain == "frequency"
    assert spectrum.grid == expected.grid
    assert spectrum.start_time_gps == MOJITO_T0
    for name in ("A", "E"):
        np.testing.assert_array_equal(
            spectrum.channel_data[name], expected.channel_data[name]
        )


def test_wdm_window_derives_its_duration_and_lands_on_the_grid(
    simulated_file, forge_data
):
    window = forge_data.load_window(simulated_file, factor=FACTOR, **wdm_options())
    assert window.data_domain == "wdm"
    assert window.wdm_grid.array_shape == (NF, NT)
    assert window.num_time_samples == NF * NT
    assert window.sample_rate_hz == 1.0 / (MOJITO_DT * FACTOR)
    assert window.start_time_gps == MOJITO_T0
    # Exactly factor * Nf * Nt native samples went in: the same pipeline by hand.
    native = load_mojito_l1(simulated_file, duration_s=FACTOR * NF * NT * MOJITO_DT)
    assert native.num_time_samples == FACTOR * NF * NT
    expected = transform(
        decimate(native, FACTOR),
        "wdm",
        num_frequency_divisions=NF,
        num_time_divisions=NT,
    )
    for name in ("A", "E", "T"):
        np.testing.assert_array_equal(
            window.channel_data[name], expected.channel_data[name]
        )


def test_wdm_window_reads_only_the_requested_samples(tmp_path, forge_data):
    # A flagged sample before the window must not matter, so the native
    # sampling interval comes from the file's attributes, not a probe load.
    flagged = tmp_path / "flagged.h5"
    write_mojito_file(flagged, n=FILE_SAMPLES, flag_index=10)
    window = forge_data.load_window(
        flagged, factor=FACTOR, start_offset_s=100.0, **wdm_options()
    )
    assert window.num_time_samples == NF * NT
    assert window.start_time_gps == MOJITO_T0 + 100.0
    # A file shorter than a minute still serves a window that fits it.
    short = tmp_path / "short.h5"
    write_mojito_file(short, n=200)
    window = forge_data.load_window(
        short, factor=2, domain="wdm", num_frequency_divisions=8, num_time_divisions=10
    )
    assert window.wdm_grid.array_shape == (8, 10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(wdm_options(), duration_s=100.0), "leave duration_s"),
        (
            dict(domain="wdm", num_frequency_divisions=NF),
            "num_frequency_divisions and num_time_divisions",
        ),
        (
            dict(domain="wdm", num_frequency_divisions=NF, num_time_divisions=15),
            "positive even integer",
        ),
        (
            dict(domain="wdm", num_frequency_divisions=7, num_time_divisions=NT),
            "positive even integer",
        ),
        (dict(domain="time"), "needs duration_s"),
        (dict(domain="frequency"), "needs duration_s"),
        (
            dict(domain="time", duration_s=100.0, num_frequency_divisions=NF),
            "only to domain='wdm'",
        ),
        (dict(domain="laplace", duration_s=100.0), "domain must be"),
    ],
)
def test_load_window_refuses_inconsistent_requests(
    tmp_path, forge_data, kwargs, message
):
    # Every refusal comes before any file I/O, so a missing path serves.
    with pytest.raises(ValueError, match=message):
        forge_data.load_window(tmp_path / "missing.h5", factor=FACTOR, **kwargs)


def test_load_window_validates_the_factor_before_reading(tmp_path, forge_data):
    missing = tmp_path / "missing.h5"
    with pytest.raises(ValueError, match="at least 2"):
        forge_data.load_window(missing, factor=1, duration_s=DURATION_S)
    for factor in (2.0, True):
        with pytest.raises(TypeError, match="integer"):
            forge_data.load_window(missing, factor=factor, duration_s=DURATION_S)


# ---- white_psd and analytic_covariance ---------------------------------------


def test_white_psd_is_flat_and_refuses_bad_levels(forge_data):
    psd = forge_data.white_psd(3.0)
    np.testing.assert_array_equal(psd(np.array([0.0, 1e-3, 0.5])), 3.0)
    assert psd(np.float64(0.1)).shape == ()
    for level in (0.0, -1.0, np.inf, np.nan, "abc", np.array([1.0, 2.0])):
        with pytest.raises(ValueError, match="finite and positive"):
            forge_data.white_psd(level)


def test_frequency_bins_carry_half_tobs_times_the_psd(simulated_file, forge_data):
    spectrum = forge_data.load_window(
        simulated_file,
        factor=FACTOR,
        duration_s=DURATION_S,
        channels=("A", "E"),
        domain="frequency",
    )
    covariance = forge_data.analytic_covariance(spectrum, coloured_psd)
    bins = np.arange(spectrum.N // 2 + 1) * spectrum.df
    expected = 0.5 * spectrum.Tobs * coloured_psd(np.maximum(bins, spectrum.df))
    for name in ("A", "E"):
        np.testing.assert_allclose(covariance.pair(name, name), expected, rtol=1e-12)
    # The DC bin takes the first bin's value instead of psd(0).
    assert covariance.A[0] == covariance.A[1]
    np.testing.assert_array_equal(covariance.AE, 0.0)


def test_wdm_cells_average_the_psd_across_each_band(simulated_file, forge_data):
    window = load_grid(forge_data, simulated_file, "wdm")
    cells = forge_data.analytic_covariance(window, steep_psd).A
    band_spacing_hz = window.fs / (2 * NF)
    # Stationary noise: one value per band. Row 0 interleaves DC and Nyquist.
    for band in range(1, NF):
        assert np.ptp(cells[band]) == 0.0
    assert np.ptp(cells[0, 0::2]) == 0.0 and np.ptp(cells[0, 1::2]) == 0.0

    # Each band's value is an average of S / 2 over that band's support, so it
    # lies strictly between the PSD's extremes there and falls with the band.
    def bounds(low_hz, high_hz):
        grid = np.linspace(max(low_hz, window.df), high_hz, 1001)
        return 0.5 * steep_psd(grid).min(), 0.5 * steep_psd(grid).max()

    values = [cells[0, 0]] + [cells[band, 0] for band in range(1, NF)] + [cells[0, 1]]
    for band, value in enumerate(values):
        low, high = bounds((band - 1) * band_spacing_hz, (band + 1) * band_spacing_hz)
        assert low < value < high, (band, value, low, high)
    assert all(a > b for a, b in zip(values[:-1], values[1:], strict=True))

    # A PSD linear in f averages to its band-centre value on interior bands.
    def linear_psd(f):
        return 1e-36 * (1.0 + 2.0 * f)

    linear = forge_data.analytic_covariance(window, linear_psd).A
    for band in range(1, NF):
        np.testing.assert_allclose(
            linear[band], 0.5 * linear_psd(band * band_spacing_hz), rtol=1e-9
        )
    # A PSD that vanishes at every band centre averages to half its peak over
    # a band (the Meyer window's partition of unity), so a band-centre
    # evaluation would give zero here.
    notched = forge_data.analytic_covariance(
        window, lambda f: 1e-36 * np.sin(np.pi * f / band_spacing_hz) ** 2
    ).A
    np.testing.assert_allclose(notched[1:], 0.25e-36, rtol=1e-12)
    # A PSD written for one-dimensional input works on the WDM path too.
    flat = forge_data.analytic_covariance(window, lambda f: np.full(len(f), 2e-36))
    np.testing.assert_allclose(flat.A, 1e-36, rtol=1e-12)


@pytest.mark.parametrize("domain", ["time", "frequency", "wdm"])
def test_white_psd_matches_the_white_covariance_convention(
    simulated_file, forge_data, domain
):
    window = load_grid(forge_data, simulated_file, domain)
    sigma2 = 1.3e-36
    covariance = forge_data.analytic_covariance(
        window, forge_data.white_psd(2 * sigma2 / window.fs)
    )
    reference = white_covariance(window, sigma2)
    for name in window.channel_names:
        np.testing.assert_allclose(
            covariance.pair(name, name), reference.pair(name, name), rtol=1e-12
        )


def test_a_translated_frequency_covariance_is_the_home_grid_covariance(
    simulated_file, forge_data
):
    """The coloured PSD's frequency covariance, translated onto the home grid,
    is what analytic_covariance builds on the home-grid data directly."""
    spectrum = load_grid(forge_data, simulated_file, "frequency")
    cells = transform(spectrum, "wdm", num_frequency_divisions=HOME_ROWS)
    translated = transform(
        forge_data.analytic_covariance(spectrum, coloured_psd),
        "wdm",
        num_frequency_divisions=HOME_ROWS,
    )
    direct = forge_data.analytic_covariance(cells, coloured_psd)
    assert translated.grid == direct.grid == cells.grid
    np.testing.assert_allclose(translated.blocks, direct.blocks, rtol=1e-12)


@pytest.mark.parametrize("domain", ["time", "frequency", "wdm"])
def test_a_constant_psd_broadcasts_like_white_psd(simulated_file, forge_data, domain):
    window = load_grid(forge_data, simulated_file, domain)
    constant = forge_data.analytic_covariance(window, lambda f: 2e-36)
    white = forge_data.analytic_covariance(window, forge_data.white_psd(2e-36))
    for name in window.channel_names:
        np.testing.assert_array_equal(constant.pair(name, name), white.pair(name, name))


def test_psd_values_must_be_finite_and_positive(simulated_file, forge_data):
    spectrum = load_grid(forge_data, simulated_file, "frequency")
    with pytest.raises(ValueError, match="not finite"):
        forge_data.analytic_covariance(spectrum, lambda f: np.nan)
    with pytest.raises(ValueError, match="non-positive"):
        forge_data.analytic_covariance(spectrum, lambda f: 0.0)
    with pytest.raises(ValueError, match="returned shape"):
        forge_data.analytic_covariance(spectrum, lambda f: np.ones(3))
    with pytest.raises(ValueError, match="negative"):
        forge_data.analytic_covariance(spectrum, lambda f: -1e-36)
    window = load_grid(forge_data, simulated_file, "wdm")
    with pytest.raises(ValueError, match="non-positive"):
        forge_data.analytic_covariance(window, lambda f: 0.0)


def test_coloured_psd_is_refused_on_the_time_grid(simulated_file, forge_data):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S
    )
    with pytest.raises(ValueError, match="frequency or WDM grid"):
        forge_data.analytic_covariance(window, coloured_psd)
    # A tilt of one part in a million across the band (f up to 0.5 Hz) is
    # still coloured.
    with pytest.raises(ValueError, match="frequency or WDM grid"):
        forge_data.analytic_covariance(window, lambda f: 1e-36 * (1.0 + 2e-6 * f))


def test_per_channel_psd_mapping(simulated_file, forge_data):
    spectrum = forge_data.load_window(
        simulated_file,
        factor=FACTOR,
        duration_s=DURATION_S,
        channels=("A", "E"),
        domain="frequency",
    )
    covariance = forge_data.analytic_covariance(
        spectrum, {"A": forge_data.white_psd(2e-36), "E": forge_data.white_psd(4e-36)}
    )
    np.testing.assert_allclose(covariance.E, 2 * covariance.A, rtol=1e-12)
    with pytest.raises(ValueError, match=r"no entry for channels \['E'\]"):
        forge_data.analytic_covariance(spectrum, {"A": forge_data.white_psd(2e-36)})


# ---- ForgeBlock --------------------------------------------------------------


@pytest.fixture
def forge_block():
    return load_forge("block")


def tiny_spectrum(num_time_samples, channel_data):
    """A frequency-grid L1Data at 1 Hz with hand-computable bins."""
    return L1Data(
        channel_data,
        sample_rate_hz=1.0,
        num_time_samples=num_time_samples,
        channel_names=tuple(channel_data),
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        data_domain="frequency",
    )


def simulated_spectrum(forge_data, path):
    return forge_data.load_window(
        path,
        factor=FACTOR,
        duration_s=DURATION_S,
        channels=("A", "E"),
        domain="frequency",
    )


def test_rfft_weights_count_each_bin_of_a_real_signal(forge_block):
    np.testing.assert_array_equal(forge_block.rfft_weights(4), [1.0, 2.0, 1.0])
    np.testing.assert_array_equal(forge_block.rfft_weights(5), [1.0, 2.0, 2.0])


def test_forge_block_is_a_block_with_validated_settings(forge_block):
    from enchilada import Block

    block = forge_block.ForgeBlock()
    assert isinstance(block, Block)
    assert block.name == "forge"
    assert block.role == "signal"
    assert (block.frequency_hz, block.prior_std, block.start_amplitude) == (
        1e-3,
        1e-17,
        None,
    )
    assert block.state.num_sample_calls == 0
    for settings in (
        dict(name=""),
        dict(frequency_hz=0.0),
        dict(frequency_hz=np.inf),
        dict(prior_std=-1.0),
        dict(prior_std=np.nan),
        dict(start_amplitude=np.nan),
    ):
        with pytest.raises(ValueError):
            forge_block.ForgeBlock(**settings)


@pytest.mark.parametrize(
    ("num_time_samples", "frequency_hz", "basis", "expected_precision"),
    [
        # sin(2 pi t / 4) at 1 Hz: bin 1 is -2j. Precision = prior 1/0.25 +
        # likelihood 2 * |-2j|^2 / 2 = 4; in the time domain sum(s^2) / sigma^2
        # = 2 / 0.5 with sigma^2 = C fs^2 / N = 2 / 4.
        (4, 0.25, [0.0, -2.0j, 0.0], 8.0),
        # sin(2 pi 0.4 t), 5 samples: the last bin, -2.5j, is interior and
        # counts twice; likelihood 2 * 6.25 / 2 = 6.25, in the time domain
        # 2.5 / 0.4 with sigma^2 = 2 / 5.
        (5, 0.4, [0.0, 0.0, -2.5j], 10.25),
    ],
)
def test_sample_draws_the_exact_conjugate_posterior(
    forge_block, num_time_samples, frequency_hz, basis, expected_precision
):
    basis = np.asarray(basis)
    data = tiny_spectrum(num_time_samples, {"A": 4.0 * basis})
    noise = Covariance.like(data, {"A": np.full(basis.size, 2.0)})
    block = forge_block.ForgeBlock(frequency_hz=frequency_hz, prior_std=0.5)
    start = block.initial_sample(data, noise, rng=np.random.default_rng(0))
    np.testing.assert_allclose(
        start.template["A"], start.parameters["amplitude"] * basis, atol=1e-12
    )
    # weighted data = 2 * Re(conj(h) 4h) / 2 = 4 |h|^2 at the one nonzero bin.
    expected_mean = 4.0 * float(np.sum(np.abs(basis) ** 2)) / expected_precision
    expected_std = expected_precision**-0.5
    moved = block.sample(data, noise, start, rng=np.random.default_rng(1))
    assert block.state.num_sample_calls == 1
    np.testing.assert_allclose(
        block.state.last_posterior_mean, expected_mean, rtol=1e-12
    )
    np.testing.assert_allclose(block.state.last_posterior_std, expected_std, rtol=1e-12)
    np.testing.assert_allclose(
        moved.template["A"], moved.parameters["amplitude"] * basis, atol=1e-12
    )
    draws = np.array(
        [
            block.sample(
                data, noise, start, rng=np.random.default_rng(seed)
            ).parameters["amplitude"]
            for seed in range(2000)
        ]
    )
    np.testing.assert_allclose(
        draws.mean(), expected_mean, atol=4 * expected_std / np.sqrt(2000), rtol=0
    )
    np.testing.assert_allclose(draws.std(), expected_std, rtol=0.1)


def test_initial_sample_renders_a_configured_start_or_a_prior_draw(forge_block):
    basis = np.array([0.0, -2.0j, 0.0])
    data = tiny_spectrum(4, {"A": 0.0 * basis, "E": 0.0 * basis})
    configured = forge_block.ForgeBlock(frequency_hz=0.25, start_amplitude=1.5)
    start = configured.initial_sample(data, None, rng=np.random.default_rng(0))
    assert start.parameters == {"amplitude": 1.5}
    for name in ("A", "E"):
        np.testing.assert_allclose(start.template[name], 1.5 * basis, atol=1e-12)
    drawn = forge_block.ForgeBlock(frequency_hz=0.25, prior_std=0.5)
    draws = [
        drawn.initial_sample(data, None, rng=np.random.default_rng(seed)).parameters[
            "amplitude"
        ]
        for seed in (0, 0, 1)
    ]
    assert draws[0] == draws[1] != draws[2]
    assert draws[0] == np.random.default_rng(0).normal(0.0, 0.5)
    assert drawn.state.num_sample_calls == 0


def test_block_refuses_inputs_its_likelihood_cannot_use(
    simulated_file, forge_data, forge_block
):
    spectrum = simulated_spectrum(forge_data, simulated_file)
    window = transform(spectrum, "time")
    noise = forge_data.analytic_covariance(spectrum, forge_data.white_psd(2e-36))
    block = forge_block.ForgeBlock()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="data_domain='frequency'"):
        block.initial_sample(window, None, rng=rng)
    with pytest.raises(ValueError, match="Nyquist"):
        forge_block.ForgeBlock(frequency_hz=0.5).initial_sample(
            spectrum, noise, rng=rng
        )
    start = block.initial_sample(spectrum, None, rng=rng)
    with pytest.raises(ValueError, match="needs a noise covariance"):
        block.sample(spectrum, None, start, rng=rng)
    with pytest.raises(ValueError, match="grid"):
        block.sample(spectrum, white_covariance(window, 1e-36), start, rng=rng)
    correlated = Covariance.like(
        spectrum, {"A": noise.A, "E": noise.E, "AE": 0.5 * noise.A}
    )
    with pytest.raises(ValueError, match="independent channels"):
        block.sample(spectrum, correlated, start, rng=rng)
    assert block.state.num_sample_calls == 0


def test_forge_block_passes_check_block_on_the_simulated_window(
    simulated_file, forge_data, forge_block
):
    from enchilada import FixedNoiseBlock
    from enchilada.testing import check_block

    spectrum = simulated_spectrum(forge_data, simulated_file)
    noise = forge_data.analytic_covariance(spectrum, forge_data.white_psd(2e-36))
    block = forge_block.ForgeBlock(frequency_hz=0.05)
    check_block(
        block,
        spectrum,
        num_cycles=3,
        noise_blocks=(FixedNoiseBlock("noise", noise),),
        random_seed=1,
        num_frequency_divisions=HOME_ROWS,
    )
    assert block.state.num_sample_calls == 3
    assert np.isfinite(block.state.last_posterior_std)


def test_inner_product_matches_the_time_domain_sum(
    simulated_file, forge_data, forge_block
):
    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S, channels=("A", "E")
    )
    spectrum = transform(window, "frequency")
    sigma2 = 1.7e-36
    noise = white_covariance(spectrum, sigma2)
    block = forge_block.ForgeBlock(frequency_hz=0.05)
    unit = {name: block._basis(spectrum) for name in ("A", "E")}
    times_s = window.start_time_gps + np.arange(window.N) / window.fs
    sinusoid = np.sin(2 * np.pi * 0.05 * times_s)
    # Parseval: the Whittle inner product with white noise is the time-domain
    # sum over samples divided by the per-sample variance.
    expected = (
        sum(np.sum(sinusoid * window.channel_data[name]) for name in ("A", "E"))
        / sigma2
    )
    np.testing.assert_allclose(
        forge_block.inner_product(unit, spectrum.channel_data, spectrum, noise),
        expected,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        forge_block.inner_product(unit, unit, spectrum, noise),
        2 * np.sum(sinusoid**2) / sigma2,
        rtol=1e-10,
    )
    # The data's own norm exercises the DC and Nyquist weights, which the
    # sinusoid (a whole number of cycles in the window) does not touch.
    data = spectrum.channel_data
    np.testing.assert_allclose(
        forge_block.inner_product(data, data, spectrum, noise),
        sum(np.sum(window.channel_data[name] ** 2) for name in ("A", "E")) / sigma2,
        rtol=1e-10,
    )


def test_forge_block_recovers_an_injected_amplitude(
    simulated_file, forge_data, forge_block
):
    from enchilada import FixedNoiseBlock, Wheel

    window = forge_data.load_window(
        simulated_file, factor=FACTOR, duration_s=DURATION_S, channels=("A", "E")
    )
    level = 2 * np.mean([window.channel_data[n].var() for n in ("A", "E")]) / window.fs
    spectrum = transform(window, "frequency")
    # 0.05 Hz is bin 30 of the 600 s window, inside the decimation passband.
    block = forge_block.ForgeBlock(frequency_hz=0.05, prior_std=1e-17)
    true_amplitude = 1e-18
    basis = block._basis(spectrum)
    observed = replace(
        spectrum,
        channel_data={
            name: values + true_amplitude * basis
            for name, values in spectrum.channel_data.items()
        },
    )
    noise = forge_data.analytic_covariance(observed, forge_data.white_psd(level))
    wheel = Wheel(observed, num_frequency_divisions=HOME_ROWS, random_seed=3)
    wheel.add(FixedNoiseBlock("noise", noise))
    wheel.add(block, data_domain="frequency")
    chain = []
    wheel.run(
        300,
        on_cycle_complete=lambda cycle, campaign: chain.append(
            campaign.ledger["forge"].parameters["amplitude"]
        ),
    )
    samples = np.asarray(chain[50:])
    posterior_std = block.state.last_posterior_std
    # The data, not the prior, decide: the prior's precision is negligible.
    assert block.prior_std**-2 < 1e-3 * posterior_std**-2
    assert posterior_std < 0.1 * true_amplitude
    # One noise realisation pulls the posterior mean; the chain follows it.
    assert abs(samples.mean() - true_amplitude) < 4 * posterior_std
    recorded_mean = block.state.last_posterior_mean
    assert abs(samples.mean() - recorded_mean) < 4 * posterior_std / np.sqrt(
        samples.size
    )
    np.testing.assert_allclose(samples.std(), posterior_std, rtol=0.25)


# ---- the notebooks -----------------------------------------------------------


def notebook_cells(name):
    notebook = json.loads((FORGE / name).read_text())
    assert notebook["nbformat"] == 4
    return notebook["cells"]


def cell_code(cell):
    """A code cell's source without IPython magics (the autoreload lines)."""
    return "\n".join(
        line
        for line in "".join(cell["source"]).splitlines()
        if not line.startswith("%")
    )


@pytest.mark.parametrize("name", ["block.ipynb", "wheel.ipynb"])
def test_forge_notebook_cells_compile(name):
    cells = notebook_cells(name)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    assert [cell["id"] for cell in cells] == [
        f"forge-{name[:-6]}-{index:02d}" for index in range(len(cells))
    ]
    for index, cell in enumerate(code_cells, 1):
        compile(cell_code(cell), f"{name}:cell-{index}", "exec")


@pytest.fixture(scope="session")
def matplotlib_config_dir(tmp_path_factory):
    """One font cache for every notebook run, not one per test."""
    return tmp_path_factory.mktemp("matplotlib")


def run_notebook(name, config_dir, tmp_dir, *, with_matplotlib):
    """Run a notebook's code cells in one process, each compiled under its cell
    id so a traceback names the cell; with matplotlib blocked when asked."""
    if with_matplotlib:
        pytest.importorskip("matplotlib")
    code = "" if with_matplotlib else "import sys\nsys.modules['matplotlib'] = None\n"
    code += "".join(
        f"exec(compile({cell_code(cell)!r}, {cell['id']!r}, 'exec'))\n"
        for cell in notebook_cells(name)
        if cell["cell_type"] == "code"
    )
    code += (
        "if plt is not None:\n"
        "    for number in plt.get_fignums():\n"
        "        plt.figure(number).canvas.draw()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=FORGE,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(config_dir),
            "TMPDIR": str(tmp_dir),  # campaign files stay under pytest's tmp_path
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


WITH_MATPLOTLIB = pytest.mark.parametrize(
    "with_matplotlib", [True, False], ids=["matplotlib", "no-matplotlib"]
)


@WITH_MATPLOTLIB
def test_block_notebook_runs_on_the_simulated_file(
    simulated_file, matplotlib_config_dir, tmp_path, with_matplotlib
):
    out = run_notebook(
        "block.ipynb", matplotlib_config_dir, tmp_path, with_matplotlib=with_matplotlib
    )
    assert "frequency grid, channels ('A', 'E')" in out, out
    assert "check_block passed" in out, out


@WITH_MATPLOTLIB
def test_wheel_notebook_runs_on_the_simulated_file(
    simulated_file, matplotlib_config_dir, tmp_path, with_matplotlib
):
    out = run_notebook(
        "wheel.ipynb", matplotlib_config_dir, tmp_path, with_matplotlib=with_matplotlib
    )
    assert "posterior amplitude:" in out, out
    assert "residual rms after the run" in out, out


# ---- the real file -------------------------------------------------------------

REAL_FILE = os.environ.get("ENCHILADA_MOJITO_L1")


@pytest.mark.skipif(
    not REAL_FILE, reason="set ENCHILADA_MOJITO_L1 to a Mojito L1 file (4 Hz)"
)
def test_forge_loads_an_hour_of_the_real_file(forge_data, forge_block):
    residual = forge_data.load_window(
        REAL_FILE, factor=10, duration_s=3600.0, domain="frequency", channels=("A", "E")
    )
    assert residual.num_time_samples == 1440
    assert residual.sample_rate_hz == 0.4
    covariance = forge_data.analytic_covariance(residual, forge_data.white_psd(2e-36))
    start = forge_block.ForgeBlock().initial_sample(
        residual, covariance, rng=np.random.default_rng(0)
    )
    for name in ("A", "E"):
        assert start.template[name].shape == residual.channel_data[name].shape
