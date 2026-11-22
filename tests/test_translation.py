"""Object conversion preserves physical grids, numerical values and owned state."""

from dataclasses import replace

import domania
import numpy as np
import pytest

from conftest import white_covariance
from enchilada import BlockState, Covariance, L1Data, WDMGrid
from enchilada.translation import transform


def observations(num_time_samples=64, sample_interval_s=0.25):
    rng = np.random.default_rng(17)
    return L1Data(
        channel_data={name: rng.normal(size=num_time_samples) for name in ("A", "E")},
        channel_names=("A", "E"),
        sample_rate_hz=1 / sample_interval_s,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=100.0,
    )


@pytest.mark.parametrize("num_time_samples", [1, 63, 64])
def test_fourier_round_trip_uses_continuous_coefficients(num_time_samples):
    data = observations(num_time_samples)
    spectrum = transform(data, "frequency")
    restored = transform(spectrum, "time")
    for name in data.channel_names:
        np.testing.assert_allclose(
            spectrum.channel_data[name],
            data.sample_interval_s * np.fft.rfft(data.channel_data[name]),
        )
        np.testing.assert_allclose(
            restored.channel_data[name], data.channel_data[name], atol=1e-14
        )
    assert restored.start_time_gps == data.start_time_gps
    assert restored.num_time_samples == num_time_samples
    assert spectrum.sample_rate_hz == data.sample_rate_hz
    assert spectrum.channel_names == data.channel_names
    assert spectrum.physical_observable == data.physical_observable


@pytest.mark.parametrize("conversion", ["transform", "to_time"])
@pytest.mark.parametrize(("amplitude", "sample_rate_hz"), [(1e-42, 1e-4), (1e38, 1e4)])
def test_frequency_normalization_preserves_complex64_dynamic_range(
    conversion, amplitude, sample_rate_hz
):
    spectrum = np.array([0, amplitude, 0], dtype=np.complex64)
    source = L1Data(
        channel_data={"A": spectrum},
        channel_names=("A",),
        sample_rate_hz=sample_rate_hz,
        num_time_samples=4,
        tdi_generation="2.0",
        physical_observable="strain",
        data_domain="frequency",
    )
    # This bin is a cosine: four samples are [a/(2*dt), 0, -a/(2*dt), 0].
    peak = float(spectrum[1].real) * sample_rate_hz / 2
    expected = np.array([peak, 0, -peak, 0])
    with np.errstate(over="raise", invalid="raise"):
        restored = (
            transform(source, "time") if conversion == "transform" else source.to_time()
        )
    np.testing.assert_allclose(restored.channel_data["A"], expected, rtol=1e-12, atol=0)


def test_same_domain_transform_is_an_independent_snapshot():
    data = observations()
    copied = transform(data, "time")
    copied.channel_data["A"][:] = 0
    assert np.any(data.channel_data["A"])


def test_result_conversion_preserves_state_without_aliasing():
    data = observations()
    original = data.block_state(
        data.channel_data,
        parameters={"amplitude": np.array([2.0])},
    )
    converted = transform(original, "frequency", reference_data=data)
    spectrum = transform(data, "frequency")
    for name in data.channel_names:
        np.testing.assert_allclose(
            converted.template[name], spectrum.channel_data[name]
        )
    converted.parameters["amplitude"][0] = 7
    assert original.parameters["amplitude"][0] == 2


def test_noise_only_state_translation_carries_its_covariance_without_a_reference():
    data = observations()
    covariance = white_covariance(data, 1.0)
    result = BlockState(covariance=covariance, parameters={"variance": 1.0})
    converted = transform(result, "frequency")
    assert converted.template is None
    assert converted.parameters == {"variance": 1.0}
    assert converted.covariance is not covariance
    assert converted.covariance.grid == data.grid  # a covariance keeps its grid
    np.testing.assert_array_equal(converted.covariance.A, 1.0)
    assert not np.shares_memory(converted.covariance.blocks, covariance.blocks)
    assert not converted.covariance.blocks.flags.writeable
    state = BlockState(parameters={"amplitude": 1.0})
    copied = transform(state, "frequency")
    assert copied is not state
    assert copied.parameters == state.parameters


def test_block_state_with_template_and_covariance_translates_only_the_template():
    data = observations()
    covariance = white_covariance(data, 2.0)
    original = data.block_state(data.channel_data, covariance=covariance)
    converted = transform(original, "frequency", reference_data=data)
    spectrum = transform(data, "frequency")
    for name in data.channel_names:
        np.testing.assert_allclose(
            converted.template[name], spectrum.channel_data[name]
        )
    assert converted.covariance.grid == covariance.grid
    np.testing.assert_array_equal(converted.covariance.blocks, covariance.blocks)
    assert not np.shares_memory(converted.covariance.blocks, covariance.blocks)


def test_transform_translates_a_covariance_and_refuses_other_types():
    data = observations()
    covariance = white_covariance(data, 2.0)
    spectrum = transform(covariance, "frequency")
    assert type(spectrum) is Covariance
    assert spectrum.grid == transform(data, "frequency").grid
    np.testing.assert_allclose(
        spectrum.A, white_covariance(transform(data, "frequency"), 2.0).A
    )
    cells = transform(covariance, "wdm", num_frequency_divisions=8)
    assert cells.grid == transform(data, "wdm", num_frequency_divisions=8).grid
    np.testing.assert_allclose(cells.A, 2.0 / data.fs, rtol=1e-12)
    with pytest.raises(ValueError, match="division counts apply only"):
        transform(covariance, "frequency", num_frequency_divisions=8)
    with pytest.raises(TypeError, match="expects L1Data, BlockState or Covariance"):
        transform({"A": np.zeros(4)}, "time")


def test_conversion_rejects_missing_or_mismatched_reference():
    data = observations()
    result = data.block_state(data.channel_data)
    with pytest.raises(ValueError, match="reference_data"):
        transform(result, "frequency")
    with pytest.raises(ValueError, match="length|shape"):
        transform(result, "frequency", reference_data=observations(32))


@pytest.mark.parametrize("target", ["", "wavelet", "Frequency"])
def test_unknown_domains_are_rejected(target):
    with pytest.raises(ValueError, match="target_domain"):
        transform(observations(), target)


def test_conversion_checks_nonfinite_data_and_irrelevant_grid_options():
    data = observations()
    with pytest.raises(ValueError, match="wdm"):
        transform(data, "frequency", num_frequency_divisions=8)
    with pytest.raises(TypeError, match="L1Data|BlockState|covariance"):
        transform(np.ones(8), "frequency")
    data.channel_data["A"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        transform(data, "frequency")


def test_transform_refuses_non_finite_output():
    # Finite input passes the input check; dt * rfft overflows to inf.
    data = observations(sample_interval_s=1e3)
    assert data.sample_rate_hz == 1e-3
    data = replace(
        data,
        channel_data={name: np.full(64, 1e308) for name in data.channel_names},
    )
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="non-finite coefficients"),
    ):
        transform(data, "frequency")


@pytest.mark.parametrize("source_domain", ["time", "frequency"])
@pytest.mark.parametrize("sample_interval_s", [0.125, 10.0])
def test_domania_wdm_coefficients_and_all_round_trip_routes(
    source_domain, sample_interval_s
):
    data = observations(sample_interval_s=sample_interval_s)
    source = transform(data, source_domain)
    converted = transform(source, "wdm", num_frequency_divisions=8)
    assert converted.wdm_grid == WDMGrid(
        num_frequency_divisions=8, num_time_divisions=8
    )
    stack = np.stack([data.channel_data[name] for name in data.channel_names])
    expected_stack = domania.transform(
        stack, "time", "wdm", N=64, dt=sample_interval_s, target_wdm=(8, 8)
    )
    for index, name in enumerate(data.channel_names):
        expected = domania.wilson.forward_time(
            data.channel_data[name], 8, 8, sample_interval_s
        )
        np.testing.assert_array_equal(expected_stack[index], expected)
        assert converted.channel_data[name].shape == (8, 8)
        assert converted.channel_data[name].dtype == np.float64
        if source_domain == "time":
            np.testing.assert_array_equal(converted.channel_data[name], expected)
        else:
            # (dt * rfft) / dt is exact only for power-of-two dt.
            np.testing.assert_allclose(
                converted.channel_data[name], expected, rtol=0, atol=1e-13
            )
    for target in ("time", "frequency"):
        restored = transform(converted, target)
        expected = transform(data, target)
        assert restored.wdm_grid is None
        for name in data.channel_names:
            np.testing.assert_allclose(
                restored.channel_data[name], expected.channel_data[name], atol=1e-12
            )
    np.testing.assert_allclose(
        converted.to_time().channel_data["A"], data.channel_data["A"], atol=1e-14
    )
    np.testing.assert_allclose(
        converted.to_frequency().channel_data["A"],
        transform(data, "frequency").channel_data["A"],
        atol=1e-12,
    )


def test_wdm_division_selectors_regrid_without_resampling():
    data = observations()
    first = transform(data, "wdm", num_frequency_divisions=8)
    same = transform(data, "wdm", num_time_divisions=8)
    regridded = transform(first, "wdm", num_time_divisions=4)
    direct = transform(data, "wdm", num_frequency_divisions=16)
    assert regridded.channel_data["A"].shape == (16, 4)
    assert regridded.wdm_grid == WDMGrid(
        num_frequency_divisions=16, num_time_divisions=4
    )
    assert regridded.num_time_samples == data.num_time_samples
    np.testing.assert_allclose(first.channel_data["A"], same.channel_data["A"])
    np.testing.assert_allclose(regridded.channel_data["A"], direct.channel_data["A"])
    copied = transform(first, "wdm")
    assert copied.wdm_grid == first.wdm_grid
    assert replace(copied).wdm_grid == first.wdm_grid
    assert not np.shares_memory(copied.channel_data["A"], first.channel_data["A"])
    rebuilt = replace(first, channel_data=regridded.channel_data)
    assert rebuilt.wdm_grid == regridded.wdm_grid
    for target in ("time", "frequency"):
        restored = transform(rebuilt, target)
        expected = transform(data, target)
        assert restored.wdm_grid is None
        for name in data.channel_names:
            np.testing.assert_allclose(
                restored.channel_data[name], expected.channel_data[name], atol=1e-12
            )


def test_first_wdm_conversion_requires_a_valid_division_count():
    with pytest.raises(ValueError, match="division"):
        transform(observations(), "wdm")
    with pytest.raises(ValueError, match="divis|even"):
        transform(observations(63), "wdm", num_time_divisions=8)


def test_joint_wdm_result_converts_signal_with_a_wdm_reference():
    data = observations()
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    reference = replace(
        data,
        data_domain="wdm",
        channel_data={name: np.zeros(grid.array_shape) for name in data.channel_names},
    )
    result = reference.block_state(reference.channel_data)
    converted = transform(result, "wdm", reference_data=reference)
    assert not np.shares_memory(converted.template["A"], result.template["A"])


def test_wdm_conversions_reach_domania_only_after_the_grid_checks(monkeypatch):
    import enchilada._transforms as transforms

    calls = []

    def recording(stacked, source_domain, target_domain, **options):
        calls.append((stacked.shape, source_domain, target_domain, options))
        return np.zeros((stacked.shape[0], 8, 8))

    monkeypatch.setattr(transforms, "_wdm_transform", recording)
    data = observations()
    with pytest.raises(ValueError, match="WDM target requires wdm_grid"):
        transforms.transform_channels(
            data.channel_data,
            source_domain="time",
            target_domain="wdm",
            num_time_samples=64,
            sample_rate_hz=4.0,
        )
    with pytest.raises(ValueError, match="WDM source requires wdm_grid"):
        transforms.transform_channels(
            {"A": np.zeros((8, 8))},
            source_domain="wdm",
            target_domain="time",
            num_time_samples=64,
            sample_rate_hz=4.0,
        )
    assert calls == []
    converted = transform(data, "wdm", num_frequency_divisions=8)
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    assert converted.wdm_grid == grid
    assert calls == [
        (
            (2, 64),
            "time",
            "wdm",
            {
                "num_time_samples": 64,
                "sample_interval_s": 0.25,
                "source_wdm_grid": None,
                "target_wdm_grid": grid,
            },
        )
    ]


def test_white_noise_wdm_coefficient_variance_matches_white_covariance():
    """E[w^2] == sigma^2 / fs at every folded cell for white noise of variance
    sigma^2 per sample: the WDM constant used by white_covariance."""
    fs, n, sigma = 0.2, 1 << 12, 0.7
    rng = np.random.default_rng(3)
    acc = None
    trials = 40
    for _ in range(trials):
        t = L1Data(
            channel_data={"A": rng.normal(0.0, sigma, n)},
            sample_rate_hz=fs,
            channel_names=("A",),
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
        )
        wdm = transform(t, "wdm", num_frequency_divisions=64)
        power = wdm.channel_data["A"] ** 2
        acc = power if acc is None else acc + power
    measured = acc / trials
    predicted = white_covariance(wdm, sigma**2).A
    np.testing.assert_array_equal(predicted, sigma**2 / fs)
    ratio = float(np.mean(measured / predicted))
    assert ratio == pytest.approx(1.0, abs=0.05)
    row_zero = float(np.mean(measured[0] / predicted[0]))  # DC and Nyquist slots
    assert row_zero == pytest.approx(1.0, abs=0.1)
