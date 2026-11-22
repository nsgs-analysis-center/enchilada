"""Wheel serves each block on its own grid from a WDM home grid.

Templates are kept on the home grid and the views come back on the observation
grid to rounding. A noise block's publication is kept natively in the ledger and
translated onto the home grid for the sum, which every block receives translated
onto its own grid; a noise block must still publish on the grid it registered on.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from conftest import starting_at, white_covariance
from enchilada import (
    BlockState,
    Covariance,
    FixedNoiseBlock,
    L1Data,
    WDMGrid,
    Wheel,
    transform,
)
from enchilada.testing import EchoBlock


class EchoNoise(EchoBlock):
    """An EchoBlock declaring the noise role; its start supplies the covariance."""

    role = "noise"


def observations():
    return L1Data(
        channel_data={name: np.full(64, 5.0) for name in ("A", "E")},
        channel_names=("A", "E"),
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=12.0,
    )


def block_reference(data, domain):
    """An L1Data on the block's grid and the matching Wheel.add options.

    Built by hand, so no test here needs a WDM transform to build a covariance.
    """
    if domain == "frequency":
        return data.to_frequency(), {}
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    zeros = {name: np.zeros(grid.array_shape) for name in data.channel_names}
    reference = replace(data, channel_data=zeros, data_domain="wdm")
    return reference, {"num_frequency_divisions": 8}


class ConstantSignal:
    role = "signal"

    @dataclass
    class State:
        updates: int = 0

    def __init__(self, name, amplitude, expected_domain):
        self.name = name
        self.amplitude = amplitude
        self.expected_domain = expected_domain
        self.state = self.State()

    def _render(self, residual):
        assert residual.data_domain == self.expected_domain
        time_data = transform(residual, "time")
        result = time_data.block_state(
            {
                name: np.full_like(values, self.amplitude)
                for name, values in time_data.channel_data.items()
            },
            parameters={"amplitude": self.amplitude},
        )
        grid_options = (
            {}
            if residual.wdm_grid is None
            else {"num_frequency_divisions": residual.wdm_grid.num_frequency_divisions}
        )
        return transform(
            result, residual.data_domain, reference_data=time_data, **grid_options
        )

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self._render(conditional_residual)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        previous = transform(
            current_block_state, "time", reference_data=conditional_residual
        )
        np.testing.assert_allclose(previous.template["A"], self.amplitude, atol=1e-13)
        residual = transform(conditional_residual, "time")
        np.testing.assert_allclose(
            residual.channel_data["A"], 1.5 + self.amplitude, atol=1e-12
        )
        self.state.updates += 1
        return self._render(conditional_residual)


@pytest.mark.parametrize("canonical_domain", ["time", "frequency", "wdm"])
@pytest.mark.parametrize("third_domain", ["frequency", "wdm"])
def test_mixed_blocks_receive_matching_inputs_and_return_canonical_states(
    canonical_domain, third_domain
):
    data = observations()
    options = {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    canonical = transform(data, canonical_domain, **options)
    wheel = Wheel(canonical, num_frequency_divisions=8, random_seed=42)
    domains = ("time", "frequency", third_domain)
    blocks = {}
    for name, amplitude, domain in zip(
        ("one", "two", "half"), (1.0, 2.0, 0.5), domains, strict=True
    ):
        options = {"num_time_divisions": 4} if domain == "wdm" else {}
        block = ConstantSignal(name, amplitude, domain)
        blocks[name] = block
        wheel.add(block, data_domain=domain, **options)
    wheel.run(3)
    assert wheel.working_residual.data_domain == canonical_domain
    restored = transform(wheel.residual(), "time")
    np.testing.assert_allclose(restored.channel_data["A"], 1.5, atol=1e-12)
    for name in ("one", "two", "half"):
        assert blocks[name].state.updates == 3
        assert wheel.contribution(name)["A"].shape == canonical.channel_data["A"].shape


def test_wdm_noise_publication_is_held_on_the_wdm_grid():
    class PixelNoise:
        role = "noise"
        name = "noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert covariance is None  # the only noise block receives no sum
            current_block_state.covariance.check_compatible(conditional_residual)
            assert isinstance(current_block_state.covariance, Covariance)
            return current_block_state

    data = observations()
    wdm_data = transform(data, "wdm", num_frequency_divisions=8)
    grid = wdm_data.wdm_grid
    variances = np.ones(grid.array_shape)
    variances[1:] *= np.arange(1, grid.num_frequency_divisions)[:, None]
    initial = BlockState(
        covariance=Covariance.like(
            wdm_data, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data, num_frequency_divisions=8)
    wheel.add(
        starting_at(PixelNoise(), initial),
        data_domain="wdm",
        num_frequency_divisions=8,
    )
    assert wheel.covariance is not None
    assert wheel.ledger["noise"].template is None
    wheel.run(2)
    np.testing.assert_allclose(
        wheel.residual().channel_data["A"], data.channel_data["A"], rtol=1e-12
    )
    assert wheel.covariance.grid == wdm_data.grid == wheel.home_grid
    np.testing.assert_array_equal(wheel.covariance.A, variances)


@pytest.mark.parametrize("failure", ["signal_shape", "signal_publishes_covariance"])
def test_failed_translated_result_preserves_accepted_state_and_rng(failure):
    fail = [True]

    class RandomSignal:
        role = "signal"
        name = "random"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            value = rng.normal()
            if fail[0]:
                if failure == "signal_shape":
                    return BlockState(template={"A": np.zeros(4, complex)})
                # A signal block may not publish a covariance, on any grid.
                return BlockState(
                    covariance=white_covariance(conditional_residual, 1.0)
                )
            return BlockState(parameters={"amplitude": value})

    actual, expected = (
        Wheel(observations(), num_frequency_divisions=8, random_seed=7),
        Wheel(observations(), num_frequency_divisions=8, random_seed=7),
    )
    for wheel in (actual, expected):
        wheel.add(
            starting_at(RandomSignal(), BlockState(parameters={"amplitude": 0.0})),
            data_domain="frequency",
        )
    before = actual.ledger["random"].parameters.copy()
    match = (
        "declares role 'signal'" if failure == "signal_publishes_covariance" else None
    )
    with pytest.raises(ValueError, match=match):
        actual.run(1)
    assert actual.ledger["random"].parameters == before
    assert actual.covariance is None
    fail[0] = False
    actual.run(1)
    expected.run(1)
    assert actual.ledger["random"].parameters == expected.ledger["random"].parameters
    np.testing.assert_allclose(
        actual.residual().channel_data["A"],
        observations().channel_data["A"],
        rtol=1e-12,
    )


def test_invalid_domain_selection_does_not_register_or_sample():
    class NeverCalled:
        role = "signal"
        name = "unregistered"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            raise AssertionError("registration must fail on the domain first")

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError

    wheel = Wheel(observations(), num_frequency_divisions=8)
    with pytest.raises(ValueError, match="data_domain|target_domain"):
        wheel.add(NeverCalled(), data_domain="unknown")
    with pytest.raises(ValueError, match="data_domain|wdm"):
        wheel.add(NeverCalled(), num_time_divisions=8)
    assert len(wheel.ledger) == 0


def test_a_start_on_the_wrong_wdm_grid_is_refused(monkeypatch):
    # add serves the residual on the block's (4, 16) grid before initial_sample
    # runs; the wrong template is then caught by the grid check on that grid,
    # before any translation of the template onto the home grid, so domania
    # never sees it. The patch is installed from inside initial_sample because
    # add legitimately uses domania to build the residual first.
    data = observations()
    wheel = Wheel(data, num_frequency_divisions=8)
    # Both grids represent 64 samples, but the selected (Nf, Nt) = (4, 16)
    # requires shape (4, 16), not the (8, 8) coefficients of an (8, 8) grid.
    wrong = BlockState(template={name: np.zeros((8, 8)) for name in data.channel_names})

    def unexpected_transform(*args, **kwargs):
        raise AssertionError("the wrong template must not reach domania")

    class WrongGrid(ConstantSignal):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            assert conditional_residual.data_domain == "wdm"  # add's transform is done
            monkeypatch.setattr(
                "enchilada._transforms._wdm_transform", unexpected_transform
            )
            return wrong

    with pytest.raises(ValueError, match="shape"):
        wheel.add(
            WrongGrid("wrong_grid", 1.0, "wdm"),
            data_domain="wdm",
            num_frequency_divisions=4,
        )
    assert len(wheel.ledger) == 0
    monkeypatch.undo()  # the view below is an inverse WDM transform
    for name in data.channel_names:
        np.testing.assert_allclose(
            wheel.residual().channel_data[name], data.channel_data[name], rtol=1e-12
        )


def test_wheel_rejects_mutated_wdm_buffers_with_same_time_sample_count():
    data = L1Data(
        {name: np.zeros((4, 16)) for name in ("A", "E")},
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        data_domain="wdm",
    )
    original_grid = data.wdm_grid
    for name in data.channel_names:
        data.channel_data[name] = np.zeros((8, 8))
    with pytest.raises(ValueError, match="shape"):
        Wheel(data, num_frequency_divisions=8)
    assert data.wdm_grid is original_grid


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_explicit_domain_revalidates_mutated_start_arrays(domain):
    data = observations()
    wheel = Wheel(data, num_frequency_divisions=8)
    start = transform(data, domain).zero_block_state()
    start.template["A"] = start.template["A"].tolist()
    with pytest.raises(TypeError, match="numpy array"):
        wheel.add(
            starting_at(ConstantSignal("invalid_array", 1.0, domain), start),
            data_domain=domain,
        )
    assert len(wheel.ledger) == 0


def test_covariance_returned_on_another_grid_than_the_block_is_refused():
    """A noise block must publish on the grid of the data it received."""

    class TimeNoiseFromFrequency:
        role = "noise"
        name = "frequency_noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            return BlockState(
                covariance=white_covariance(
                    transform(conditional_residual, "time"), 1.0
                ),
                parameters={"level": 1.0},
            )

    data = observations()
    spectrum = data.to_frequency()
    start = BlockState(
        covariance=white_covariance(spectrum, 0.5), parameters={"level": 0.5}
    )
    wheel = Wheel(data, num_frequency_divisions=8, random_seed=3)
    wheel.add(
        starting_at(TimeNoiseFromFrequency(), start),
        data_domain="frequency",
    )
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(
        ValueError, match="frequency_noise.sample covariance is on grid"
    ):
        wheel.run(1)
    assert wheel._rng.bit_generator.state == rng_before
    entry = wheel.ledger["frequency_noise"]
    assert entry.parameters == {"level": 0.5}
    assert entry.template is None
    assert entry.covariance.grid == spectrum.grid
    np.testing.assert_array_equal(entry.covariance.A, white_covariance(spectrum, 0.5).A)
    assert wheel.covariance.grid == wheel.home_grid
    np.testing.assert_allclose(
        transform(wheel.covariance, "frequency").A,
        white_covariance(spectrum, 0.5).A,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        wheel.residual().channel_data["A"], data.channel_data["A"], rtol=1e-12
    )


def test_covariance_published_on_the_block_grid_is_held_there_in_the_ledger():
    class FrequencyNoise:
        role = "noise"
        name = "frequency_noise"

        def _publish(self, conditional_residual, level):
            assert conditional_residual.data_domain == "frequency"
            return BlockState(
                covariance=Covariance.like(
                    conditional_residual,
                    {
                        name: np.full(values.shape, level)
                        for name, values in conditional_residual.channel_data.items()
                    },
                ),
                parameters={"level": level},
            )

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return self._publish(conditional_residual, 1.0)

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return self._publish(conditional_residual, 1.0 + abs(rng.normal()))

    data = observations()
    wheel = Wheel(data, num_frequency_divisions=8, random_seed=3)
    wheel.add(FrequencyNoise(), data_domain="frequency")
    wheel.run(2)
    entry = wheel.ledger["frequency_noise"]
    level = entry.parameters["level"]
    assert entry.covariance.grid == data.to_frequency().grid
    np.testing.assert_array_equal(entry.covariance.A, level)
    assert wheel.covariance.grid == wheel.home_grid
    np.testing.assert_allclose(
        transform(wheel.covariance, "frequency").A, level, rtol=1e-12
    )
    assert wheel.working_residual.data_domain == "time"


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_start_on_the_block_grid_is_adopted_there(domain):
    """A start covariance is on the block's grid; the ledger keeps it there and
    the sum carries its translation."""
    data = observations()
    reference, options = block_reference(data, domain)
    variances = np.ones(reference.grid.channel_shape)
    start = BlockState(
        covariance=Covariance.like(
            reference, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data, num_frequency_divisions=8)
    wheel.add(
        starting_at(EchoNoise("noise"), start),
        data_domain=domain,
        **options,
    )
    assert wheel.ledger["noise"].covariance.grid == reference.grid
    np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, variances)
    assert wheel.covariance.grid == wheel.home_grid
    np.testing.assert_allclose(
        transform(wheel.covariance, domain, **options).A, variances, rtol=1e-12
    )


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_start_on_another_grid_than_the_block_is_refused_at_registration(
    domain,
):
    data = observations()
    _, options = block_reference(data, domain)
    start = BlockState(covariance=white_covariance(data, 1.0))  # on the time grid
    wheel = Wheel(data, num_frequency_divisions=8)
    with pytest.raises(ValueError, match="noise.initial_sample covariance is on grid"):
        wheel.add(
            starting_at(EchoNoise("noise"), start),
            data_domain=domain,
            **options,
        )
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None


def test_wheel_refuses_a_covariance_with_reordered_channel_names():
    data = observations()
    reordered = replace(data, channel_names=("E", "A"))
    wheel = Wheel(data, num_frequency_divisions=8)
    with pytest.raises(ValueError, match="channel_names"):
        wheel.add(FixedNoiseBlock("noise", white_covariance(reordered, 1.0)))
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None


def test_start_covariance_with_reordered_channel_names_is_refused_at_registration():
    data = observations()
    reordered = replace(data, channel_names=("E", "A"))
    start = BlockState(covariance=white_covariance(reordered, 1.0))
    wheel = Wheel(data, num_frequency_divisions=8, random_seed=3)
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(ValueError, match="channel_names"):
        wheel.add(starting_at(EchoNoise("noise"), start))
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._block_domains == {}
    assert wheel._block_indices == {}
    assert wheel._roles == {}
    assert wheel.covariance is None
    assert wheel._rng.bit_generator.state == rng_before


def test_noise_blocks_on_several_grids_add_on_the_home_grid():
    data = observations()
    reference, options = block_reference(data, "wdm")
    ones = np.ones(reference.grid.channel_shape)
    floor = Covariance.like(reference, {name: ones for name in data.channel_names})
    excess = Covariance.like(reference, {name: 2 * ones for name in data.channel_names})
    wheel = Wheel(data, num_frequency_divisions=8)
    wheel.add(FixedNoiseBlock("floor", floor), data_domain="wdm", **options)
    wheel.add(FixedNoiseBlock("excess", excess), data_domain="wdm", **options)
    assert wheel.covariance.grid == reference.grid == wheel.home_grid
    np.testing.assert_array_equal(wheel.covariance.A, 3.0)
    # A third noise block on the frequency grid: white noise of variance 1 per
    # time sample, which is 1 / fs per WDM cell once translated onto the home grid.
    spectral = white_covariance(data.to_frequency(), 1.0)
    wheel.add(FixedNoiseBlock("spectral", spectral), data_domain="frequency")
    total = 3.0 + 1.0 / data.fs
    np.testing.assert_allclose(wheel.covariance.A, total, rtol=1e-12)
    received: list[np.ndarray] = []

    class WdmSignal(EchoBlock):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            received.append(covariance.A.copy())
            return super().sample(
                conditional_residual, covariance, current_block_state, rng=rng
            )

    wheel.add(WdmSignal("wdm_signal"), data_domain="wdm", **options)
    wheel.run(1)
    np.testing.assert_allclose(wheel.covariance.A, total, rtol=1e-12)
    assert len(received) == 1
    np.testing.assert_allclose(received[0], total, rtol=1e-12)
    assert list(wheel.ledger) == ["floor", "excess", "spectral", "wdm_signal"]
