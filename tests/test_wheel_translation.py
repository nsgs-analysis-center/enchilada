"""Wheel keeps accepted signals on its observation grid.

The noise sum lives on the grid of the first noise publication. A block
registered on that grid receives it; a noise block on another grid is refused
at add, and a signal block on another grid is refused at add or at the next
run once a noise block has published elsewhere.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from conftest import const_block_state, starting_at, white_covariance
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

    def state_from_parameters(self, parameters, reference_data):
        return self._render(reference_data)

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
    wheel = Wheel(canonical, random_seed=42)
    domains = ("time", "frequency", third_domain)
    blocks = {}
    for name, amplitude, domain in zip(
        ("one", "two", "half"), (1.0, 2.0, 0.5), domains, strict=True
    ):
        options = {"num_time_divisions": 4} if domain == "wdm" else {}
        block = ConstantSignal(name, amplitude, domain)
        blocks[name] = block
        wheel.add(block, role="signal", data_domain=domain, **options)
    wheel.run(3)
    assert wheel.working_residual.data_domain == canonical_domain
    restored = transform(wheel.residual(), "time")
    np.testing.assert_allclose(restored.channel_data["A"], 1.5, atol=1e-12)
    for name in ("one", "two", "half"):
        assert blocks[name].state.updates == 3
        assert wheel.contribution(name)["A"].shape == canonical.channel_data["A"].shape


def test_wdm_noise_publication_is_held_on_the_wdm_grid():
    class PixelNoise:
        name = "noise"

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

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
    wheel = Wheel(data)
    wheel.add(
        starting_at(PixelNoise(), initial),
        role="noise",
        data_domain="wdm",
        num_frequency_divisions=8,
    )
    assert wheel.covariance is not None
    assert wheel.ledger["noise"].template is None
    wheel.run(2)
    np.testing.assert_array_equal(
        wheel.residual().channel_data["A"], data.channel_data["A"]
    )
    assert wheel.covariance.grid == wdm_data.grid
    np.testing.assert_array_equal(wheel.covariance.A, variances)


@pytest.mark.parametrize("failure", ["signal_shape", "signal_publishes_covariance"])
def test_failed_translated_result_preserves_accepted_state_and_rng(failure):
    fail = [True]

    class RandomSignal:
        name = "random"

        def state_from_parameters(self, parameters, reference_data):
            return BlockState(parameters=dict(parameters))

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
        Wheel(observations(), random_seed=7),
        Wheel(observations(), random_seed=7),
    )
    for wheel in (actual, expected):
        wheel.add(
            starting_at(RandomSignal(), BlockState(parameters={"amplitude": 0.0})),
            role="signal",
            data_domain="frequency",
        )
    before = actual.ledger["random"].parameters.copy()
    match = (
        "registered as a signal block"
        if failure == "signal_publishes_covariance"
        else None
    )
    with pytest.raises(ValueError, match=match):
        actual.run(1)
    assert actual.ledger["random"].parameters == before
    assert actual.covariance is None
    fail[0] = False
    actual.run(1)
    expected.run(1)
    assert actual.ledger["random"].parameters == expected.ledger["random"].parameters
    np.testing.assert_array_equal(
        actual.residual().channel_data["A"], observations().channel_data["A"]
    )


def test_invalid_domain_selection_does_not_register_or_sample():
    class NeverCalled:
        name = "unregistered"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            raise AssertionError("registration must fail on the domain first")

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError

    wheel = Wheel(observations())
    with pytest.raises(ValueError, match="data_domain|target_domain"):
        wheel.add(NeverCalled(), role="signal", data_domain="unknown")
    with pytest.raises(ValueError, match="data_domain|wdm"):
        wheel.add(NeverCalled(), role="signal", num_time_divisions=8)
    assert len(wheel.ledger) == 0


def test_a_start_on_the_wrong_wdm_grid_is_refused(monkeypatch):
    # add transforms the residual to WDM before initial_sample runs; the wrong
    # template is then caught by the grid check inside transform, before any
    # WDM conversion of the template itself, so domania never sees it. The patch
    # is installed from inside initial_sample because add legitimately uses
    # domania to build the residual first.
    data = observations()
    wheel = Wheel(data)
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
            role="signal",
            data_domain="wdm",
            num_frequency_divisions=4,
        )
    assert len(wheel.ledger) == 0
    for name in data.channel_names:
        np.testing.assert_array_equal(
            wheel.residual().channel_data[name], data.channel_data[name]
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
        Wheel(data)
    assert data.wdm_grid is original_grid


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_explicit_domain_revalidates_mutated_start_arrays(domain):
    data = observations()
    wheel = Wheel(data)
    start = transform(data, domain).zero_block_state()
    start.template["A"] = start.template["A"].tolist()
    with pytest.raises(TypeError, match="numpy array"):
        wheel.add(
            starting_at(ConstantSignal("invalid_array", 1.0, domain), start),
            role="signal",
            data_domain=domain,
        )
    assert len(wheel.ledger) == 0


@pytest.mark.parametrize("role", ["signal", "noise"])
def test_block_on_another_grid_than_the_covariance_is_refused_at_add(role):
    """The Wheel neither translates nor adds a covariance across grids: a block
    registered on another grid than the noise sum is refused before
    initial_sample is called, with a message for its role."""

    class NeverSampled:
        name = "frequency_block"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            raise AssertionError(
                "the Wheel must refuse before initial_sample is called"
            )

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError(
                "unreachable: sample exists only to pass add's callable check"
            )

    data = observations()
    wheel = Wheel(data, random_seed=3)
    wheel.add(FixedNoiseBlock("time_noise", white_covariance(data, 0.25)), role="noise")
    rng_before = wheel._rng.bit_generator.state
    if role == "signal":
        match = "does not translate between grids"
    else:
        match = "does not add across grids"
    with pytest.raises(ValueError, match=match) as info:
        wheel.add(NeverSampled(), role=role, data_domain="frequency")
    message = str(info.value)
    assert "'frequency_block'" in message
    assert "data_domain='frequency'" in message and "data_domain='time'" in message
    if role == "signal":
        assert "Register the block on the covariance's grid" in message
    else:
        assert "campaign covariance is on grid" in message
    assert wheel._rng.bit_generator.state == rng_before
    assert list(wheel.ledger) == ["time_noise"]
    assert "frequency_block" not in wheel._roles
    assert wheel.covariance.grid == data.grid
    np.testing.assert_array_equal(wheel.covariance.A, 0.25)


def test_wdm_block_on_a_time_covariance_is_refused_before_the_transform(monkeypatch):
    # The covariance-grid check in _block_inputs runs at add, before the
    # residual is transformed and before initial_sample.
    data = observations()
    wheel = Wheel(data)
    wheel.add(FixedNoiseBlock("noise", white_covariance(data, 1.0)), role="noise")

    def unexpected_transform(*args, **kwargs):
        raise AssertionError("the grid check must come before any WDM transform")

    monkeypatch.setattr("enchilada._transforms._wdm_transform", unexpected_transform)
    with pytest.raises(ValueError, match="does not translate between grids"):
        wheel.add(
            EchoBlock("wdm_block"),
            role="signal",
            data_domain="wdm",
            num_frequency_divisions=8,
        )
    with pytest.raises(ValueError, match="does not add across grids"):
        wheel.add(
            FixedNoiseBlock("wdm_noise", white_covariance(data, 1.0)),
            role="noise",
            data_domain="wdm",
            num_frequency_divisions=8,
        )
    assert list(wheel.ledger) == ["noise"]
    assert wheel.covariance.grid == data.grid


def test_default_block_is_refused_once_a_frequency_noise_block_has_published():
    """A signal block registered without data_domain is on the observation
    grid; once a noise block registered on the frequency grid publishes there,
    the signal block is refused at run."""

    class TimeSignal:
        name = "signal"

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError("refused at run before sample is reached")

    class FrequencyNoise:
        name = "noise"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            assert conditional_residual.data_domain == "frequency"
            return BlockState(
                covariance=white_covariance(conditional_residual, 1.0),
                parameters={"level": 1.0},
            )

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError("the signal block fails first in the cycle")

    data = observations()
    wheel = Wheel(data, random_seed=3)
    start = const_block_state(data, 0.5)
    start.parameters["amplitude"] = 0.5
    wheel.add(starting_at(TimeSignal(), start), role="signal")
    wheel.add(FrequencyNoise(), role="noise", data_domain="frequency")
    assert wheel.covariance.grid == transform(data, "frequency").grid
    signal_before = wheel.ledger["signal"]
    covariance_before = wheel.covariance
    residual_before = {
        name: values.copy()
        for name, values in wheel.working_residual.channel_data.items()
    }
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(
        ValueError,
        match="is registered on grid .* but the current covariance is on grid",
    ) as info:
        wheel.run(1)
    assert "'signal'" in str(info.value)
    assert wheel._rng.bit_generator.state == rng_before
    entry = wheel.ledger["signal"]
    assert entry.parameters == signal_before.parameters
    for values in entry.template.values():
        np.testing.assert_array_equal(values, 0.5)
    for name, values in residual_before.items():
        np.testing.assert_array_equal(wheel.working_residual.channel_data[name], values)
    assert wheel.ledger["noise"].parameters == {"level": 1.0}
    assert wheel.covariance.grid == covariance_before.grid
    np.testing.assert_array_equal(wheel.covariance.blocks, covariance_before.blocks)


def test_covariance_returned_on_another_grid_than_the_block_is_refused():
    """A noise block must publish on the grid of the data it received."""

    class TimeNoiseFromFrequency:
        name = "frequency_noise"

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

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
    wheel = Wheel(data, random_seed=3)
    wheel.add(
        starting_at(TimeNoiseFromFrequency(), start),
        role="noise",
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
    assert wheel.covariance.grid == spectrum.grid
    np.testing.assert_array_equal(wheel.covariance.A, white_covariance(spectrum, 0.5).A)
    np.testing.assert_array_equal(
        wheel.residual().channel_data["A"], data.channel_data["A"]
    )


def test_covariance_published_on_the_block_grid_is_held_there():
    class FrequencyNoise:
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

        def state_from_parameters(self, parameters, reference_data):
            return self._publish(reference_data, parameters["level"])

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return self._publish(conditional_residual, 1.0 + abs(rng.normal()))

    data = observations()
    wheel = Wheel(data, random_seed=3)
    wheel.add(FrequencyNoise(), role="noise", data_domain="frequency")
    wheel.run(2)
    assert wheel.covariance.grid == data.to_frequency().grid
    level = wheel.ledger["frequency_noise"].parameters["level"]
    np.testing.assert_array_equal(wheel.covariance.A, level)
    assert wheel.working_residual.data_domain == "time"


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_start_on_the_block_grid_is_adopted_there(domain):
    """A start covariance is on the block's grid; the Wheel keeps it there."""
    data = observations()
    reference, options = block_reference(data, domain)
    variances = np.ones(reference.grid.channel_shape)
    start = BlockState(
        covariance=Covariance.like(
            reference, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data)
    wheel.add(
        starting_at(EchoBlock("noise"), start),
        role="noise",
        data_domain=domain,
        **options,
    )
    assert wheel.covariance.grid == reference.grid
    np.testing.assert_array_equal(wheel.covariance.A, variances)
    assert wheel.ledger["noise"].covariance.grid == reference.grid


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_start_on_another_grid_than_the_block_is_refused_at_registration(
    domain,
):
    data = observations()
    _, options = block_reference(data, domain)
    start = BlockState(covariance=white_covariance(data, 1.0))  # on the time grid
    wheel = Wheel(data)
    with pytest.raises(ValueError, match="noise.initial_sample covariance is on grid"):
        wheel.add(
            starting_at(EchoBlock("noise"), start),
            role="noise",
            data_domain=domain,
            **options,
        )
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None


def test_wheel_refuses_a_covariance_with_reordered_channel_names():
    data = observations()
    reordered = replace(data, channel_names=("E", "A"))
    wheel = Wheel(data)
    with pytest.raises(ValueError, match="channel_names"):
        wheel.add(
            FixedNoiseBlock("noise", white_covariance(reordered, 1.0)), role="noise"
        )
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None


def test_start_covariance_with_reordered_channel_names_is_refused_at_registration():
    data = observations()
    reordered = replace(data, channel_names=("E", "A"))
    start = BlockState(covariance=white_covariance(reordered, 1.0))
    wheel = Wheel(data, random_seed=3)
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(ValueError, match="channel_names"):
        wheel.add(starting_at(EchoBlock("noise"), start), role="noise")
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._block_domains == {}
    assert wheel._block_indices == {}
    assert wheel._roles == {}
    assert wheel.covariance is None
    assert wheel._rng.bit_generator.state == rng_before


def test_two_wdm_noise_blocks_add_on_the_wdm_grid():
    data = observations()
    reference, options = block_reference(data, "wdm")
    ones = np.ones(reference.grid.channel_shape)
    floor = Covariance.like(reference, {name: ones for name in data.channel_names})
    excess = Covariance.like(reference, {name: 2 * ones for name in data.channel_names})
    wheel = Wheel(data)
    wheel.add(
        FixedNoiseBlock("floor", floor), role="noise", data_domain="wdm", **options
    )
    wheel.add(
        FixedNoiseBlock("excess", excess), role="noise", data_domain="wdm", **options
    )
    assert wheel.covariance.grid == reference.grid
    np.testing.assert_array_equal(wheel.covariance.A, 3.0)
    spectral = white_covariance(data.to_frequency(), 1.0)
    with pytest.raises(ValueError, match="does not add across grids"):
        wheel.add(
            FixedNoiseBlock("spectral", spectral), role="noise", data_domain="frequency"
        )
    received: list[np.ndarray] = []

    class WdmSignal(EchoBlock):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            received.append(covariance.A.copy())
            return super().sample(
                conditional_residual, covariance, current_block_state, rng=rng
            )

    wheel.add(WdmSignal("wdm_signal"), role="signal", data_domain="wdm", **options)
    wheel.run(1)
    np.testing.assert_array_equal(wheel.covariance.A, 3.0)
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], 3.0)
    assert list(wheel.ledger) == ["floor", "excess", "wdm_signal"]
