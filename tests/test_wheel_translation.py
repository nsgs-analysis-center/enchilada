"""Wheel keeps accepted signals on its observation grid.

A covariance stays on the grid its owner published it on. A block registered
on that grid receives it; a block on another grid is refused at run.
"""

import importlib.util
from dataclasses import dataclass, replace

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import BlockState, Covariance, L1Data, WDMGrid, Wheel, transform
from enchilada.testing import EchoBlock

requires_wdm = pytest.mark.skipif(
    importlib.util.find_spec("wdm") is None, reason="local WDM backend is optional"
)


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

    Built by hand, so no test here needs the WDM backend to build a covariance.
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


@pytest.mark.parametrize(
    "canonical_domain", ["time", "frequency", pytest.param("wdm", marks=requires_wdm)]
)
@pytest.mark.parametrize(
    "third_domain", ["frequency", pytest.param("wdm", marks=requires_wdm)]
)
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
        block_data = transform(data, domain, **options)
        wheel.add(
            block,
            initial_block_state=block._render(block_data),
            data_domain=domain,
            **options,
        )
    wheel.run(3)
    assert wheel.working_residual.data_domain == canonical_domain
    restored = transform(wheel.residual(), "time")
    np.testing.assert_allclose(restored.channel_data["A"], 1.5, atol=1e-12)
    for name in ("one", "two", "half"):
        assert blocks[name].state.updates == 3
        assert wheel.contribution(name)["A"].shape == canonical.channel_data["A"].shape


@requires_wdm
def test_wdm_noise_publication_is_held_on_the_wdm_grid():
    class PixelNoise:
        name = "noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            assert isinstance(current_block_state.covariance, Covariance)
            return current_block_state

    data = observations()
    wdm_data = transform(data, "wdm", num_frequency_divisions=8)
    grid = wdm_data.wdm_grid
    variances = np.where(grid.active_mask, 1.0, 0.0)
    variances[1:-1] *= np.arange(1, grid.num_frequency_divisions)[:, None]
    initial = BlockState(
        covariance=Covariance.like(
            wdm_data, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data)
    wheel.add(
        PixelNoise(),
        initial_block_state=initial,
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


@pytest.mark.parametrize("failure", ["signal_shape", "covariance_grid"])
def test_failed_translated_result_preserves_accepted_state_and_rng(failure):
    fail = [True]

    class RandomSignal:
        name = "random"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            value = rng.normal()
            if fail[0]:
                if failure == "signal_shape":
                    return BlockState(template={"A": np.zeros(4, complex)})
                return BlockState(
                    covariance=white_covariance(
                        transform(conditional_residual, "time"), 1.0
                    )
                )
            return BlockState(parameters={"amplitude": value})

    actual, expected = (
        Wheel(observations(), random_seed=7),
        Wheel(observations(), random_seed=7),
    )
    for wheel in (actual, expected):
        wheel.add(
            RandomSignal(),
            initial_block_state=BlockState(parameters={"amplitude": 0.0}),
            data_domain="frequency",
        )
    before = actual.ledger["random"].parameters.copy()
    match = "covariance is on grid" if failure == "covariance_grid" else None
    with pytest.raises(ValueError, match=match):
        actual.run(1)
    assert actual.ledger["random"].parameters == before
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

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError

    wheel = Wheel(observations())
    with pytest.raises(ValueError, match="data_domain|target_domain"):
        wheel.add(
            NeverCalled(), initial_block_state=BlockState(), data_domain="unknown"
        )
    with pytest.raises(ValueError, match="data_domain|wdm"):
        wheel.add(NeverCalled(), initial_block_state=BlockState(), num_time_divisions=8)
    assert len(wheel.ledger) == 0


def test_initial_wdm_signal_must_match_selected_grid_before_backend(monkeypatch):
    def unexpected_backend():
        raise AssertionError("invalid initial WDM grid reached the optional backend")

    monkeypatch.setattr("enchilada._transforms._load_wdm", unexpected_backend)
    data = observations()
    wheel = Wheel(data)
    # Both grids represent 64 samples, but selected (Nf, Nt) = (4, 16)
    # requires shape (5, 16), not the (9, 8) coefficients of an (8, 8) grid.
    initial = BlockState(
        template={name: np.zeros((9, 8)) for name in data.channel_names}
    )
    with pytest.raises(ValueError, match="shape"):
        wheel.add(
            ConstantSignal("wrong_grid", 1.0, "wdm"),
            initial_block_state=initial,
            data_domain="wdm",
            num_frequency_divisions=4,
            num_time_divisions=16,
        )
    assert len(wheel.ledger) == 0
    for name in data.channel_names:
        np.testing.assert_array_equal(
            wheel.residual().channel_data[name], data.channel_data[name]
        )


def test_wheel_rejects_mutated_wdm_buffers_with_same_time_sample_count():
    data = L1Data(
        {name: np.zeros((5, 16)) for name in ("A", "E")},
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        data_domain="wdm",
    )
    original_grid = data.wdm_grid
    for name in data.channel_names:
        data.channel_data[name] = np.zeros((9, 8))
    with pytest.raises(ValueError, match="shape"):
        Wheel(data)
    assert data.wdm_grid is original_grid


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_explicit_domain_revalidates_mutated_initial_signal_arrays(domain):
    data = observations()
    wheel = Wheel(data)
    initial = transform(data, domain).zero_block_state()
    initial.template["A"] = initial.template["A"].tolist()
    with pytest.raises(TypeError, match="numpy array"):
        wheel.add(
            ConstantSignal("invalid_array", 1.0, domain),
            initial_block_state=initial,
            data_domain=domain,
        )
    assert len(wheel.ledger) == 0


def test_block_on_another_grid_than_the_covariance_is_refused_at_run():
    """The Wheel does not translate a covariance: a block registered on another
    grid than the current covariance is refused before sample is called."""

    class NeverSampled:
        name = "frequency_block"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError("the Wheel must refuse before sample is called")

    data = observations()
    wheel = Wheel(data, white_covariance(data, 0.25), random_seed=3)
    wheel.add(NeverSampled(), initial_block_state=BlockState(), data_domain="frequency")
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(ValueError, match="does not translate between grids") as info:
        wheel.run(1)
    message = str(info.value)
    assert "'frequency_block'" in message
    assert "data_domain='frequency'" in message and "data_domain='time'" in message
    assert "Register the block on the covariance's grid" in message
    assert wheel._rng.bit_generator.state == rng_before
    entry = wheel.ledger["frequency_block"]
    assert entry.parameters == {}
    assert entry.template is None and entry.covariance is None
    assert wheel.covariance.grid == data.grid
    np.testing.assert_array_equal(wheel.covariance.A, 0.25)


def test_wdm_block_on_a_time_covariance_is_refused_before_the_backend(monkeypatch):
    def unexpected_backend():
        raise AssertionError("the grid check must come before any WDM transform")

    monkeypatch.setattr("enchilada._transforms._load_wdm", unexpected_backend)
    data = observations()
    wheel = Wheel(data, white_covariance(data, 1.0))
    wheel.add(
        EchoBlock("wdm_block"),
        initial_block_state=BlockState(),
        data_domain="wdm",
        num_frequency_divisions=8,
    )
    with pytest.raises(ValueError, match="does not translate between grids"):
        wheel.run(1)
    assert wheel.covariance.grid == data.grid


def test_default_block_is_refused_once_a_frequency_noise_owner_has_published():
    """A block registered without data_domain is on the observation grid; once a
    noise owner on the frequency grid publishes there, it is refused at run."""

    class TimeSignal:
        name = "signal"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert covariance is None  # only the first cycle reaches sample
            amplitude = rng.normal()
            return conditional_residual.block_state(
                {
                    name: np.full_like(values, amplitude)
                    for name, values in conditional_residual.channel_data.items()
                },
                parameters={"amplitude": amplitude},
            )

    class FrequencyNoise:
        name = "noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            return BlockState(
                covariance=white_covariance(conditional_residual, 1.0),
                parameters={"level": 1.0},
            )

    data = observations()
    wheel = Wheel(data, random_seed=3)
    wheel.add(
        TimeSignal(), initial_block_state=BlockState(parameters={"amplitude": 0.0})
    )
    wheel.add(
        FrequencyNoise(),
        initial_block_state=BlockState(parameters={"level": 0.0}),
        data_domain="frequency",
    )
    wheel.run(1)  # the signal samples without a covariance; the owner publishes
    assert wheel.covariance.grid == transform(data, "frequency").grid
    signal_before = wheel.ledger["signal"]
    covariance_before = wheel.covariance
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
    np.testing.assert_array_equal(entry.template["A"], signal_before.template["A"])
    assert wheel.ledger["noise"].parameters == {"level": 1.0}
    assert wheel.covariance.grid == covariance_before.grid
    np.testing.assert_array_equal(wheel.covariance.blocks, covariance_before.blocks)


def test_covariance_returned_on_another_grid_than_the_block_is_refused():
    """A block must publish on the grid of the data it received."""

    class TimeNoiseFromFrequency:
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
    wheel = Wheel(data, random_seed=3)
    wheel.add(
        TimeNoiseFromFrequency(),
        initial_block_state=BlockState(parameters={"level": 0.0}),
        data_domain="frequency",
    )
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(
        ValueError, match="frequency_noise.sample covariance is on grid"
    ):
        wheel.run(1)
    assert wheel.covariance is None
    assert wheel._rng.bit_generator.state == rng_before
    entry = wheel.ledger["frequency_noise"]
    assert entry.parameters == {"level": 0.0}
    assert entry.template is None and entry.covariance is None
    np.testing.assert_array_equal(
        wheel.residual().channel_data["A"], data.channel_data["A"]
    )


def test_covariance_published_on_the_block_grid_is_held_there():
    class FrequencyNoise:
        name = "frequency_noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            level = 1.0 + abs(rng.normal())
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

    data = observations()
    wheel = Wheel(data, random_seed=3)
    wheel.add(
        FrequencyNoise(),
        initial_block_state=BlockState(parameters={"level": 0.0}),
        data_domain="frequency",
    )
    wheel.run(2)
    assert wheel.covariance.grid == data.to_frequency().grid
    level = wheel.ledger["frequency_noise"].parameters["level"]
    np.testing.assert_array_equal(wheel.covariance.A, level)
    assert wheel.working_residual.data_domain == "time"


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_seed_on_the_block_grid_is_adopted_there(domain):
    """A seed covariance is on the block's grid; the Wheel keeps it there."""
    data = observations()
    reference, options = block_reference(data, domain)
    variances = np.where(reference.grid.active_mask, 1.0, 0.0)
    seed = BlockState(
        covariance=Covariance.like(
            reference, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data)
    wheel.add(
        EchoBlock("noise"), initial_block_state=seed, data_domain=domain, **options
    )
    assert wheel.covariance.grid == reference.grid
    np.testing.assert_array_equal(wheel.covariance.A, variances)
    assert wheel.ledger["noise"].covariance.grid == reference.grid


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_seed_on_another_grid_than_the_block_is_refused_at_registration(
    domain,
):
    data = observations()
    _, options = block_reference(data, domain)
    seed = BlockState(covariance=white_covariance(data, 1.0))  # on the time grid
    wheel = Wheel(data)
    with pytest.raises(
        ValueError, match="noise.initial_block_state covariance is on grid"
    ):
        wheel.add(
            EchoBlock("noise"), initial_block_state=seed, data_domain=domain, **options
        )
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None


def test_wheel_refuses_a_covariance_with_reordered_channel_names():
    data = observations()
    reordered = replace(data, channel_names=("E", "A"))
    with pytest.raises(ValueError, match="channel_names"):
        Wheel(data, white_covariance(reordered, 1.0))


def test_seed_covariance_with_reordered_channel_names_is_refused_at_registration():
    data = observations()
    reordered = replace(data, channel_names=("E", "A"))
    seed = BlockState(covariance=white_covariance(reordered, 1.0))
    wheel = Wheel(data, random_seed=3)
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(ValueError, match="channel_names"):
        wheel.add(EchoBlock("noise"), initial_block_state=seed)
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._block_domains == {}
    assert wheel._block_indices == {}
    assert wheel.covariance is None
    assert wheel._rng.bit_generator.state == rng_before
