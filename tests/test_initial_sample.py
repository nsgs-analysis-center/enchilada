"""Wheel.add draws a block's start from initial_sample and adopts it like a sample."""

from copy import deepcopy
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import (
    Block,
    BlockState,
    Covariance,
    FixedNoiseBlock,
    L1Data,
    Wheel,
    transform,
)
from enchilada.testing import check_block


@pytest.fixture
def observed():
    return L1Data(
        channel_data={"A": np.full(64, 5.0)},
        channel_names=("A",),
        sample_rate_hz=0.4,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )


class StartingBlock:
    """Starts at a configured amplitude on whatever grid it is handed.

    `state.starts` records every `initial_sample` call's inputs; `state.counter`
    and `state.draw` record `sample` calls, as the old SeededBlock did.
    """

    name = "starting"
    role = "signal"

    @dataclass
    class State:
        counter: int = 0
        draw: float = 0.0
        starts: list = field(default_factory=list)

    def __init__(self, amplitude=2.0):
        self.amplitude = amplitude
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self.state.starts.append((conditional_residual, covariance, rng))
        time_reference = transform(conditional_residual, "time")
        start = time_reference.block_state(
            {"A": np.full(time_reference.num_time_samples, self.amplitude)},
            parameters={"amplitude": self.amplitude},
        )
        grid = conditional_residual.wdm_grid
        grid_options = (
            {}
            if grid is None
            else {"num_frequency_divisions": grid.num_frequency_divisions}
        )
        return transform(
            start,
            conditional_residual.data_domain,
            reference_data=time_reference,
            **grid_options,
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        assert current_block_state.parameters["amplitude"] == self.amplitude
        assert (
            current_block_state.template["A"].shape
            == conditional_residual.channel_data["A"].shape
        )
        self.state.counter += 1
        self.state.draw = rng.normal()
        return current_block_state


def noise_start(level):
    """A StartingBlock whose start publishes only white noise of `level`."""

    class NoiseStart(StartingBlock):
        role = "noise"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            self.state.starts.append((conditional_residual, covariance, rng))
            return BlockState(covariance=white_covariance(conditional_residual, level))

    return NoiseStart()


def test_block_without_initial_sample_is_refused(observed):
    class SampleOnly:
        name = "sample_only"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    assert not isinstance(SampleOnly(), Block)
    assert isinstance(StartingBlock(), Block)
    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=1)
    before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises(TypeError, match="does not implement initial_sample"):
        wheel.add(SampleOnly())
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._rng.bit_generator.state == before


def test_registration_calls_initial_sample_once_with_the_block_inputs(observed):
    shared = []

    class ChecksAliasing(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            # Checked at call time: _adopt replaces the working residual afterwards.
            shared.append(
                np.shares_memory(
                    conditional_residual.channel_data["A"],
                    wheel._working_residual.channel_data["A"],
                )
            )
            return super().initial_sample(conditional_residual, covariance, rng=rng)

    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=5)
    block = ChecksAliasing()
    wheel.add(block)
    assert len(block.state.starts) == 1
    residual, covariance, rng = block.state.starts[0]
    np.testing.assert_allclose(residual.channel_data["A"], 5.0)
    assert covariance is None
    assert isinstance(rng, np.random.Generator)
    np.testing.assert_allclose(wheel.contribution("starting")["A"], 2.0)
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 3.0)
    # The block received its own copy of the residual.
    assert shared == [False]
    wheel.run(1)
    assert len(block.state.starts) == 1
    assert block.state.counter == 1


def test_initial_sample_receives_the_campaign_covariance(observed):
    wheel = Wheel(observed, num_frequency_divisions=8)
    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 0.25)))
    block = StartingBlock()
    wheel.add(block)
    _, covariance, _ = block.state.starts[0]
    assert isinstance(covariance, Covariance)
    np.testing.assert_allclose(covariance.A, 0.25)
    assert covariance is not wheel._noise_sum.total
    assert not covariance.blocks.flags.writeable


def test_initial_sample_consumes_the_campaign_rng_reproducibly(observed):
    class PriorStart(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            self.amplitude = float(rng.normal())
            return super().initial_sample(conditional_residual, covariance, rng=rng)

    def start(seed, block):
        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=seed)
        wheel.add(block)
        wheel.run(1)
        return wheel.ledger["starting"].parameters["amplitude"], block.state.draw

    same_a, same_b, other = (start(seed, PriorStart()) for seed in (11, 11, 12))
    assert same_a == same_b
    assert same_a != other
    rng = np.random.default_rng(11)
    first_draw = rng.normal()
    # The start took the seed's first draw, so the first sample sees the second.
    assert same_a[0] == first_draw
    assert same_a[1] == rng.normal()
    # A start that draws nothing leaves the stream where it was.
    _, draw = start(11, StartingBlock())
    assert draw == first_draw


@pytest.mark.parametrize(
    ("canonical_domain", "block_domain"),
    [
        ("time", None),
        ("frequency", "time"),
        ("time", "frequency"),
        ("time", "wdm"),
        ("wdm", "wdm"),
    ],
)
def test_initial_sample_receives_inputs_on_the_selected_grid_and_round_trips(
    observed, canonical_domain, block_domain
):
    canonical_options = (
        {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    )
    canonical = transform(observed, canonical_domain, **canonical_options)
    block_options = {"num_time_divisions": 4} if block_domain == "wdm" else {}
    wheel = Wheel(canonical, num_frequency_divisions=8, random_seed=23)
    block = StartingBlock()
    wheel.add(block, data_domain=block_domain, **block_options)
    residual, _, _ = block.state.starts[0]
    assert residual.data_domain == (block_domain or canonical_domain)
    if block_domain == "wdm":
        assert residual.wdm_grid.num_time_divisions == 4
    np.testing.assert_allclose(transform(residual, "time").channel_data["A"], 5.0)
    assert block.state.counter == 0
    np.testing.assert_allclose(transform(wheel.residual(), "time").channel_data["A"], 3)
    np.testing.assert_array_equal(
        wheel.observed_data.channel_data["A"], canonical.channel_data["A"]
    )
    assert (
        wheel.contribution("starting")["A"].shape == canonical.channel_data["A"].shape
    )
    wheel.run(1)
    assert block.state.counter == 1
    assert block.state.draw == np.random.default_rng(23).normal()


def test_adopted_start_is_an_independent_snapshot(observed):
    returned = {}

    class RememberedStart(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            start = replace(
                super().initial_sample(conditional_residual, covariance, rng=rng),
                parameters={"amplitude": 2.0, "positions": [np.array([1, 2])]},
            )
            returned["start"] = start
            return start

    class RememberedNoise(StartingBlock):
        name = "noise"
        role = "noise"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            start = BlockState(covariance=white_covariance(conditional_residual, 0.25))
            returned["noise"] = start
            return start

    wheel = Wheel(observed, num_frequency_divisions=8)
    wheel.add(RememberedNoise())
    wheel.add(RememberedStart())
    start = returned["start"]
    start.template["A"][:] = 100
    start.parameters["amplitude"] = 100
    start.parameters["positions"][0][:] = 100
    noise = returned["noise"]
    noise.covariance.blocks.setflags(write=True)
    noise.covariance.A[:] = 100
    accepted = wheel.ledger["starting"]
    assert accepted.parameters["amplitude"] == 2.0
    np.testing.assert_array_equal(accepted.parameters["positions"][0], [1, 2])
    np.testing.assert_allclose(wheel.contribution("starting")["A"], 2)
    np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 0.25)
    np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)


@pytest.mark.parametrize("block_domain", [None, "frequency"])
@pytest.mark.parametrize(
    "failure", ["raises", "type", "shape", "nonfinite", "covariance", "copy"]
)
def test_failed_initial_sample_leaves_registration_and_rng_unchanged(
    observed, failure, block_domain
):
    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("start state cannot be copied")

    class FailingStart(StartingBlock):
        def initial_sample(self, conditional_residual, covariance, *, rng):
            rng.normal()  # a discarded draw must never reach the campaign RNG
            # A mutated input must never reach the campaign either.
            conditional_residual.channel_data["A"][:] = 0.0
            good = super().initial_sample(conditional_residual, covariance, rng=rng)
            if failure == "raises":
                raise RuntimeError("the prior is improper here")
            if failure == "type":
                return {"amplitude": 2.0}
            if failure == "shape":
                return replace(good, template={"A": np.zeros(3)})
            if failure == "nonfinite":
                good.template["A"][1] = np.nan
                return good
            if failure == "covariance":
                return replace(
                    good, covariance=white_covariance(conditional_residual, 1.0)
                )
            return replace(good, parameters={"amplitude": CannotCopy()})

    # Each kind has its own error, so one failure cannot pass for another.
    expected, message = {
        "raises": (RuntimeError, "the prior is improper here"),
        "type": (TypeError, "starting.initial_sample must return a BlockState"),
        "shape": (ValueError, "length 3, expected"),
        "nonfinite": (ValueError, "finite"),
        "covariance": (ValueError, "declares role 'signal'"),
        "copy": (RuntimeError, "start state cannot be copied"),
    }[failure]
    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=7)
    before = deepcopy(wheel._rng.bit_generator.state)
    with pytest.raises(expected, match=message):
        wheel.add(FailingStart(), data_domain=block_domain)
    assert len(wheel.ledger) == 0
    assert wheel._blocks == []
    assert wheel._block_domains == {}
    assert wheel._roles == {}
    assert wheel._rng.bit_generator.state == before
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 5)
    # The same name can be retried, and the discarded draw never reached the RNG.
    block = StartingBlock()
    wheel.add(block, data_domain=block_domain)
    wheel.run(1)
    assert block.state.draw == np.random.default_rng(7).normal()


@pytest.mark.parametrize("with_noise", [False, True])
def test_start_without_signal_can_sample_in_selected_domain(observed, with_noise):
    class NoiseOrNothing(StartingBlock):
        role = "noise" if with_noise else "signal"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            assert conditional_residual.data_domain == "frequency"
            # White noise with variance 0.25 on the block's frequency grid:
            # C = (Tobs / 2) * (2 sigma^2 / fs) = N sigma^2 / fs^2 = 64 * 0.25 / 0.4**2.
            published = (
                Covariance.like(
                    conditional_residual,
                    {
                        ch: np.full(33, 100.0)
                        for ch in conditional_residual.channel_names
                    },
                )
                if with_noise
                else None
            )
            return BlockState(covariance=published, parameters={"variance": 0.25})

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            assert current_block_state.template is None
            assert current_block_state.parameters == {"variance": 0.25}
            # The only noise block receives no sum; its own model is on its state.
            assert covariance is None
            if with_noise:
                current_block_state.covariance.check_compatible(conditional_residual)
                np.testing.assert_allclose(current_block_state.covariance.A, 100.0)
            self.state.counter += 1
            return current_block_state

    wheel = Wheel(observed, num_frequency_divisions=8)
    block = NoiseOrNothing()
    wheel.add(block, data_domain="frequency")
    wheel.run(1)
    assert block.state.counter == 1
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 5)
    if with_noise:
        assert wheel.covariance.grid == wheel.home_grid
        np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)
        np.testing.assert_allclose(transform(wheel.covariance, "frequency").A, 100.0)
    else:
        assert wheel.covariance is None


def test_a_noise_block_must_publish_a_covariance_on_every_call(observed):
    class DropsNoise(StartingBlock):
        role = "noise"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return BlockState(
                covariance=white_covariance(conditional_residual, 0.25),
                parameters={"variance": 0.25},
            )

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return BlockState(parameters={"variance": 0.25})

    wheel = Wheel(observed, num_frequency_divisions=8)
    wheel.add(DropsNoise())
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 5)
    np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)
    with pytest.raises(
        ValueError,
        match="starting.sample must return a covariance because 'starting' "
        "declares role 'noise'",
    ):
        wheel.run(1)
    np.testing.assert_array_equal(wheel.ledger["starting"].covariance.A, 0.25)
    np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)


def test_two_noise_starts_add_and_the_second_receives_the_first(observed):
    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
    first, second = noise_start(1.0), noise_start(2.0)
    first.name, second.name = "first", "second"
    wheel.add(first)
    wheel.add(second)
    assert first.state.starts[0][1] is None  # alone when it started
    np.testing.assert_allclose(second.state.starts[0][1].A, 1.0)
    assert list(wheel.ledger) == ["first", "second"]
    np.testing.assert_array_equal(wheel.ledger["first"].covariance.A, 1.0)
    np.testing.assert_array_equal(wheel.ledger["second"].covariance.A, 2.0)
    np.testing.assert_allclose(wheel.covariance.A, 3.0 / observed.fs)


def test_check_block_can_exercise_a_configured_start(observed):
    block = StartingBlock(amplitude=4.0)
    check_block(block, observed, num_cycles=2, num_frequency_divisions=8)
    assert len(block.state.starts) == 1
    assert block.state.counter == 2
    np.testing.assert_allclose(block.state.starts[0][0].channel_data["A"], 5.0)


def test_later_blocks_start_from_the_residual_of_earlier_starts(observed):
    wheel = Wheel(observed, num_frequency_divisions=8)
    first, second = StartingBlock(amplitude=2.0), StartingBlock(amplitude=1.0)
    first.name, second.name = "first", "second"
    wheel.add(first)
    wheel.add(second, data_domain="frequency")
    first_residual = first.state.starts[0][0]
    second_residual = second.state.starts[0][0]
    np.testing.assert_allclose(first_residual.channel_data["A"], 5.0)
    assert second_residual.data_domain == "frequency"
    np.testing.assert_allclose(transform(second_residual, "time").channel_data["A"], 3)
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 2.0)


def test_a_signal_block_on_another_grid_receives_the_translated_covariance(observed):
    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 0.25)))
    block = StartingBlock()
    wheel.add(block, data_domain="frequency")
    residual, covariance, _ = block.state.starts[0]
    spectrum = observed.to_frequency()
    assert residual.grid == spectrum.grid
    assert covariance.grid == spectrum.grid
    np.testing.assert_allclose(covariance.A, white_covariance(spectrum, 0.25).A)
    assert list(wheel.ledger) == ["noise", "starting"]
    assert wheel.covariance.grid == wheel.home_grid
    wheel.run(1)
    assert block.state.counter == 1


def test_start_that_publishes_a_covariance_is_handed_to_the_next_block(observed):
    wheel = Wheel(observed, num_frequency_divisions=8)
    noise, signal = noise_start(0.5), StartingBlock()
    noise.name = "noise"
    wheel.add(noise)
    wheel.add(signal)
    _, covariance, _ = signal.state.starts[0]
    np.testing.assert_allclose(covariance.A, 0.5)
    assert covariance.grid == observed.grid
