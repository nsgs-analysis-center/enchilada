"""Published estimates and campaign RNG are isolated from block-local state."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from conftest import starting_at, white_covariance
from enchilada import FixedNoiseBlock, Wheel, transform


class Counter:
    role = "signal"
    name = "counter"

    @dataclass
    class State:
        n: int = 0

    def __init__(self):
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.zero_block_state(parameters={"a": 0.0})

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.n += 1
        current_block_state.parameters["a"] = rng.normal()
        return current_block_state


def test_ledger_carries_estimates_and_block_keeps_continuation(observed):
    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=4)
    block = Counter()
    wheel.add(block)
    assert block.state.n == 0
    initial = wheel.ledger["counter"].parameters["a"]
    wheel.run(2)
    assert block.state.n == 2
    assert wheel.ledger["counter"].parameters["a"] != initial


def test_pristine_working_and_input_are_independent(observed):
    original = observed.channel_data["A"].copy()
    wheel = Wheel(observed, num_frequency_divisions=8)
    observed.channel_data["A"][:] = 100
    wheel.observed_data.channel_data["A"][:] = 200
    wheel.working_residual.channel_data["A"][:] = 300
    np.testing.assert_array_equal(wheel.observed_data.channel_data["A"], original)
    np.testing.assert_allclose(
        wheel.working_residual.channel_data["A"], original, rtol=1e-12, atol=1e-12
    )


def test_block_state_parameters_are_copied_on_adoption_and_read(observed):
    initial = observed.zero_block_state(
        parameters={"a": 0.0, "amplitudes": [np.ones(2)]}
    )
    wheel = Wheel(observed, num_frequency_divisions=8)
    wheel.add(starting_at(Counter(), initial))
    initial.parameters["amplitudes"][0][:] = 10
    wheel.ledger["counter"].parameters["amplitudes"][0][:] = 20
    np.testing.assert_array_equal(
        wheel.ledger["counter"].parameters["amplitudes"][0], [1, 1]
    )


def test_failed_sample_rolls_back_published_state_data_and_rng(observed):
    fail = [True]

    class FailsOnce(Counter):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            result = super().sample(
                conditional_residual, covariance, current_block_state, rng=rng
            )
            conditional_residual.channel_data["A"][:] = 999
            if fail[0]:
                fail[0] = False
                return replace(
                    result,
                    template={
                        ch: np.full_like(x, np.nan) for ch, x in result.template.items()
                    },
                )
            return result

    actual, expected = (
        Wheel(observed, num_frequency_divisions=8, random_seed=5),
        Wheel(observed, num_frequency_divisions=8, random_seed=5),
    )
    block = FailsOnce()
    actual.add(block)
    expected.add(Counter())
    before = actual.ledger["counter"].parameters.copy()
    with pytest.raises(ValueError, match="non-finite"):
        actual.run(1)
    assert actual.ledger["counter"].parameters == before
    assert block.state.n == 1
    np.testing.assert_allclose(
        actual.working_residual.channel_data["A"],
        observed.channel_data["A"],
        rtol=1e-12,
        atol=1e-12,
    )
    actual.run(1)
    expected.run(1)
    assert actual.ledger["counter"].parameters == expected.ledger["counter"].parameters


def test_separate_blocks_keep_independent_campaign_continuation(observed):
    first, second = Counter(), Counter()
    one, two = (
        Wheel(observed, num_frequency_divisions=8, random_seed=7),
        Wheel(observed, num_frequency_divisions=8, random_seed=7),
    )
    one.add(first)
    two.add(second)
    one.run(2)
    assert first.state.n == 2
    assert second.state.n == 0
    two.run(1)
    two.run(1)
    assert first.state.n == second.state.n == 2
    assert one.ledger["counter"].parameters == two.ledger["counter"].parameters


def test_retained_rng_reference_cannot_advance_campaign(observed):
    retained = []

    class RetainsRng(Counter):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            retained.append(rng)
            return super().sample(
                conditional_residual, covariance, current_block_state, rng=rng
            )

    actual, expected = (
        Wheel(observed, num_frequency_divisions=8, random_seed=11),
        Wheel(observed, num_frequency_divisions=8, random_seed=11),
    )
    actual.add(RetainsRng())
    expected.add(Counter())
    actual.run(1)
    expected.run(1)
    retained[0].normal(size=100)
    actual.run(1)
    expected.run(1)
    assert actual.ledger["counter"].parameters == expected.ledger["counter"].parameters


@pytest.mark.parametrize("failure", ["copy", "covariance", "exception"])
def test_failed_adoption_preserves_rng_and_published_state(observed, failure):
    fail = [True]

    class CannotCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy parameter")

    class Invalid(Counter):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            result = super().sample(
                conditional_residual, covariance, current_block_state, rng=rng
            )
            if fail[0]:
                fail[0] = False
                if failure == "copy":
                    return replace(result, parameters={"amplitude": CannotCopy()})
                if failure == "exception":
                    raise RuntimeError("sampler failed")
                # A signal block may not publish a covariance at all.
                return result.with_covariance(white_covariance(conditional_residual, 1))
            return result

    actual, expected = (
        Wheel(observed, num_frequency_divisions=8, random_seed=12),
        Wheel(observed, num_frequency_divisions=8, random_seed=12),
    )
    block = Invalid()
    actual.add(block)
    expected.add(Counter())
    expected_error, message = {
        "copy": (RuntimeError, "cannot copy parameter"),
        "covariance": (ValueError, "declares role 'signal'"),
        "exception": (RuntimeError, "sampler failed"),
    }[failure]
    with pytest.raises(expected_error, match=message):
        actual.run(1)
    assert actual.ledger["counter"].parameters == {"a": 0.0}
    assert block.state.n == 1
    assert actual.covariance is None
    actual.run(1)
    expected.run(1)
    assert actual.ledger["counter"].parameters == expected.ledger["counter"].parameters


# None and "time" both register the block on the arrival (time) grid.
@pytest.mark.parametrize("data_domain", [None, "time"])
def test_covariance_input_and_output_snapshots_are_isolated(observed, data_domain):
    covariance = white_covariance(observed, 1)
    wheel = Wheel(observed, num_frequency_divisions=8)
    wheel.add(FixedNoiseBlock("noise", covariance))
    covariance.blocks.setflags(write=True)
    covariance.A[:] *= 2
    inspected = wheel.covariance
    inspected.blocks.setflags(write=True)
    inspected.A[:] *= 3

    class EditsNoise(Counter):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.blocks.setflags(write=True)
            covariance.A[:] *= 4
            return current_block_state

    wheel.add(EditsNoise(), data_domain=data_domain)
    wheel.run(1)
    np.testing.assert_allclose(wheel.covariance.A, 1 / observed.fs, rtol=1e-12)
    np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 1)


def test_residual_overflow_is_rejected_before_registration(observed):
    # WDM observations are the home grid itself, so nothing is transformed and
    # the overflow happens in the subtraction, as this test means it to.
    data = transform(observed, "wdm", num_frequency_divisions=8)
    data = replace(
        data,
        channel_data={
            ch: np.full_like(x, 1e308) for ch, x in data.channel_data.items()
        },
    )

    wheel = Wheel(data)
    with pytest.raises(ValueError, match="non-finite"):
        wheel.add(
            starting_at(
                Counter(),
                data.block_state({ch: -x for ch, x in data.channel_data.items()}),
            ),
        )
    assert len(wheel.ledger) == 0
    np.testing.assert_array_equal(
        wheel.working_residual.channel_data["A"], data.channel_data["A"]
    )
