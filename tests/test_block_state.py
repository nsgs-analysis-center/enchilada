"""Optional physical estimates are published atomically across sampling calls."""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from conftest import const_block_state, make_observed, starting_at, white_covariance
from enchilada import BlockState, Wheel


def test_wdm_result_factory_accepts_a_full_grid_and_zero_factory_matches_grid(
    observed,
):
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    observed = replace(
        observed,
        data_domain="wdm",
        channel_data={ch: np.zeros(grid.array_shape) for ch in observed.channel_names},
    )
    signal = {ch: np.ones(grid.array_shape) for ch in observed.channel_names}
    result = observed.block_state(signal, parameters={"amplitude": 1})
    assert result.parameters == {"amplitude": 1}
    np.testing.assert_array_equal(result.template["A"], signal["A"])
    zero = observed.zero_block_state()
    assert zero.template["A"].shape == grid.array_shape
    np.testing.assert_array_equal(zero.template["A"], 0.0)
    assert observed.block_state().template is None


@pytest.mark.parametrize("invalid", ["shape", "complex"])
def test_wdm_result_factory_rejects_incompatible_signal(observed, invalid):
    from enchilada.domains import WDMGrid

    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    observed = replace(
        observed,
        data_domain="wdm",
        channel_data={ch: np.zeros(grid.array_shape) for ch in observed.channel_names},
    )
    signal = {ch: np.zeros(grid.array_shape) for ch in observed.channel_names}
    error = ValueError
    if invalid == "shape":
        signal["A"] = np.zeros((8, 8))
    else:
        signal["A"] = signal["A"].astype(complex)
        error = TypeError
    with pytest.raises(error, match="A"):
        observed.block_state(signal)


def test_one_dimensional_domains_reject_two_dimensional_block_states(observed):
    for data in (observed, observed.to_frequency()):
        signal = {ch: arr[:, None] for ch, arr in data.channel_data.items()}
        with pytest.raises((TypeError, ValueError), match="1-D"):
            data.block_state(signal)


@pytest.mark.parametrize(
    "num_time_samples,index,endpoint",
    [(1, 0, "DC"), (4, 0, "DC"), (4, -1, "Nyquist"), (5, 0, "DC")],
)
def test_result_factory_rejects_imaginary_rfft_endpoints(
    observed, num_time_samples, index, endpoint
):
    observed = replace(
        observed,
        num_time_samples=num_time_samples,
        channel_data={ch: np.zeros(num_time_samples) for ch in observed.channel_names},
    ).to_frequency()
    signal = {ch: samples.copy() for ch, samples in observed.channel_data.items()}
    signal["A"][index] = 1j
    with pytest.raises(ValueError, match=rf"{endpoint}.*real"):
        observed.block_state(template=signal)


@pytest.mark.parametrize("num_time_samples", [1, 4, 5])
def test_valid_rfft_signal_preserves_all_allowed_components(observed, num_time_samples):
    observed = replace(
        observed,
        num_time_samples=num_time_samples,
        channel_data={ch: np.zeros(num_time_samples) for ch in observed.channel_names},
    ).to_frequency()
    signal = {ch: samples.copy() for ch, samples in observed.channel_data.items()}
    signal["A"][0] = 1.0
    if num_time_samples > 1:
        signal["A"][-1] = 2.0 if num_time_samples % 2 == 0 else 2.0 + 3j

    class Signal:
        name = "signal"

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    wheel = Wheel(observed)
    wheel.add(
        starting_at(Signal(), observed.block_state(template=signal)), role="signal"
    )
    wheel.run(1)
    np.testing.assert_array_equal(wheel.contribution("signal")["A"], signal["A"])
    np.testing.assert_array_equal(
        wheel.working_residual.channel_data["A"], -signal["A"]
    )


class NoiseOnly:
    name = "noise"

    @dataclass
    class State:
        steps: int = 0
        draw: float = 0.0

    def __init__(self):
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.block_state(
            covariance=white_covariance(conditional_residual, 1),
            parameters={"variance": 1},
        )

    def state_from_parameters(self, parameters, reference_data):
        return reference_data.block_state(
            covariance=white_covariance(reference_data, parameters["variance"]),
            parameters=dict(parameters),
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.steps += 1
        self.state.draw = rng.normal()
        variance = current_block_state.parameters["variance"] + 1
        return conditional_residual.block_state(
            covariance=white_covariance(conditional_residual, variance),
            parameters={"variance": variance},
        )


class SignalOnly:
    name = "signal"

    @dataclass
    class State:
        steps: int = 0
        draw: float = 0.0

    def __init__(self):
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.block_state(
            template={
                ch: np.full_like(samples, 0.5)
                for ch, samples in conditional_residual.channel_data.items()
            },
            parameters={"amplitude": 0.5},
        )

    def state_from_parameters(self, parameters, reference_data):
        return reference_data.block_state(
            template={
                ch: np.full_like(samples, parameters["amplitude"])
                for ch, samples in reference_data.channel_data.items()
            },
            parameters=dict(parameters),
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.steps += 1
        self.state.draw = rng.normal()
        amplitude = current_block_state.parameters["amplitude"] + 0.5
        return conditional_residual.block_state(
            template={
                ch: np.full_like(samples, amplitude)
                for ch, samples in conditional_residual.channel_data.items()
            },
            parameters={"amplitude": amplitude},
        )


@pytest.mark.parametrize(
    "num_time_samples,index,endpoint",
    [(1, 0, "DC"), (4, 0, "DC"), (4, -1, "Nyquist"), (5, 0, "DC")],
)
def test_rejected_rfft_signal_preserves_estimates_state_and_rng(
    observed, num_time_samples, index, endpoint
):
    observed = replace(
        observed,
        num_time_samples=num_time_samples,
        channel_data={ch: np.zeros(num_time_samples) for ch in observed.channel_names},
    ).to_frequency()
    fail = [True]

    class InvalidOnce(SignalOnly):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            result = super().sample(
                conditional_residual, covariance, current_block_state, rng=rng
            )
            if fail[0]:
                fail[0] = False
                signal = {ch: samples.copy() for ch, samples in result.template.items()}
                signal["A"][index] += 1j
                return replace(result, template=signal)
            return result

    actual, expected = Wheel(observed, random_seed=8), Wheel(observed, random_seed=8)
    actual_block, expected_block = InvalidOnce(), SignalOnly()
    actual.add(actual_block, role="signal")
    expected.add(expected_block, role="signal")
    with pytest.raises(ValueError, match=rf"{endpoint}.*real"):
        actual.run(1)
    assert actual_block.state.steps == 1
    assert expected_block.state.steps == 0
    assert actual.ledger["signal"].parameters == expected.ledger["signal"].parameters
    assert actual.ledger["signal"].parameters == {"amplitude": 0.5}
    assert actual.covariance is None
    np.testing.assert_array_equal(actual.contribution("signal")["A"], 0.5)
    np.testing.assert_array_equal(actual.working_residual.channel_data["A"], -0.5)

    actual.run(1)
    expected.run(1)
    assert actual_block.state.steps == 2
    assert expected_block.state.steps == 1
    assert actual_block.state.draw == expected_block.state.draw
    assert actual.ledger["signal"].parameters == expected.ledger["signal"].parameters
    np.testing.assert_array_equal(actual.working_residual.channel_data["A"], -1.0)


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_noise_only_result_preserves_data_without_signal_arrays(observed, domain):
    if domain == "frequency":
        observed = observed.to_frequency()
    wheel = Wheel(observed, random_seed=3)
    block = NoiseOnly()
    wheel.add(block, role="noise")
    wheel.run(2)

    result = wheel.ledger["noise"]
    assert result.template is None
    assert result.parameters == {"variance": 3}
    assert block.state.steps == 2
    np.testing.assert_array_equal(result.covariance.A, white_covariance(observed, 3).A)
    np.testing.assert_array_equal(wheel.covariance.A, white_covariance(observed, 3).A)
    contribution = wheel.contribution("noise")
    assert set(contribution) == set(observed.channel_names)
    for channel, samples in observed.channel_data.items():
        np.testing.assert_array_equal(
            wheel.working_residual.channel_data[channel], samples
        )
        np.testing.assert_array_equal(contribution[channel], np.zeros_like(samples))
        assert contribution[channel].dtype == samples.dtype
    contribution["A"][:] = 100
    np.testing.assert_array_equal(wheel.contribution("noise")["A"], 0)
    assert wheel.ledger["noise"].template is None


@pytest.mark.parametrize("domain", ["time", "frequency"])
def test_absent_signal_removes_previous_contribution_and_can_return(observed, domain):
    if domain == "frequency":
        observed = observed.to_frequency()

    def population_start(data):
        return data.block_state(
            template={
                ch: np.full_like(samples, 0.25)
                for ch, samples in data.channel_data.items()
            },
            parameters={"num_sources": 1},
        )

    class Population:
        name = "population"

        @dataclass
        class State:
            steps: int = 0

        def __init__(self):
            self.state = self.State()

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return population_start(conditional_residual)

        def state_from_parameters(self, parameters, reference_data):
            # Stub: no test rebuilds this block's state.
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            for channel, samples in observed.channel_data.items():
                np.testing.assert_allclose(
                    conditional_residual.channel_data[channel], samples
                )
            self.state.steps += 1
            if self.state.steps == 1:
                return BlockState(parameters={"num_sources": 0})
            return population_start(conditional_residual)

    wheel = Wheel(observed)
    wheel.add(NoiseOnly(), role="noise")
    population = Population()
    wheel.add(population, role="signal")
    np.testing.assert_allclose(
        wheel.working_residual.channel_data["A"], observed.channel_data["A"] - 0.25
    )
    wheel.run(1)
    assert wheel.ledger["population"].template is None
    assert wheel.ledger["population"].parameters == {"num_sources": 0}
    np.testing.assert_allclose(
        wheel.working_residual.channel_data["A"], observed.channel_data["A"]
    )
    np.testing.assert_array_equal(wheel.covariance.A, white_covariance(observed, 2).A)
    wheel.run(1)
    assert wheel.ledger["population"].parameters == {"num_sources": 1}
    assert population.state.steps == 2
    np.testing.assert_allclose(
        wheel.working_residual.channel_data["A"], observed.channel_data["A"] - 0.25
    )


class Baseline:
    name = "baseline"

    def state_from_parameters(self, parameters, reference_data):
        # Stub: no test rebuilds this block's state.
        return BlockState(parameters=dict(parameters))

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return current_block_state


@pytest.mark.parametrize("failure", ["omitted", "wrong_grid", "template"])
def test_rejected_noise_result_preserves_covariance_state_and_rng(observed, failure):
    fail = [True]

    class FailsOnce(NoiseOnly):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            result = super().sample(
                conditional_residual, covariance, current_block_state, rng=rng
            )
            if fail[0]:
                fail[0] = False
                if failure == "omitted":
                    return replace(result, covariance=None)
                if failure == "template":
                    return replace(
                        result,
                        template={
                            ch: np.full_like(samples, 0.5)
                            for ch, samples in conditional_residual.channel_data.items()
                        },
                    )
                wrong_grid = make_observed(
                    np.random.default_rng(0), num_time_samples=32
                )
                return replace(result, covariance=white_covariance(wrong_grid, 2))
            return result

    actual, expected = Wheel(observed, random_seed=8), Wheel(observed, random_seed=8)
    actual_block, expected_block = FailsOnce(), NoiseOnly()
    for wheel, block in ((actual, actual_block), (expected, expected_block)):
        # A constant signal gives the residual a non-zero baseline to preserve.
        wheel.add(
            starting_at(Baseline(), const_block_state(observed, 0.5)), role="signal"
        )
        wheel.add(block, role="noise")
    error = {
        "omitted": (
            "must return a covariance because 'noise' is registered as a noise block"
        ),
        "template": "returned a template but 'noise' is registered as a noise block",
        "wrong_grid": "grid",
    }[failure]
    with pytest.raises(ValueError, match=error):
        actual.run(1)
    assert actual_block.state.steps == 1
    assert expected_block.state.steps == 0
    assert actual.ledger["noise"].parameters == expected.ledger["noise"].parameters
    np.testing.assert_array_equal(actual.covariance.A, white_covariance(observed, 1).A)
    np.testing.assert_array_equal(actual.contribution("noise")["A"], 0.0)
    np.testing.assert_array_equal(actual.contribution("baseline")["A"], 0.5)
    np.testing.assert_allclose(
        actual.working_residual.channel_data["A"], observed.channel_data["A"] - 0.5
    )
    actual.run(1)
    expected.run(1)
    assert actual_block.state.steps == 2
    assert expected_block.state.steps == 1
    assert actual_block.state.draw == expected_block.state.draw
    assert actual.ledger["noise"].parameters == expected.ledger["noise"].parameters
    assert actual.ledger["noise"].template is None
    np.testing.assert_array_equal(actual.covariance.A, white_covariance(observed, 2).A)
    np.testing.assert_allclose(
        actual.working_residual.channel_data["A"], observed.channel_data["A"] - 0.5
    )
