"""enchilada.testing: EchoBlock and the check_block conformance helper."""

import re

import numpy as np
import pytest

from conftest import make_observed, white_covariance
from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel
from enchilada.testing import EchoBlock, check_block


class RecordingEcho(EchoBlock):
    """EchoBlock that records the covariance's A variances (or None) per call."""

    def __init__(self, name, seen):
        super().__init__(name)
        self.seen = seen

    def _record(self, covariance):
        self.seen.append(None if covariance is None else covariance.A.copy())

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self._record(covariance)
        return super().initial_sample(conditional_residual, covariance, rng=rng)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self._record(covariance)
        return super().sample(
            conditional_residual, covariance, current_block_state, rng=rng
        )


class FlatNoiseBlock:
    """A noise block publishing white noise of `level` from both methods."""

    def __init__(self, name, level, seen=None):
        self.name = name
        self.level = level
        self.seen = seen

    def _publish(self, conditional_residual, covariance):
        if self.seen is not None:
            self.seen.append(None if covariance is None else covariance.A.copy())
        return conditional_residual.block_state().with_covariance(
            white_covariance(conditional_residual, self.level)
        )

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self._publish(conditional_residual, covariance)

    def state_from_parameters(self, parameters, reference_data):
        return reference_data.block_state().with_covariance(
            white_covariance(reference_data, self.level)
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return self._publish(conditional_residual, covariance)


class TestCheckBlock:
    def test_conforming_signal_block_passes_with_fixed_noise(self, observed):
        check_block(
            block_under_test=EchoBlock(name="echo"),
            observed_data=observed,
            num_cycles=2,
            role="signal",
            noise_blocks=(FixedNoiseBlock("noise", white_covariance(observed, 1.0)),),
            random_seed=0,
        )

    def test_conforming_signal_block_passes_without_noise(self, observed):
        check_block(EchoBlock(name="echo"), observed, role="signal")

    def test_role_is_required(self, observed):
        with pytest.raises(TypeError, match="role"):
            check_block(EchoBlock(name="echo"), observed)

    def test_conforming_in_frequency_domain(self, rng):
        n = 64
        obs = make_observed(
            rng,
            data_domain="frequency",
            channel_data={
                ch: rng.standard_normal(n // 2 + 1) + 0j for ch in ("A", "E", "T")
            },
        )
        check_block(EchoBlock(name="echo"), obs, role="signal")

    def test_non_block_state_return_caught(self, observed):
        class Bad(EchoBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return {"A": np.zeros(1)}  # not a BlockState

        with pytest.raises(TypeError, match="must return a BlockState"):
            check_block(Bad(name="bad"), observed, role="signal")

    def test_off_grid_block_state_caught(self, observed):
        class Drifter(EchoBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return BlockState(
                    template={
                        ch: np.zeros(3) for ch in conditional_residual.channel_names
                    }
                )

        with pytest.raises(ValueError, match="length 3, expected"):
            check_block(Drifter(name="drift"), observed, role="signal")

    def test_conforming_noise_block_passes(self, observed):
        check_block(FlatNoiseBlock("noise", 1.0), observed.to_frequency(), role="noise")

    def test_noise_model_violating_contract_caught(self, observed):
        class BadNoise(EchoBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return conditional_residual.block_state().with_covariance(object())

        with pytest.raises(TypeError, match="Covariance"):
            check_block(BadNoise(name="bad"), observed, role="noise")

    def test_role_violations_are_caught(self, observed):
        with pytest.raises(ValueError, match="registered as a signal block"):
            check_block(FlatNoiseBlock("noise", 1.0), observed, role="signal")
        with pytest.raises(ValueError, match="registered as a noise block"):
            check_block(EchoBlock(name="echo"), observed, role="noise")

    def test_noise_blocks_register_first_and_the_signal_block_receives_their_sum(
        self, observed
    ):
        seen = []
        check_block(
            RecordingEcho("echo", seen),
            observed,
            num_cycles=2,
            role="signal",
            noise_blocks=(
                FixedNoiseBlock("floor", white_covariance(observed, 1.0)),
                FlatNoiseBlock("excess", 2.0),
            ),
        )
        assert len(seen) == 3  # initial_sample and two samples
        for received in seen:
            np.testing.assert_array_equal(received, 3.0)

    def test_a_noise_block_under_test_receives_the_other_noise_blocks(self, observed):
        seen = []
        check_block(
            FlatNoiseBlock("excess", 2.0, seen),
            observed,
            num_cycles=2,
            role="noise",
            noise_blocks=(FixedNoiseBlock("floor", white_covariance(observed, 1.0)),),
        )
        assert len(seen) == 3
        for received in seen:
            np.testing.assert_array_equal(received, 1.0)  # the floor, not itself

    def test_block_without_initial_sample_is_refused(self, observed):
        class SampleOnly:
            name = "sample_only"

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return current_block_state

        with pytest.raises(TypeError, match="does not implement initial_sample"):
            check_block(SampleOnly(), observed, role="signal")

    def test_failing_initial_sample_is_caught(self, observed):
        class BadStart(EchoBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return {"A": np.zeros(1)}  # not a BlockState

        with pytest.raises(TypeError, match="initial_sample must return a BlockState"):
            check_block(BadStart(name="bad"), observed, role="signal")


class TestFixedNoiseBlock:
    def test_passes_check_block_as_the_block_under_test(self, observed):
        check_block(
            FixedNoiseBlock("floor", white_covariance(observed, 1.0)),
            observed,
            role="noise",
        )

    def test_gives_a_campaign_a_known_covariance(self, observed):
        covariance = white_covariance(observed, 7.0)
        wheel = Wheel(observed)
        wheel.add(FixedNoiseBlock("floor", covariance), role="noise")
        np.testing.assert_array_equal(wheel.covariance.A, 7.0)
        wheel.add(EchoBlock("echo"), role="signal")
        wheel.run(2)
        np.testing.assert_array_equal(wheel.covariance.A, 7.0)
        np.testing.assert_array_equal(wheel.ledger["floor"].covariance.A, 7.0)
        assert wheel.ledger["floor"].template is None

    def test_must_be_registered_on_its_covariance_grid(self, observed):
        wheel = Wheel(observed)
        spectral = white_covariance(observed.to_frequency(), 1.0)
        with pytest.raises(
            ValueError, match="floor.initial_sample covariance is on grid"
        ):
            wheel.add(FixedNoiseBlock("floor", spectral), role="noise")
        assert len(wheel.ledger) == 0
        wheel.add(
            FixedNoiseBlock("floor", spectral), role="noise", data_domain="frequency"
        )
        assert wheel.covariance.grid == observed.to_frequency().grid


class TestEchoBlock:
    def test_keeps_sampling_counter_in_local_state(self, observed):
        echo = EchoBlock(name="echo")
        wheel = Wheel(observed)
        wheel.add(echo, role="signal")
        wheel.run(3)
        assert echo.state.num_sample_calls == 3
        published = wheel.ledger["echo"]
        assert published.template is None
        assert published.covariance is None
        assert published.parameters == {}
        assert echo.state.metadata == {}

    def test_contributes_a_zero_signal(self, observed):
        wheel = Wheel(observed)
        wheel.add(EchoBlock(name="echo"), role="signal")
        wheel.run(1)
        for ch in observed.channel_names:
            np.testing.assert_array_equal(
                wheel.residual().channel_data[ch], observed.channel_data[ch]
            )


def test_echo_continuation_belongs_to_each_block_instance(observed, rng):
    echo = EchoBlock(name="echo")
    initial = BlockState()
    updated = echo.sample(
        conditional_residual=observed,
        covariance=None,
        current_block_state=initial,
        rng=rng,
    )
    for published in (initial, updated):
        assert published.template is None
        assert published.covariance is None
        assert published.parameters == {}
    assert echo.state.num_sample_calls == 1
    echo.sample(observed, None, BlockState(), rng=rng)
    assert echo.state.num_sample_calls == 2

    fresh = EchoBlock(name="echo")
    assert fresh.state.num_sample_calls == 0
    restarted = fresh.sample(observed, None, initial, rng=rng)
    assert restarted.template is None
    assert restarted.covariance is None
    assert restarted.parameters == {}
    assert fresh.state.num_sample_calls == 1
    assert echo.state.num_sample_calls == 2
    echo.state.metadata["diagnostic"] = 1
    assert fresh.state.metadata == {}
    start = echo.initial_sample(observed, None, rng=rng)
    assert start.template is None
    assert start.covariance is None
    assert start.parameters == {}
    assert echo.state.num_sample_calls == 2


class _Publishing:
    """Test-block base: both sampling methods publish `self._state(...)`.

    A subclass supplies `name`, `_state(data)` and `state_from_parameters`.
    """

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self._state(conditional_residual)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return self._state(conditional_residual)


class _Amplitude(_Publishing):
    """Publishes a constant template of `amplitude` per channel."""

    name = "amplitude"
    amplitude = 1.0

    def _state(self, data, amplitude=None):
        amplitude = self.amplitude if amplitude is None else amplitude
        return data.block_state(
            {ch: np.full(len(arr), amplitude) for ch, arr in data.channel_data.items()},
            parameters={"amplitude": amplitude},
        )


class TestRoundTrip:
    """check_block rebuilds the block under test from its own parameters."""

    @pytest.mark.parametrize(("num_cycles", "calls"), [(3, 5), (0, 1)])
    def test_rebuilds_after_registration_and_after_every_cycle(
        self, observed, num_cycles, calls
    ):
        class Counting(FlatNoiseBlock):
            def __init__(self, name, level):
                super().__init__(name, level)
                self.calls = 0

            def state_from_parameters(self, parameters, reference_data):
                self.calls += 1
                return super().state_from_parameters(parameters, reference_data)

        block = Counting("noise", 1.0)
        check_block(block, observed, num_cycles=num_cycles, role="noise")
        # step 0, one per cycle, and step 0 again after the run (not with 0 cycles)
        assert block.calls == calls

    def test_a_template_the_parameters_do_not_determine_is_caught(self, observed):
        class Doubling(_Amplitude):
            name = "double"

            def state_from_parameters(self, parameters, reference_data):
                return self._state(reference_data, 2 * parameters["amplitude"])

        with pytest.raises(
            ValueError,
            match=re.escape(
                "double.state_from_parameters does not reproduce the published "
                "template in channel 'A' at step 0: parameters must determine the "
                "template"
            ),
        ):
            check_block(Doubling(), observed, role="signal")

    def test_a_covariance_the_parameters_do_not_determine_is_caught_at_its_step(
        self, observed
    ):
        class Drifting(FlatNoiseBlock):
            """Publishes `level + 1` from its first `sample` on, while the rebuild
            gives `level`."""

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return conditional_residual.block_state().with_covariance(
                    white_covariance(conditional_residual, self.level + 1.0)
                )

        with pytest.raises(ValueError, match="published covariance at step 1"):
            check_block(Drifting("noise", 1.0), observed, role="noise")

    def test_the_rebuilt_state_must_be_a_block_state(self, observed):
        class Wrong(EchoBlock):
            def state_from_parameters(self, parameters, reference_data):
                return {}

        with pytest.raises(
            TypeError, match="state_from_parameters must return a BlockState"
        ):
            check_block(Wrong("wrong"), observed, role="signal")

    @pytest.mark.parametrize(("error", "passes"), [(1e-12, True), (1e-7, False)])
    def test_round_off_is_tolerated_but_a_real_difference_is_not(
        self, observed, error, passes
    ):
        class Scaled(_Publishing):
            name = "thirds"

            def _state(self, data, amplitude=0.3):
                basis = np.linspace(-1.0, 1.0, data.num_time_samples)
                return data.block_state(
                    {ch: amplitude * basis for ch in data.channel_names},
                    parameters={"amplitude": amplitude},
                )

            def state_from_parameters(self, parameters, reference_data):
                return self._state(
                    reference_data, parameters["amplitude"] * (1 + error)
                )

        if passes:
            check_block(Scaled(), observed, role="signal")
        else:
            with pytest.raises(
                ValueError, match="published template in channel 'A' at step 0"
            ):
                check_block(Scaled(), observed, role="signal")

    def test_an_all_zero_channel_tolerates_nothing(self, observed):
        class Silent(_Publishing):
            name = "silent"

            def _state(self, data):
                return data.block_state(
                    {ch: np.zeros(data.num_time_samples) for ch in data.channel_names}
                )

            def state_from_parameters(self, parameters, reference_data):
                return reference_data.block_state(
                    {
                        ch: np.full(reference_data.num_time_samples, 1e-300)
                        for ch in reference_data.channel_names
                    }
                )

        with pytest.raises(
            ValueError, match="published template in channel 'A' at step 0"
        ):
            check_block(Silent(), observed, role="signal")

    @pytest.mark.parametrize(
        ("rebuilt", "passes"), [(1e-12 * (1 + 1e-12), True), (2e-12, False)]
    )
    def test_a_quiet_covariance_channel_is_checked_on_its_own_scale(
        self, observed, rebuilt, passes
    ):
        n = observed.num_time_samples

        class QuietT(_Publishing):
            name = "quiet_t"

            def _state(self, data, quiet=1e-12):
                return data.block_state().with_covariance(
                    Covariance.like(
                        data,
                        {"A": np.ones(n), "E": np.ones(n), "T": np.full(n, quiet)},
                    )
                )

            def state_from_parameters(self, parameters, reference_data):
                return self._state(reference_data, rebuilt)

        if passes:
            check_block(QuietT(), observed, role="noise")
        else:
            with pytest.raises(ValueError, match="published covariance at step 0"):
                check_block(QuietT(), observed, role="noise")

    def test_a_rebuild_that_reads_the_block_state_is_caught(self, observed):
        class Remembering(_Amplitude):
            name = "remember"

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                self.amplitude += 1.0
                return self._state(conditional_residual)

            def state_from_parameters(self, parameters, reference_data):
                return self._state(reference_data)  # reads self, not parameters

        with pytest.raises(ValueError, match="rebuilt again after 2 cycles"):
            check_block(Remembering(), observed, num_cycles=2, role="signal")

    def test_a_rebuild_that_reads_channel_values_is_caught(self, observed):
        # a zero observation would equal the zeroed reference
        assert np.any(observed.channel_data["A"])

        class Copying(_Publishing):
            name = "copy"

            def _state(self, data):
                return data.block_state(
                    {ch: np.array(arr) for ch, arr in data.channel_data.items()}
                )

            def state_from_parameters(self, parameters, reference_data):
                return self._state(reference_data)  # copies the reference values

        with pytest.raises(
            ValueError, match="published template in channel 'A' at step 0"
        ):
            check_block(Copying(), observed, role="signal")

    def test_a_rebuild_with_the_wrong_channels_says_which(self, observed):
        class NoTemplate(_Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                return BlockState()

        with pytest.raises(
            ValueError,
            match=re.escape("rebuilt channels [], published ['A', 'E', 'T']"),
        ):
            check_block(NoTemplate(), observed, role="signal")

    def test_a_rebuild_without_the_covariance_says_so(self, observed):
        class NoCovariance(FlatNoiseBlock):
            def state_from_parameters(self, parameters, reference_data):
                return BlockState()

        with pytest.raises(ValueError, match="the rebuild has none"):
            check_block(NoCovariance("noise", 1.0), observed, role="noise")

    def test_a_rebuilt_template_off_the_grid_is_caught(self, observed):
        class Complex(_Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                return BlockState(
                    template={
                        ch: np.zeros(reference_data.num_time_samples, dtype=complex)
                        for ch in reference_data.channel_names
                    }
                )

        with pytest.raises(TypeError, match="complex"):
            check_block(Complex(), observed, role="signal")

    def test_the_rebuild_receives_file_typed_parameters(self, observed):
        class Typed(_Amplitude):
            def _state(self, data, amplitude=0.5, weights=(1.0, 2.0)):
                level = amplitude * float(np.sum(weights))
                return data.block_state(
                    {
                        ch: np.full(len(arr), level)
                        for ch, arr in data.channel_data.items()
                    },
                    parameters={"amplitude": amplitude, "weights": list(weights)},
                )

            def state_from_parameters(self, parameters, reference_data):
                assert isinstance(parameters["amplitude"], np.floating)
                assert isinstance(parameters["weights"], np.ndarray)
                return self._state(
                    reference_data, parameters["amplitude"], parameters["weights"]
                )

        check_block(Typed(), observed, role="signal")

    def test_a_rebuild_that_mutates_its_parameters_is_harmless(self, observed):
        class Popping(_Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                amplitude = parameters.pop("amplitude")
                return self._state(reference_data, amplitude)

        check_block(Popping(), observed, num_cycles=2, role="signal")


class TestRoundTripCovarianceBranches:
    def test_a_rebuild_with_a_covariance_the_published_state_lacks(self, observed):
        class Extra(_Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                return self._state(
                    reference_data, parameters["amplitude"]
                ).with_covariance(white_covariance(reference_data, 1.0))

        with pytest.raises(
            ValueError, match="the rebuild has one but the published state has none"
        ):
            check_block(Extra(), observed, role="signal")

    def test_a_rebuilt_covariance_with_other_channels(self, observed, rng):
        two = make_observed(rng, channel_names=("A", "E"))

        class TwoChannels(FlatNoiseBlock):
            def state_from_parameters(self, parameters, reference_data):
                return BlockState().with_covariance(white_covariance(two, self.level))

        with pytest.raises(ValueError, match=r"channels \('A', 'E'\) differ"):
            check_block(TwoChannels("noise", 1.0), observed, role="noise")

    def test_a_rebuilt_covariance_on_another_grid(self, observed):
        spectral = observed.to_frequency()

        class OtherGrid(FlatNoiseBlock):
            def state_from_parameters(self, parameters, reference_data):
                return BlockState().with_covariance(
                    white_covariance(spectral, self.level)
                )

        with pytest.raises(ValueError, match="grid .* differs from the published"):
            check_block(OtherGrid("noise", 1.0), observed, role="noise")

    def test_a_rebuild_cannot_write_into_its_reference(self, observed):
        class Writer(_Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                reference_data.channel_data["A"][0] = 1.0
                return self._state(reference_data, parameters["amplitude"])

        with pytest.raises(ValueError, match="read-only"):
            check_block(Writer(), observed, role="signal")
