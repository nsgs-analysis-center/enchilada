"""enchilada.testing: EchoBlock and the check_block conformance helper."""

import numpy as np
import pytest

from conftest import make_observed, white_covariance
from enchilada import BlockState, FixedNoiseBlock, Wheel, transform
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

    role = "noise"

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

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return self._publish(conditional_residual, covariance)


class TestCheckBlock:
    def test_conforming_signal_block_passes_with_fixed_noise(self, observed):
        check_block(
            block_under_test=EchoBlock(name="echo"),
            observed_data=observed,
            num_cycles=2,
            num_frequency_divisions=8,
            noise_blocks=(FixedNoiseBlock("noise", white_covariance(observed, 1.0)),),
            random_seed=0,
        )

    def test_conforming_signal_block_passes_without_noise(self, observed):
        check_block(EchoBlock(name="echo"), observed, num_frequency_divisions=8)

    def test_a_signal_block_among_the_noise_blocks_is_refused(self, observed):
        with pytest.raises(ValueError, match="noise_blocks entry 'echo' declares role"):
            check_block(
                EchoBlock(name="under_test"),
                observed,
                noise_blocks=(EchoBlock(name="echo"),),
                num_frequency_divisions=8,
            )
        with pytest.raises(TypeError, match="role"):
            check_block(EchoBlock(name="echo"), observed, **{"role": "signal"})

    def test_conforming_in_frequency_domain(self, rng):
        n = 64
        obs = make_observed(
            rng,
            data_domain="frequency",
            channel_data={
                ch: rng.standard_normal(n // 2 + 1) + 0j for ch in ("A", "E", "T")
            },
        )
        check_block(EchoBlock(name="echo"), obs, num_frequency_divisions=8)

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
            check_block(Bad(name="bad"), observed, num_frequency_divisions=8)

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
            check_block(Drifter(name="drift"), observed, num_frequency_divisions=8)

    def test_conforming_noise_block_passes(self, observed):
        check_block(
            FlatNoiseBlock("noise", 1.0),
            observed.to_frequency(),
            num_frequency_divisions=8,
        )

    def test_noise_model_violating_contract_caught(self, observed):
        class BadNoise(EchoBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return conditional_residual.block_state().with_covariance(object())

        with pytest.raises(TypeError, match="Covariance"):
            check_block(BadNoise(name="bad"), observed, num_frequency_divisions=8)

    def test_role_violations_are_caught(self, observed):
        turncoat = FlatNoiseBlock("noise", 1.0)
        turncoat.role = "signal"
        with pytest.raises(ValueError, match="declares role 'signal'"):
            check_block(turncoat, observed, num_frequency_divisions=8)
        echo = EchoBlock(name="echo")
        echo.role = "noise"
        with pytest.raises(ValueError, match="declares role 'noise'"):
            check_block(echo, observed, num_frequency_divisions=8)

    def test_noise_blocks_register_first_and_the_signal_block_receives_their_sum(
        self, observed
    ):
        seen = []
        check_block(
            RecordingEcho("echo", seen),
            observed,
            num_cycles=2,
            num_frequency_divisions=8,
            noise_blocks=(
                FixedNoiseBlock("floor", white_covariance(observed, 1.0)),
                FlatNoiseBlock("excess", 2.0),
            ),
        )
        assert len(seen) == 3  # initial_sample and two samples
        for received in seen:
            np.testing.assert_allclose(received, 3.0)

    def test_a_noise_block_under_test_receives_the_other_noise_blocks(self, observed):
        seen = []
        check_block(
            FlatNoiseBlock("excess", 2.0, seen),
            observed,
            num_cycles=2,
            num_frequency_divisions=8,
            noise_blocks=(FixedNoiseBlock("floor", white_covariance(observed, 1.0)),),
        )
        assert len(seen) == 3
        for received in seen:
            np.testing.assert_allclose(received, 1.0)  # the floor, not itself

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
            check_block(SampleOnly(), observed, num_frequency_divisions=8)

    def test_failing_initial_sample_is_caught(self, observed):
        class BadStart(EchoBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return {"A": np.zeros(1)}  # not a BlockState

        with pytest.raises(TypeError, match="initial_sample must return a BlockState"):
            check_block(BadStart(name="bad"), observed, num_frequency_divisions=8)


class TestFixedNoiseBlock:
    def test_passes_check_block_as_the_block_under_test(self, observed):
        check_block(
            FixedNoiseBlock("floor", white_covariance(observed, 1.0)),
            observed,
            num_frequency_divisions=8,
        )

    def test_gives_a_campaign_a_known_covariance(self, observed):
        covariance = white_covariance(observed, 7.0)
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(FixedNoiseBlock("floor", covariance))
        np.testing.assert_allclose(wheel.covariance.A, 7.0 / observed.fs)
        wheel.add(EchoBlock("echo"))
        wheel.run(2)
        np.testing.assert_allclose(wheel.covariance.A, 7.0 / observed.fs)
        np.testing.assert_array_equal(wheel.ledger["floor"].covariance.A, 7.0)
        assert wheel.ledger["floor"].template is None

    def test_must_be_registered_on_its_covariance_grid(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=8)
        spectral = white_covariance(observed.to_frequency(), 1.0)
        with pytest.raises(
            ValueError, match="floor.initial_sample covariance is on grid"
        ):
            wheel.add(FixedNoiseBlock("floor", spectral))
        assert len(wheel.ledger) == 0
        wheel.add(FixedNoiseBlock("floor", spectral), data_domain="frequency")
        assert wheel.covariance.grid == wheel.home_grid
        np.testing.assert_allclose(
            transform(wheel.covariance, "frequency").A, spectral.A, rtol=1e-12
        )


class TestEchoBlock:
    def test_keeps_sampling_counter_in_local_state(self, observed):
        echo = EchoBlock(name="echo")
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(echo)
        wheel.run(3)
        assert echo.state.num_sample_calls == 3
        published = wheel.ledger["echo"]
        assert published.template is None
        assert published.covariance is None
        assert published.parameters == {}
        assert echo.state.metadata == {}

    def test_contributes_a_zero_signal(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(EchoBlock(name="echo"))
        wheel.run(1)
        for ch in observed.channel_names:
            np.testing.assert_allclose(
                wheel.residual().channel_data[ch],
                observed.channel_data[ch],
                rtol=1e-12,
                atol=1e-12,
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
