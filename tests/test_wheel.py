"""Wheel: the block-state ledger, the data-minus-others handoff, and its guards."""

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from conftest import const_block_state, make_observed, starting_at, white_covariance
from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel
from enchilada.testing import EchoBlock


class ConstBlock:
    """Returns a block state with the same signal value in every sample.

    The residual it is handed is already the data minus every other block; a
    real block would fit it. This one ignores it and returns its (constant)
    template, so its ledger entry carries that constant signal.
    """

    @dataclass
    class State:
        updates: int = 0
        metadata: dict = field(default_factory=lambda: {"model": "constant"})

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return replace(
            self._block_state(conditional_residual), parameters={"value": self.value}
        )

    def state_from_parameters(self, parameters, reference_data):
        return replace(self._block_state(reference_data), parameters=dict(parameters))

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.updates += 1
        return replace(
            current_block_state,
            template=self._block_state(conditional_residual).template,
        )

    def _block_state(self, residual):
        return const_block_state(residual, self.value)


class LevelNoiseBlock:
    """A covariance level driven by a sampling-call counter."""

    @dataclass
    class State:
        updates: int = 0

    def __init__(self, name):
        self.name = name
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.block_state().with_covariance(
            white_covariance(conditional_residual, 1.0)
        )

    def state_from_parameters(self, parameters, reference_data):
        return reference_data.block_state().with_covariance(
            white_covariance(reference_data, float(self.state.updates + 1))
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.updates += 1
        return conditional_residual.block_state().with_covariance(
            white_covariance(conditional_residual, float(self.state.updates + 1))
        )


class TestLedger:
    def test_full_residual_subtracts_every_signal_template(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(observed_data=obs)
        wheel.add(block_to_register=ConstBlock("a", 1.0), role="signal")
        wheel.add(ConstBlock("b", 10.0), role="signal")
        wheel.add(ConstBlock("c", 100.0), role="signal")
        wheel.run(num_cycles=3)
        full = wheel.residual()
        for ch in obs.channel_names:
            np.testing.assert_allclose(
                full.channel_data[ch], obs.channel_data[ch] - 111.0
            )

    def test_exclude_leaves_that_block_in(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstBlock("a", 1.0), role="signal")
        wheel.add(ConstBlock("b", 10.0), role="signal")
        wheel.add(ConstBlock("c", 100.0), role="signal")
        wheel.run(2)
        # residual(exclude_block_name=b) = data minus a and c (110), b left in
        seen = wheel.residual(exclude_block_name="b")
        for ch in obs.channel_names:
            np.testing.assert_allclose(
                seen.channel_data[ch], obs.channel_data[ch] - 101.0
            )

    def test_contribution_is_the_returned_signal_template(self, rng):
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstBlock("a", 7.0), role="signal")
        wheel.run(2)
        for ch in obs.channel_names:
            np.testing.assert_allclose(wheel.contribution(block_name="a")[ch], 7.0)

    def test_a_template_built_in_place_is_accepted(self, rng):
        # a block may overwrite the arrays it was handed with its template and
        # return the same object: the Wheel handed it copies, so the pristine
        # residual is still there to compare against, and observed is untouched
        obs = make_observed(rng)

        class InPlace:
            name = "ip"

            def initial_sample(self, conditional_residual, covariance, *, rng):
                return BlockState()

            def state_from_parameters(self, parameters, reference_data):
                # Not rebuilt: this test block's template is not a function of its
                # parameters.
                return BlockState(parameters=dict(parameters))

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                for ch in conditional_residual.channel_data:
                    conditional_residual.channel_data[ch][:] = (
                        3.0  # overwrite the handed arrays
                    )
                return conditional_residual.block_state(
                    conditional_residual.channel_data
                )  # ...and claim them

        wheel = Wheel(obs)
        wheel.add(InPlace(), role="signal")
        wheel.run(2)
        for ch in obs.channel_names:
            np.testing.assert_allclose(wheel.contribution("ip")[ch], 3.0)
            np.testing.assert_allclose(
                wheel.residual().channel_data[ch], obs.channel_data[ch] - 3.0
            )

    def test_unknown_exclude_or_contribution_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 1.0), role="signal")
        with pytest.raises(ValueError, match="unknown block"):
            wheel.residual(exclude_block_name="ghost")
        with pytest.raises(ValueError, match="unknown block"):
            wheel.contribution("ghost")

    def test_observed_stays_pristine(self, rng):
        obs = make_observed(rng)
        snapshot = {ch: arr.copy() for ch, arr in obs.channel_data.items()}

        class Mutator(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                # Mutate the data handed to this call.
                conditional_residual.channel_data[
                    next(iter(conditional_residual.channel_data))
                ][:] = -999.0
                return super().sample(
                    conditional_residual,
                    covariance,
                    current_block_state,
                    rng=rng,
                )

        wheel = Wheel(obs)
        wheel.add(Mutator("mut", 0.0), role="signal")
        wheel.run(2)
        for ch, arr in snapshot.items():
            np.testing.assert_array_equal(
                obs.channel_data[ch], arr
            )  # observed untouched

    def test_sampler_continuation_lives_on_block_and_estimates_live_in_ledger(
        self, observed
    ):
        block = ConstBlock("a", 1.0)
        state = block.state
        wheel = Wheel(observed)
        wheel.add(block, role="signal")
        assert block.state.updates == 0
        wheel.run(4)
        accepted = wheel.ledger["a"]
        assert isinstance(accepted, BlockState)
        assert accepted.parameters == {"value": 1.0}
        assert block.state is state
        assert block.state.updates == 4
        assert block.state.metadata == {"model": "constant"}
        assert block.name == "a"
        assert block.value == 1.0

    def test_zero_blocks_is_a_noop(self, observed):
        wheel = Wheel(observed)
        wheel.run(5)
        for ch in observed.channel_names:
            np.testing.assert_array_equal(
                wheel.residual().channel_data[ch], observed.channel_data[ch]
            )

    def test_run_rejects_bad_turn_counts(self, observed):
        wheel = Wheel(observed)
        with pytest.raises(ValueError, match="num_cycles"):
            wheel.run(-1)
        with pytest.raises(ValueError, match="num_cycles"):
            wheel.run(True)  # bool is not a cycle count

    def test_run_accepts_numpy_integers(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 1.0), role="signal")
        wheel.run(np.int64(2))


class TestNoAddBack:
    def test_each_block_sees_data_minus_others(self, rng):
        # with two non-trivial blocks, each must be handed the data minus the
        # OTHER (never itself); record what each sees at sampling time.
        obs = make_observed(rng)
        seen = {}

        class Recorder(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                seen[self.name] = {
                    ch: conditional_residual.channel_data[ch].copy()
                    for ch in conditional_residual.channel_data
                }
                return super().sample(
                    conditional_residual,
                    covariance,
                    current_block_state,
                    rng=rng,
                )

        wheel = Wheel(obs)
        wheel.add(Recorder("a", 2.0), role="signal")
        wheel.add(Recorder("b", 5.0), role="signal")
        wheel.run(1)
        # a sees data minus b (5); b sees data minus a (2)
        for ch in obs.channel_names:
            np.testing.assert_allclose(seen["a"][ch], obs.channel_data[ch] - 5.0)
            np.testing.assert_allclose(seen["b"][ch], obs.channel_data[ch] - 2.0)

    def test_samples_see_latest_signals_in_registration_order(self, observed):
        seen = []

        class Growing(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                seen.append((self.name, conditional_residual.channel_data["A"].copy()))
                self.state.updates += 1
                updates = self.state.updates
                return replace(
                    current_block_state,
                    template=const_block_state(
                        conditional_residual, self.value + updates
                    ).template,
                )

        wheel = Wheel(observed)
        wheel.add(Growing("a", 2.0), role="signal")
        wheel.add(Growing("b", 5.0), role="signal")
        wheel.run(2)
        assert [name for name, _ in seen] == ["a", "b", "a", "b"]
        for (_, handed), subtracted in zip(seen, [5.0, 3.0, 6.0, 4.0], strict=True):
            np.testing.assert_allclose(handed, observed.channel_data["A"] - subtracted)


class TestAtomicRegistration:
    def test_registration_calls_initial_sample_and_nothing_else(self, observed):
        calls = []

        class Recorder(ConstBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                calls.append("initial_sample")
                return super().initial_sample(conditional_residual, covariance, rng=rng)

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                calls.append("sample")
                return super().sample(
                    conditional_residual,
                    covariance,
                    current_block_state,
                    rng=rng,
                )

        wheel = Wheel(observed)
        block = Recorder("rec", 2.0)
        wheel.add(block, role="signal")
        assert calls == ["initial_sample"]
        assert block.state.updates == 0
        np.testing.assert_array_equal(wheel.contribution("rec")["A"], 2.0)
        wheel.run(2)
        assert calls == ["initial_sample", "sample", "sample"]

    def test_duplicate_name_rejected(self, observed):
        wheel = Wheel(observed)
        wheel.add(EchoBlock(name="x"), role="signal")
        with pytest.raises(ValueError, match="already registered"):
            wheel.add(EchoBlock(name="x"), role="signal")

    def test_empty_name_rejected(self, observed):
        with pytest.raises(ValueError, match="non-empty"):
            Wheel(observed).add(EchoBlock(name=""), role="signal")

    def test_invalid_start_leaves_wheel_untouched(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("ok", 1.0), role="signal")
        with pytest.raises(
            TypeError, match="bad.initial_sample must return a BlockState"
        ):
            wheel.add(
                starting_at(ConstBlock("bad", 0.0), "not a result"), role="signal"
            )
        with pytest.raises(ValueError, match="unknown block"):
            wheel.contribution("bad")  # not registered
        wheel.contribution("ok")  # the good one still is
        wheel.run(1)  # still healthy


class TestReturnedBlockStateValidation:
    @pytest.mark.parametrize("return_input_data", [False, True])
    def test_non_block_state_return_named(self, observed, return_input_data):
        class Bad(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return conditional_residual if return_input_data else {"A": np.zeros(1)}

        wheel = Wheel(observed)
        wheel.add(Bad("bad", 0.0), role="signal")
        with pytest.raises(TypeError, match="bad.sample must return a BlockState"):
            wheel.run(1)

    def test_bad_tdi_shape_rejected(self, observed):
        class Drifter(ConstBlock):
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
                        ch: np.zeros(2) for ch in conditional_residual.channel_names
                    }
                )

        wheel = Wheel(observed)
        wheel.add(Drifter("drift", 0.0), role="signal")
        with pytest.raises(ValueError, match="length 2, expected"):
            wheel.run(1)

    def test_missing_channel_rejected(self, observed):
        class Partial(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return BlockState(
                    template={"A": np.zeros(conditional_residual.num_time_samples)}
                )

        wheel = Wheel(observed)
        wheel.add(Partial("partial", 0.0), role="signal")
        with pytest.raises(ValueError, match="must match the run's channels"):
            wheel.run(1)


class TestExplicitNoise:
    def test_covariance_refreshes_each_cycle(self, observed):
        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"), role="noise")
        np.testing.assert_array_equal(wheel.covariance.A, 1.0)
        wheel.run(2)
        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
        np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 3.0)

    def test_noise_block_contributes_zero_without_signal(self, observed):
        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"), role="noise")
        wheel.run(1)
        assert wheel.ledger["noise"].template is None
        for ch in observed.channel_names:
            np.testing.assert_array_equal(
                wheel.contribution("noise")[ch],
                np.zeros_like(observed.channel_data[ch]),
            )

    def test_initial_sample_receives_the_noise_published_at_registration(
        self, observed
    ):
        seen = {}

        class Reader(ConstBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                seen["covariance"] = covariance
                return super().initial_sample(conditional_residual, covariance, rng=rng)

        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"), role="noise")
        wheel.add(Reader("rec", 1.0), role="signal")
        assert isinstance(seen["covariance"], Covariance)
        np.testing.assert_array_equal(seen["covariance"].A, 1.0)

    def test_sample_sees_current_noise(self, observed):
        levels = []

        class Recorder(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                levels.append(float(covariance.A[0]))
                return super().sample(
                    conditional_residual,
                    covariance,
                    current_block_state,
                    rng=rng,
                )

        wheel = Wheel(observed)
        wheel.add(LevelNoiseBlock("noise"), role="noise")  # samples first each cycle
        wheel.add(Recorder("rec", 0.0), role="signal")
        wheel.run(3)
        assert levels == [2.0, 3.0, 4.0]

    def test_copied_covariance_arrays_stay_read_only(self, observed):
        seen = []

        class Inspector(LevelNoiseBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                seen.append(current_block_state.covariance.A)
                return super().sample(
                    conditional_residual,
                    covariance,
                    current_block_state,
                    rng=rng,
                )

        wheel = Wheel(observed)
        wheel.add(Inspector("noise"), role="noise")
        wheel.run(2)
        assert len(seen) == 2
        assert not any(arr.flags.writeable for arr in seen)
        for state in (wheel.ledger["noise"], wheel.ledger.snapshot()["noise"]):
            assert not state.covariance.A.flags.writeable

    def test_fixed_noise_block_supplies_a_known_covariance(self, observed):
        spectrum = observed.to_frequency()
        covariance = white_covariance(spectrum, 7.0)
        wheel = Wheel(spectrum)
        assert wheel.covariance is None
        wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
        np.testing.assert_array_equal(wheel.covariance.A, covariance.A)
        assert wheel.covariance.grid == spectrum.grid
        wheel.add(EchoBlock("echo"), role="signal")
        wheel.run(2)
        np.testing.assert_array_equal(wheel.covariance.A, covariance.A)
        assert wheel.ledger["noise"].template is None

    def test_fixed_noise_block_republishes_each_cycle(self, observed):
        seen = []

        class Reader(ConstBlock):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                seen.append(covariance.A.copy())
                return super().sample(
                    conditional_residual, covariance, current_block_state, rng=rng
                )

        covariance = white_covariance(observed, 7.0)
        wheel = Wheel(observed_data=observed)
        wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
        wheel.add(Reader("rec", 1.0), role="signal")
        wheel.run(2)
        assert len(seen) == 2
        for received in seen:
            np.testing.assert_array_equal(received, 7.0)
        np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 7.0)


class TestOnCycle:
    def test_callback_called_per_turn_with_wheel(self, observed):
        calls = []
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 1.0), role="signal")
        wheel.run(
            num_cycles=3, on_cycle_complete=lambda i, w: calls.append((i, w is wheel))
        )
        assert calls == [(0, True), (1, True), (2, True)]


class TestBoundaryGuards:
    """The Wheel refuses returns that would silently corrupt other blocks."""

    def test_a_signal_block_does_not_disturb_the_noise_model(self, rng):
        """A signal block never publishes a covariance, so the noise sum is
        exactly what the noise blocks published."""
        obs = make_observed(rng).to_frequency()
        covariance = white_covariance(obs, 3.0)
        wheel = Wheel(obs)
        wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
        expected = wheel.covariance.A.copy()
        wheel.add(ConstBlock("signal", 1.0), role="signal")
        wheel.run(2)
        np.testing.assert_array_equal(wheel.covariance.A, expected)

    def test_non_finite_return_raises_and_names_the_channel(self, observed):
        class Blowup(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return conditional_residual.block_state(
                    {
                        ch: arr * np.nan
                        for ch, arr in conditional_residual.channel_data.items()
                    }
                )

        wheel = Wheel(observed)
        wheel.add(Blowup("boom", 0.0), role="signal")
        with pytest.raises(ValueError, match="non-finite sample"):
            wheel.run(1)

    def test_missing_sample_is_caught_at_registration(self, observed):
        class NoSample:
            name = "nosample"

        with pytest.raises(TypeError, match="does not implement sample"):
            Wheel(observed).add(NoSample(), role="signal")

    def test_block_with_name_initial_sample_and_sample_can_register(self, observed):
        class Minimal:
            name = "minimal"

            def initial_sample(self, conditional_residual, covariance, *, rng):
                return BlockState()

            def state_from_parameters(self, parameters, reference_data):
                # Not rebuilt: this test block's template is not a function of its
                # parameters.
                return BlockState(parameters=dict(parameters))

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return current_block_state

        wheel = Wheel(observed)
        wheel.add(Minimal(), role="signal")
        wheel.run(1)
        assert list(wheel.ledger) == ["minimal"]

    def test_a_death_move_to_a_zero_template_is_silent(self, observed):
        # a reversible-jump block whose last source dies returns a zero
        # template explicitly; there is no heuristic left to trip
        import warnings as _w

        class Death(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return conditional_residual.zero_block_state(parameters={"value": 0.0})

        wheel = Wheel(observed)
        wheel.add(Death("rj", 1.0), role="signal")
        with _w.catch_warnings():
            _w.simplefilter("error")
            wheel.run(1)
        np.testing.assert_array_equal(wheel.contribution("rj")["A"], 0.0)

    def test_a_genuinely_zero_template_does_not_warn(self, observed):
        # EchoBlock contributes a zero template on every sampling call
        import warnings as _w

        wheel = Wheel(observed)
        wheel.add(EchoBlock("echo"), role="signal")
        with _w.catch_warnings():
            _w.simplefilter("error")  # any warning becomes a failure
            wheel.run(3)

    def test_real_frequency_domain_data_is_rejected(self, rng):
        # a real one-sided spectrum would let a block return `.real` and be
        # credited with the entire imaginary part as its model
        with pytest.raises(TypeError, match="real but data_domain='frequency'"):
            make_observed(
                rng,
                data_domain="frequency",
                channel_data={ch: np.zeros(33) for ch in ("A", "E", "T")},
                num_time_samples=64,
            )

    def test_wider_model_promotes_rather_than_raising(self, rng):
        # float32 observed + float64 template must promote, not fail the way
        # the old in-place subtraction did
        obs = make_observed(
            rng, channel_data={ch: np.zeros(64, np.float32) for ch in ("A", "E", "T")}
        )

        class Wider(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return conditional_residual.block_state(
                    {
                        ch: np.ones(64, np.float64)
                        for ch in conditional_residual.channel_data
                    }
                )

        wheel = Wheel(obs)
        wheel.add(Wider("w", 0.0), role="signal")
        wheel.run(1)
        assert wheel.residual().channel_data["A"].dtype == np.float64
        assert wheel.observed_data.channel_data["A"].dtype == np.float32  # left alone


def test_cancelling_signals_do_not_erase_the_observation(rng):
    observed = make_observed(
        rng, channel_data={name: np.ones(64) for name in ("A", "E", "T")}
    )
    wheel = Wheel(observed)
    wheel.add(ConstBlock("positive", 1e16), role="signal")
    wheel.add(ConstBlock("negative", -1e16), role="signal")
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], np.ones(64))
    wheel.run(2)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], np.ones(64))


def test_cached_signal_sums_follow_updates_withdrawals_and_failures(rng):
    observed = make_observed(
        rng, channel_data={name: np.full(64, 100.0) for name in ("A", "E", "T")}
    )

    class Changing(ConstBlock):
        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            self.state.updates += 1
            count = self.state.updates
            if count == 2 and self.name == "4":
                # Signal leaves are finite individually, but their sum overflows.
                return conditional_residual.block_state(
                    {name: np.full(64, np.finfo(float).max) for name in ("A", "E", "T")}
                )
            if count == 2:
                return BlockState()
            return conditional_residual.block_state(
                {name: np.full(64, self.value + count) for name in ("A", "E", "T")},
            )

    wheel = Wheel(observed)
    for index in range(9):
        wheel.add(Changing(str(index), float(index + 1)), role="signal")
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 55.0)
    wheel.run(1)
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 46.0)
    for index in range(9):
        np.testing.assert_array_equal(
            wheel.residual(exclude_block_name=str(index)).channel_data["A"],
            48.0 + index,
        )
    # Force an overflow in the candidate tree after its leaves pass validation.
    wheel.add(ConstBlock("overflow_guard", np.finfo(float).max), role="signal")
    before_failed_block = wheel.contribution("4")["A"]
    with pytest.raises(ValueError, match="non-finite"):
        wheel.run(1)
    np.testing.assert_array_equal(wheel.contribution("4")["A"], before_failed_block)
    # The first four withdrawals were accepted; the rejected candidate never
    # enters the cached sum. Excluding the enormous guard recovers the small sum.
    np.testing.assert_array_equal(
        wheel.residual(exclude_block_name="overflow_guard").channel_data["A"], 60.0
    )


def test_signal_aggregation_preserves_small_float32_contributions(rng):
    observed = make_observed(
        rng, channel_data={name: np.ones(64) for name in ("A", "E", "T")}
    )

    class NarrowSignal(ConstBlock):
        def _block_state(self, residual):
            return residual.block_state(
                {
                    name: np.full(64, self.value, dtype=np.float32)
                    for name in residual.channel_names
                }
            )

    wheel = Wheel(observed)
    for index, value in enumerate((1e8, 1.0, -1e8)):
        wheel.add(NarrowSignal(str(index), value), role="signal")
    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 0.0)


class TestDocumentedContracts:
    """Guarantees the docstrings make that nothing else pins."""

    def test_initial_sample_reads_the_current_residual(self, rng):
        """A new block starts from the observations minus the blocks registered
        before it."""
        obs = make_observed(rng)
        wheel = Wheel(obs)
        wheel.add(ConstBlock("a", 1.0), role="signal")
        seen = {}

        class Reader(ConstBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                seen["residual"] = conditional_residual
                return super().initial_sample(conditional_residual, covariance, rng=rng)

        wheel.add(Reader("b", 2.0), role="signal")
        np.testing.assert_allclose(
            seen["residual"].channel_data["A"], obs.channel_data["A"] - 1.0
        )
        np.testing.assert_allclose(
            wheel.residual().channel_data["A"], obs.channel_data["A"] - 3.0
        )

    def test_residual_returns_fresh_arrays(self, rng):
        """residual()'s docstring: 'callers may mutate freely'."""
        obs = make_observed(rng)
        snapshot = obs.channel_data["A"].copy()
        wheel = Wheel(obs)  # no blocks: the aliasing case
        wheel.residual().channel_data["A"][0] = -999.0
        np.testing.assert_array_equal(obs.channel_data["A"], snapshot)
        wheel.add(ConstBlock("a", 1.0), role="signal")
        wheel.residual(exclude_block_name="a").channel_data["A"][0] = -999.0
        np.testing.assert_array_equal(obs.channel_data["A"], snapshot)

    def test_contribution_returns_a_copy_of_the_ledger_entry(self, observed):
        wheel = Wheel(observed)
        wheel.add(ConstBlock("a", 2.0), role="signal")
        got = wheel.contribution("a")
        got["A"][:] = 100.0
        np.testing.assert_allclose(wheel.contribution("a")["A"], 2.0)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("channel", ["A", "E", "T"])
    def test_non_finite_is_caught_in_any_channel_and_any_form(self, rng, bad, channel):
        obs = make_observed(rng)

        class Blowup(ConstBlock):
            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                tdi = {
                    ch: np.zeros_like(arr)
                    for ch, arr in conditional_residual.channel_data.items()
                }
                tdi[channel][3] = bad
                return conditional_residual.block_state(tdi)

        wheel = Wheel(obs)
        wheel.add(Blowup("b", 0.0), role="signal")
        with pytest.raises(ValueError, match="non-finite sample"):
            wheel.run(1)

    def test_a_template_zero_in_one_channel_is_recorded_per_channel(self, rng):
        obs = make_observed(rng)

        def partly_zero_result(residual):
            # non-zero in A and E, zero in T
            return residual.block_state(
                {
                    ch: np.full_like(arr, 0.0 if ch == "T" else 1.0)
                    for ch, arr in residual.channel_data.items()
                }
            )

        class PartlyZero:
            name = "p"

            def initial_sample(self, conditional_residual, covariance, *, rng):
                return partly_zero_result(conditional_residual)

            def state_from_parameters(self, parameters, reference_data):
                # Not rebuilt: this test block's template is not a function of its
                # parameters.
                return BlockState(parameters=dict(parameters))

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return partly_zero_result(conditional_residual)

        wheel = Wheel(obs)
        wheel.add(PartlyZero(), role="signal")
        wheel.run(2)
        np.testing.assert_allclose(wheel.contribution("p")["T"], 0.0)
        np.testing.assert_allclose(wheel.contribution("p")["A"], 1.0)

    def test_block_name_must_be_a_string_not_just_non_empty(self, observed):
        class Numbered(ConstBlock):
            pass

        block = Numbered("x", 0.0)
        block.name = 5
        with pytest.raises(ValueError, match="non-empty string"):
            Wheel(observed).add(block, role="signal")

    def test_missing_name_gets_the_protocol_message(self, observed):
        class Anonymous:
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return BlockState()

            def state_from_parameters(self, parameters, reference_data):
                # Not rebuilt: this test block's template is not a function of its
                # parameters.
                return BlockState(parameters=dict(parameters))

            def sample(
                self,
                conditional_residual,
                covariance,
                current_block_state,
                *,
                rng,
            ):
                return conditional_residual.zero_block_state()

        with pytest.raises((ValueError, TypeError), match="name"):
            Wheel(observed).add(Anonymous(), role="signal")


class TestObservedDataIsChecked:
    """Wheel.__init__ checks the data itself, so the first block to touch it
    is not blamed by the return-value guard for damage it did not do."""

    def test_non_finite_observed_data_is_refused_at_construction(self, rng):
        obs = make_observed(rng)
        obs.channel_data["E"][7] = np.nan
        with pytest.raises(
            ValueError, match=r"observed_data.channel_data\['E'\] has non-finite"
        ):
            Wheel(obs)

    def test_the_message_points_at_the_missing_gap_support(self, rng):
        """NaN is how a user marks a gap today, and gaps are not in the
        contract yet -- the error has to say so or it reads as a bug."""
        obs = make_observed(rng)
        obs.channel_data["A"][:3] = np.nan
        with pytest.raises(ValueError, match="no data-quality mask"):
            Wheel(obs)

    def test_finite_data_constructs_normally(self, observed):
        assert Wheel(observed).residual() is not None

    def test_covariance_subclass_cannot_skip_grid_validation(self, observed):
        """Wheel and FixedNoiseBlock copy through Covariance's own methods, so
        overriding copy() and check_compatible() cannot store an unvalidated
        object."""

        class UncheckedCovariance(Covariance):
            def copy(self):
                return self

            def check_compatible(self, reference_data):
                pass

        mismatched_grid = make_observed(np.random.default_rng(1), num_time_samples=32)
        covariance = white_covariance(mismatched_grid, 1.0)
        unchecked = UncheckedCovariance(
            covariance.blocks,
            channel_names=covariance.channel_names,
            grid=covariance.grid,
        )
        wheel = Wheel(observed)
        with pytest.raises(ValueError, match="grid"):
            wheel.add(FixedNoiseBlock("fixed", unchecked), role="noise")
        assert len(wheel.ledger) == 0
        assert wheel.covariance is None

        class PublishesUnchecked(EchoBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return BlockState(
                    covariance=white_covariance(conditional_residual, 1.0)
                )

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return BlockState(covariance=unchecked)

        wheel = Wheel(observed)
        wheel.add(PublishesUnchecked("noise"), role="noise")
        with pytest.raises(ValueError, match="grid"):
            wheel.run(1)
        assert wheel.covariance.A.shape == observed.channel_data["A"].shape
        np.testing.assert_array_equal(wheel.covariance.A, 1.0)

        # On the right grid, validation passes; only the copy keeps the stored
        # object a plain Covariance rather than the caller's subclass.
        compatible = UncheckedCovariance(
            white_covariance(observed, 2.0).blocks,
            channel_names=observed.channel_names,
            grid=observed.grid,
        )
        wheel = Wheel(observed)
        wheel.add(FixedNoiseBlock("fixed", compatible), role="noise")
        assert type(wheel._noise_sum.total) is Covariance
        assert wheel._noise_sum.total is not compatible
        assert type(wheel.ledger["fixed"].covariance) is Covariance

        class PublishesCompatible(EchoBlock):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                return BlockState(covariance=compatible)

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return BlockState(covariance=compatible)

        wheel = Wheel(observed)
        wheel.add(PublishesCompatible("noise"), role="noise")
        assert type(wheel._noise_sum.total) is Covariance
        assert type(wheel.ledger["noise"].covariance) is Covariance
        wheel.run(1)
        assert type(wheel._noise_sum.total) is Covariance
        assert wheel._noise_sum.total is not compatible
        assert type(wheel.ledger["noise"].covariance) is Covariance
