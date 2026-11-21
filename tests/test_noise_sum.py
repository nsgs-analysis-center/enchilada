"""Roles at Wheel.add: signal blocks sum templates, noise blocks sum covariances."""

import re
from dataclasses import dataclass

import numpy as np
import pytest

import enchilada
import enchilada.wheel
from conftest import const_block_state, white_covariance
from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel, transform
from enchilada.testing import EchoBlock


class ConstSignal:
    """A signal block publishing a constant template; `state.updates` counts samples."""

    @dataclass
    class State:
        updates: int = 0

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return const_block_state(conditional_residual, self.value)

    def state_from_parameters(self, parameters, reference_data):
        return const_block_state(reference_data, self.value)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.updates += 1
        return const_block_state(conditional_residual, self.value)


def record(seen, name, covariance):
    """Append the handed covariance's A variances (or None) to seen[name]."""
    seen.setdefault(name, []).append(
        None if covariance is None else covariance.A.copy()
    )


class WhiteNoise:
    """A noise block publishing white noise of `level`; records what it receives.

    `seen[name]` collects the covariance handed to each call, `initial_sample`
    first; `sample` republishes the last accepted state unchanged.
    """

    def __init__(self, name, level, seen):
        self.name = name
        self.level = level
        self.seen = seen

    def initial_sample(self, conditional_residual, covariance, *, rng):
        record(self.seen, self.name, covariance)
        return BlockState(
            covariance=white_covariance(conditional_residual, self.level),
            parameters={"level": self.level},
        )

    def state_from_parameters(self, parameters, reference_data):
        return BlockState(
            covariance=white_covariance(reference_data, float(parameters["level"])),
            parameters={"level": parameters["level"]},
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        record(self.seen, self.name, covariance)
        return current_block_state


class RecordingSignal(ConstSignal):
    """A signal block that records the covariance handed to each call."""

    def __init__(self, name, value, seen):
        super().__init__(name, value)
        self.seen = seen

    def initial_sample(self, conditional_residual, covariance, *, rng):
        record(self.seen, self.name, covariance)
        return super().initial_sample(conditional_residual, covariance, rng=rng)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        record(self.seen, self.name, covariance)
        return super().sample(
            conditional_residual, covariance, current_block_state, rng=rng
        )


def offender(role, method, *, with_template, with_covariance):
    """A block of `role` whose `method` returns the given fields.

    The other method returns what the role allows, so the block registers and
    samples normally until `method` is called.
    """

    class Offender:
        name = "offender"

        @staticmethod
        def _allowed(conditional_residual):
            if role == "noise":
                return BlockState(
                    covariance=white_covariance(conditional_residual, 1.0)
                )
            return const_block_state(conditional_residual, 0.5)

        @staticmethod
        def _offending(conditional_residual):
            template = const_block_state(conditional_residual, 0.5).template
            return BlockState(
                template=template if with_template else None,
                covariance=(
                    white_covariance(conditional_residual, 1.0)
                    if with_covariance
                    else None
                ),
            )

        def initial_sample(self, conditional_residual, covariance, *, rng):
            make = self._offending if method == "initial_sample" else self._allowed
            return make(conditional_residual)

        def state_from_parameters(self, parameters, reference_data):
            # Not rebuilt: this test block's template is not a function of its
            # parameters.
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            make = self._offending if method == "sample" else self._allowed
            return make(conditional_residual)

    return Offender()


def snapshot(wheel):
    """Everything a refused adoption must leave alone."""
    total = wheel.covariance
    return {
        "names": list(wheel.ledger),
        "residual": wheel.working_residual.channel_data["A"].copy(),
        "covariance": None if total is None else total.A.copy(),
        "rng": wheel._rng.bit_generator.state,
        "signal_sum": wheel._signal_sum,
        "noise_sum": wheel._noise_sum,
        "covariance_grid": wheel._covariance_grid,
        "ledger": {
            name: (
                None if entry.template is None else entry.template["A"].copy(),
                None if entry.covariance is None else entry.covariance.A.copy(),
                entry.parameters,
            )
            for name, entry in wheel.ledger.snapshot().items()
        },
    }


def _same_array(after, before):
    if before is None:
        assert after is None
    else:
        np.testing.assert_array_equal(after, before)


def assert_unchanged(wheel, before):
    after = snapshot(wheel)
    assert after["names"] == before["names"]
    assert after["rng"] == before["rng"]
    assert after["signal_sum"] is before["signal_sum"]
    assert after["noise_sum"] is before["noise_sum"]
    assert after["covariance_grid"] == before["covariance_grid"]
    np.testing.assert_array_equal(after["residual"], before["residual"])
    _same_array(after["covariance"], before["covariance"])
    for name, (template, covariance, parameters) in before["ledger"].items():
        after_template, after_covariance, after_parameters = after["ledger"][name]
        _same_array(after_template, template)
        _same_array(after_covariance, covariance)
        assert after_parameters == parameters


class TestRoles:
    def test_role_is_validated_before_the_block_is_inspected(self, observed):
        class Untouchable:
            @property
            def name(self):
                raise AssertionError(
                    "add must refuse the role before reading the block"
                )

        wheel = Wheel(observed)
        for role in ("Signal", "noise ", "", None, 1):
            with pytest.raises(ValueError, match="role must be 'signal' or 'noise'"):
                wheel.add(Untouchable(), role=role)
        with pytest.raises(TypeError, match="role"):
            wheel.add(ConstSignal("a", 1.0))
        assert len(wheel.ledger) == 0
        assert wheel._blocks == []
        assert wheel._roles == {}

    def test_the_role_message_shows_the_offending_value(self, observed):
        with pytest.raises(ValueError, match=re.escape("got 'Noise'")):
            Wheel(observed).add(EchoBlock("e"), role="Noise")

    def test_roles_are_recorded_per_block(self, observed):
        wheel = Wheel(observed)
        wheel.add(WhiteNoise("noise", 1.0, {}), role="noise")
        wheel.add(ConstSignal("signal", 1.0), role="signal")
        assert wheel._roles == {"noise": "noise", "signal": "signal"}

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_signal_block_may_not_return_a_covariance(self, observed, method):
        self._check_refusal(
            observed,
            method,
            role="signal",
            with_template=True,
            with_covariance=True,
            message=(
                f"offender.{method} returned a covariance but 'offender' is "
                "registered as a signal block"
            ),
        )

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_noise_block_may_not_return_a_template(self, observed, method):
        self._check_refusal(
            observed,
            method,
            role="noise",
            with_template=True,
            with_covariance=True,
            message=(
                f"offender.{method} returned a template but 'offender' is "
                "registered as a noise block"
            ),
        )

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_noise_block_must_return_a_covariance(self, observed, method):
        self._check_refusal(
            observed,
            method,
            role="noise",
            with_template=False,
            with_covariance=False,
            message=(
                f"offender.{method} must return a covariance because 'offender' "
                "is registered as a noise block"
            ),
        )

    @staticmethod
    def _check_refusal(
        observed, method, *, role, with_template, with_covariance, message
    ):
        wheel = Wheel(observed, random_seed=3)
        block = offender(
            role, method, with_template=with_template, with_covariance=with_covariance
        )
        if method == "sample":
            # First in the cycle, so nothing else commits before the refusal.
            wheel.add(block, role=role)
        wheel.add(WhiteNoise("noise", 1.0, {}), role="noise")
        wheel.add(ConstSignal("signal", 1.0), role="signal")
        before = snapshot(wheel)
        with pytest.raises(ValueError, match=re.escape(message)):
            if method == "initial_sample":
                wheel.add(block, role=role)
            else:
                wheel.run(1)
        assert_unchanged(wheel, before)
        if method == "initial_sample":
            assert "offender" not in wheel._roles
            assert "offender" not in wheel._block_indices
        else:
            assert wheel._roles["offender"] == role

    def test_a_noise_block_cannot_smuggle_a_non_finite_covariance_past_revalidation(
        self, observed
    ):
        class Smuggler(WhiteNoise):
            """Builds a valid covariance in sample, then writes NaN into it."""

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                record(self.seen, self.name, covariance)
                published = white_covariance(conditional_residual, self.level)
                published.blocks.setflags(write=True)
                published.blocks[0, 0, 0] = np.nan
                return BlockState(
                    covariance=published, parameters={"level": self.level}
                )

        wheel = Wheel(observed, random_seed=3)
        # The only noise block, so no Covariance.__add__ in the sum revalidates it:
        # the refusal can only come from the copy Wheel takes at adoption.
        wheel.add(Smuggler("smuggler", 1.0, {}), role="noise")
        wheel.add(ConstSignal("signal", 0.5), role="signal")
        before = snapshot(wheel)
        with pytest.raises(ValueError, match=re.escape("blocks must be finite")):
            wheel.run(1)
        assert_unchanged(wheel, before)
        assert wheel._roles == {"smuggler": "noise", "signal": "signal"}
        np.testing.assert_array_equal(wheel.covariance.A, 1.0)


class TestNoiseSum:
    def test_the_campaign_has_no_covariance_until_a_noise_block_publishes(
        self, observed
    ):
        seen = {}
        wheel = Wheel(observed)
        assert wheel.covariance is None
        wheel.add(RecordingSignal("signal", 0.5, seen), role="signal")
        wheel.run(1)
        assert seen["signal"] == [None, None]
        assert wheel.covariance is None
        assert wheel._covariance_grid is None

    def test_two_noise_blocks_each_receive_the_other_and_wheel_holds_the_sum(
        self, observed
    ):
        seen = {}
        wheel = Wheel(observed)
        wheel.add(WhiteNoise("instrument", 1.0, seen), role="noise")
        wheel.add(WhiteNoise("confusion", 2.0, seen), role="noise")
        assert seen["instrument"] == [None]  # alone at registration
        np.testing.assert_array_equal(seen["confusion"][0], 1.0)
        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
        wheel.run(1)
        np.testing.assert_array_equal(seen["instrument"][1], 2.0)
        np.testing.assert_array_equal(seen["confusion"][1], 1.0)
        np.testing.assert_array_equal(wheel.ledger["instrument"].covariance.A, 1.0)
        np.testing.assert_array_equal(wheel.ledger["confusion"].covariance.A, 2.0)
        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
        assert wheel._covariance_grid == observed.grid

    def test_a_signal_block_receives_the_sum_over_every_noise_block(self, observed):
        seen = {}
        wheel = Wheel(observed)
        wheel.add(RecordingSignal("early", 0.5, seen), role="signal")
        wheel.add(WhiteNoise("instrument", 1.0, seen), role="noise")
        wheel.add(WhiteNoise("confusion", 2.0, seen), role="noise")
        wheel.add(RecordingSignal("late", 0.5, seen), role="signal")
        assert seen["early"] == [None]  # registered before any noise block
        np.testing.assert_array_equal(seen["late"][0], 3.0)
        wheel.run(2)
        assert len(seen["early"]) == len(seen["late"]) == 3
        for name in ("early", "late"):
            for received in seen[name][1:]:
                np.testing.assert_array_equal(received, 3.0)

    def test_three_noise_blocks_each_receive_the_sum_of_the_other_two(self, observed):
        seen = {}
        wheel = Wheel(observed)
        for name, level in (("a", 1.0), ("b", 2.0), ("c", 4.0)):
            wheel.add(WhiteNoise(name, level, seen), role="noise")
        np.testing.assert_array_equal(seen["c"][0], 3.0)
        wheel.run(1)
        np.testing.assert_array_equal(seen["a"][1], 6.0)
        np.testing.assert_array_equal(seen["b"][1], 5.0)
        np.testing.assert_array_equal(seen["c"][1], 3.0)
        np.testing.assert_array_equal(wheel.covariance.A, 7.0)

    def test_a_republished_covariance_replaces_that_block_s_slot(self, observed):
        class Growing(WhiteNoise):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                self.level += 1.0
                return self.initial_sample(conditional_residual, covariance, rng=rng)

        wheel = Wheel(observed)
        wheel.add(Growing("growing", 1.0, {}), role="noise")
        wheel.add(
            FixedNoiseBlock("fixed", white_covariance(observed, 10.0)), role="noise"
        )
        np.testing.assert_array_equal(wheel.covariance.A, 11.0)
        wheel.run(1)
        np.testing.assert_array_equal(wheel.covariance.A, 12.0)
        wheel.run(1)
        np.testing.assert_array_equal(wheel.covariance.A, 13.0)
        np.testing.assert_array_equal(wheel.ledger["growing"].covariance.A, 3.0)
        np.testing.assert_array_equal(wheel.ledger["fixed"].covariance.A, 10.0)

    def test_noise_blocks_contribute_no_signal(self, observed):
        wheel = Wheel(observed)
        wheel.add(WhiteNoise("a", 1.0, {}), role="noise")
        wheel.add(WhiteNoise("b", 2.0, {}), role="noise")
        wheel.run(2)
        for name in observed.channel_names:
            np.testing.assert_array_equal(
                wheel.residual().channel_data[name], observed.channel_data[name]
            )
            np.testing.assert_array_equal(wheel.contribution("a")[name], 0.0)
        assert wheel.ledger["a"].template is None
        assert wheel.ledger["b"].template is None

    def test_handed_over_sums_are_independent_read_only_copies(self, observed):
        received = []

        class Mutates(ConstSignal):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                received.append(covariance)
                assert not covariance.blocks.flags.writeable
                covariance.blocks.setflags(write=True)
                covariance.A[:] = 99.0
                return super().sample(
                    conditional_residual, covariance, current_block_state, rng=rng
                )

        wheel = Wheel(observed)
        wheel.add(WhiteNoise("a", 1.0, {}), role="noise")
        wheel.add(WhiteNoise("b", 2.0, {}), role="noise")
        wheel.add(Mutates("signal", 0.5), role="signal")
        wheel.run(2)
        assert len(received) == 2
        assert received[0] is not received[1]
        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
        total = wheel.covariance
        total.blocks.setflags(write=True)
        total.A[:] = 99.0
        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
        assert type(wheel.covariance) is Covariance
        assert not wheel.covariance.blocks.flags.writeable

    def test_the_sum_describes_the_observations(self, observed):
        wheel = Wheel(observed)
        wheel.add(WhiteNoise("a", 1.0, {}), role="noise")
        wheel.add(WhiteNoise("b", 2.0, {}), role="noise")
        wheel.covariance.check_compatible(observed)

    def test_a_noise_block_cannot_corrupt_the_others_through_its_hand_over(
        self, observed
    ):
        class Corrupting(WhiteNoise):
            """Overwrites the other noise blocks' sum it receives, then republishes."""

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                if covariance is not None:
                    covariance.blocks.setflags(write=True)
                    covariance.blocks[...] = 999.0
                return super().sample(
                    conditional_residual, covariance, current_block_state, rng=rng
                )

        seen = {}
        wheel = Wheel(observed)
        wheel.add(WhiteNoise("a", 2.0, seen), role="noise")
        wheel.add(Corrupting("b", 3.0, seen), role="noise")
        wheel.run(1)
        np.testing.assert_array_equal(seen["b"][0], 2.0)  # a's model at registration
        np.testing.assert_array_equal(seen["b"][1], 999.0)  # its copy was overwritten
        np.testing.assert_array_equal(wheel.ledger["a"].covariance.A, 2.0)
        np.testing.assert_array_equal(wheel.covariance.A, 5.0)
        np.testing.assert_array_equal(
            wheel._noise_sum.excluding(wheel._block_indices["b"]).A, 2.0
        )

    def test_a_refused_noise_block_keeps_its_siblings_new_publication(self, observed):
        class Rising(WhiteNoise):
            """Publishes level 3.0 from sample instead of its registration level."""

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                self.level = 3.0
                return self.initial_sample(conditional_residual, covariance, rng=rng)

        class TurnsSignal(WhiteNoise):
            """Registers with a covariance, then returns a template from sample."""

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                record(self.seen, self.name, covariance)
                return const_block_state(conditional_residual, 0.5)

        seen = {}
        wheel = Wheel(observed)
        wheel.add(Rising("a", 2.0, seen), role="noise")
        wheel.add(TurnsSignal("b", 1.0, seen), role="noise")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "b.sample returned a template but 'b' is registered as a noise block"
            ),
        ):
            wheel.run(1)  # registration order: a is adopted, then b is refused
        assert len(seen["b"]) == 2
        np.testing.assert_array_equal(seen["b"][1], 3.0)  # a's new leaf, handed to b
        np.testing.assert_array_equal(wheel.ledger["a"].covariance.A, 3.0)
        np.testing.assert_array_equal(wheel.ledger["b"].covariance.A, 1.0)
        assert wheel.ledger["b"].template is None
        np.testing.assert_array_equal(wheel.covariance.A, 4.0)


class TestCovarianceGrid:
    """The first noise publication fixes the grid of the sum."""

    def test_a_noise_block_on_another_grid_is_refused_before_initial_sample(
        self, observed
    ):
        class NeverStarted:
            name = "spectral"

            def initial_sample(self, conditional_residual, covariance, *, rng):
                raise AssertionError("add must refuse the grid first")

            def state_from_parameters(self, parameters, reference_data):
                # Not rebuilt: this test block's template is not a function of its
                # parameters.
                return BlockState(parameters=dict(parameters))

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                raise AssertionError("unreachable")

        wheel = Wheel(observed, random_seed=3)
        wheel.add(WhiteNoise("time", 1.0, {}), role="noise")
        before = snapshot(wheel)
        with pytest.raises(ValueError, match="does not add across grids") as info:
            wheel.add(NeverStarted(), role="noise", data_domain="frequency")
        message = str(info.value)
        assert "'spectral'" in message
        assert "campaign covariance is on grid" in message
        assert "data_domain='frequency'" in message
        assert "data_domain='time'" in message
        assert_unchanged(wheel, before)
        assert "spectral" not in wheel._roles

    def test_a_signal_block_on_another_grid_is_refused_at_add_and_at_run(
        self, observed
    ):
        wheel = Wheel(observed, random_seed=3)
        wheel.add(ConstSignal("early", 0.5), role="signal")
        wheel.add(WhiteNoise("time", 1.0, {}), role="noise")
        wheel.add(ConstSignal("same_grid", 0.5), role="signal", data_domain="time")
        with pytest.raises(ValueError, match="does not translate between grids"):
            wheel.add(ConstSignal("late", 0.5), role="signal", data_domain="frequency")
        wheel.run(1)  # the blocks on the time grid still sample
        wheel = Wheel(observed, random_seed=3)
        wheel.add(ConstSignal("spectral", 0.5), role="signal", data_domain="frequency")
        wheel.add(WhiteNoise("time", 1.0, {}), role="noise")
        with pytest.raises(
            ValueError,
            match="'spectral' is registered on grid .* but the current covariance",
        ):
            wheel.run(1)

    def test_the_first_noise_publication_fixes_the_grid(self, observed):
        spectrum = observed.to_frequency()
        wheel = Wheel(observed)
        wheel.add(
            FixedNoiseBlock("spectral", white_covariance(spectrum, 1.0)),
            role="noise",
            data_domain="frequency",
        )
        assert wheel._covariance_grid == spectrum.grid
        assert wheel.covariance.grid == spectrum.grid
        wheel.add(
            FixedNoiseBlock("more", white_covariance(spectrum, 2.0)),
            role="noise",
            data_domain="frequency",
        )
        np.testing.assert_array_equal(
            wheel.covariance.A, white_covariance(spectrum, 3.0).A
        )
        with pytest.raises(ValueError, match="does not add across grids"):
            wheel.add(
                FixedNoiseBlock("time", white_covariance(observed, 1.0)), role="noise"
            )

    def test_a_refused_first_noise_publication_leaves_the_grid_unset(self, observed):
        class Publishes:
            """A noise block on the time grid, publishing on frequency if off_grid."""

            name = "noise"

            def __init__(self, off_grid):
                self.off_grid = off_grid

            def initial_sample(self, conditional_residual, covariance, *, rng):
                published_on = (
                    transform(conditional_residual, "frequency")
                    if self.off_grid
                    else conditional_residual
                )
                return BlockState(covariance=white_covariance(published_on, 1.0))

            def state_from_parameters(self, parameters, reference_data):
                # Not rebuilt: this test block's template is not a function of its
                # parameters.
                return BlockState(parameters=dict(parameters))

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return current_block_state

        wheel = Wheel(observed, random_seed=3)
        before = snapshot(wheel)
        with pytest.raises(
            ValueError, match=re.escape("noise.initial_sample covariance is on grid")
        ):
            wheel.add(Publishes(off_grid=True), role="noise")
        assert_unchanged(wheel, before)
        assert wheel._covariance_grid is None
        assert wheel.covariance is None
        assert wheel._roles == {}
        wheel.add(Publishes(off_grid=False), role="noise")
        assert wheel._covariance_grid == observed.grid
        np.testing.assert_array_equal(wheel.covariance.A, 1.0)


class TestRemovedApi:
    def test_noise_overwritten_warning_is_gone(self):
        assert not hasattr(enchilada, "NoiseOverwrittenWarning")
        assert not hasattr(enchilada.wheel, "NoiseOverwrittenWarning")
        assert "NoiseOverwrittenWarning" not in enchilada.__all__
        with pytest.raises(ImportError):
            from enchilada import NoiseOverwrittenWarning  # noqa: F401

    def test_initial_covariance_is_gone(self, observed):
        covariance = white_covariance(observed, 1.0)
        with pytest.raises(TypeError, match="initial_covariance"):
            Wheel(observed, initial_covariance=covariance)
        with pytest.raises(TypeError, match="positional"):
            Wheel(observed, covariance)
        assert not hasattr(Wheel(observed), "_covariance")
        assert not hasattr(Wheel(observed), "_noise_owner_block_name")
