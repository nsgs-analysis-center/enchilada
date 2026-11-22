"""Declared roles: signal blocks sum templates, noise blocks sum covariances."""

import re
from dataclasses import dataclass

import numpy as np
import pytest

import enchilada
import enchilada.wheel
from conftest import const_block_state, white_covariance
from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel, transform
from enchilada.testing import EchoBlock


def home_level(observed, variance):
    """White time variance `variance` as it reads on the WDM home grid: dt * sigma^2."""
    return variance / observed.fs


class ConstSignal:
    """A signal block publishing a constant template; `state.updates` counts samples."""

    role = "signal"

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

    role = "noise"

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


def offender(declared, method, *, with_template, with_covariance):
    """A block declaring role `declared` whose `method` returns the given fields.

    The other method returns what the role allows, so the block registers and
    samples normally until `method` is called.
    """

    class Offender:
        name = "offender"
        role = declared

        @staticmethod
        def _allowed(conditional_residual):
            if declared == "noise":
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
            # Stub: no test rebuilds this block's state.
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
    np.testing.assert_array_equal(after["residual"], before["residual"])
    _same_array(after["covariance"], before["covariance"])
    for name, (template, covariance, parameters) in before["ledger"].items():
        after_template, after_covariance, after_parameters = after["ledger"][name]
        _same_array(after_template, template)
        _same_array(after_covariance, covariance)
        assert after_parameters == parameters


class TestRoles:
    def test_the_role_is_read_from_the_block_before_the_method_checks(self, observed):
        class RoleLess(ConstSignal):
            role = None

        class Bare:
            name = "bare"

        wheel = Wheel(observed, num_frequency_divisions=8)
        with pytest.raises(TypeError, match="block 'a' declares no role"):
            wheel.add(RoleLess("a", 1.0))
        with pytest.raises(TypeError, match="block 'bare' declares no role"):
            wheel.add(Bare())
        typo = ConstSignal("b", 1.0)
        typo.role = "Noise"
        with pytest.raises(
            ValueError, match=re.escape("block 'b' declares role 'Noise'; a role is")
        ):
            wheel.add(typo)
        assert len(wheel.ledger) == 0
        assert wheel._blocks == []
        assert wheel._roles == {}

    def test_roles_are_recorded_per_block(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(WhiteNoise("noise", 1.0, {}))
        wheel.add(ConstSignal("signal", 1.0))
        assert wheel._roles == {"noise": "noise", "signal": "signal"}

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_signal_block_may_not_return_a_covariance(self, observed, method):
        self._check_refusal(
            observed,
            method,
            declared="signal",
            with_template=True,
            with_covariance=True,
            message=(
                f"offender.{method} returned a covariance but 'offender' "
                "declares role 'signal'"
            ),
        )

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_noise_block_may_not_return_a_template(self, observed, method):
        self._check_refusal(
            observed,
            method,
            declared="noise",
            with_template=True,
            with_covariance=True,
            message=(
                f"offender.{method} returned a template but 'offender' "
                "declares role 'noise'"
            ),
        )

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_noise_block_must_return_a_covariance(self, observed, method):
        self._check_refusal(
            observed,
            method,
            declared="noise",
            with_template=False,
            with_covariance=False,
            message=(
                f"offender.{method} must return a covariance because 'offender' "
                "declares role 'noise'"
            ),
        )

    @staticmethod
    def _check_refusal(
        observed, method, *, declared, with_template, with_covariance, message
    ):
        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
        block = offender(
            declared,
            method,
            with_template=with_template,
            with_covariance=with_covariance,
        )
        if method == "sample":
            # First in the cycle, so nothing else commits before the refusal.
            wheel.add(block)
        wheel.add(WhiteNoise("noise", 1.0, {}))
        wheel.add(ConstSignal("signal", 1.0))
        before = snapshot(wheel)
        with pytest.raises(ValueError, match=re.escape(message)):
            if method == "initial_sample":
                wheel.add(block)
            else:
                wheel.run(1)
        assert_unchanged(wheel, before)
        if method == "initial_sample":
            assert "offender" not in wheel._roles
            assert "offender" not in wheel._block_indices
        else:
            assert wheel._roles["offender"] == declared

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

        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
        # The only noise block, so no Covariance.__add__ in the sum revalidates it:
        # the refusal can only come from the copy Wheel takes at adoption.
        wheel.add(Smuggler("smuggler", 1.0, {}))
        wheel.add(ConstSignal("signal", 0.5))
        before = snapshot(wheel)
        with pytest.raises(ValueError, match=re.escape("blocks must be finite")):
            wheel.run(1)
        assert_unchanged(wheel, before)
        assert wheel._roles == {"smuggler": "noise", "signal": "signal"}
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 1.0))


class TestNoiseSum:
    def test_the_campaign_has_no_covariance_until_a_noise_block_publishes(
        self, observed
    ):
        seen = {}
        wheel = Wheel(observed, num_frequency_divisions=8)
        assert wheel.covariance is None
        wheel.add(RecordingSignal("signal", 0.5, seen))
        wheel.run(1)
        assert seen["signal"] == [None, None]
        assert wheel.covariance is None

    def test_two_noise_blocks_each_receive_the_other_and_wheel_holds_the_sum(
        self, observed
    ):
        seen = {}
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(WhiteNoise("instrument", 1.0, seen))
        wheel.add(WhiteNoise("confusion", 2.0, seen))
        assert seen["instrument"] == [None]  # alone at registration
        np.testing.assert_allclose(seen["confusion"][0], 1.0)
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
        wheel.run(1)
        np.testing.assert_allclose(seen["instrument"][1], 2.0)
        np.testing.assert_allclose(seen["confusion"][1], 1.0)
        np.testing.assert_array_equal(wheel.ledger["instrument"].covariance.A, 1.0)
        np.testing.assert_array_equal(wheel.ledger["confusion"].covariance.A, 2.0)
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
        assert wheel.covariance.grid == wheel.home_grid

    def test_a_signal_block_receives_the_sum_over_every_noise_block(self, observed):
        seen = {}
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(RecordingSignal("early", 0.5, seen))
        wheel.add(WhiteNoise("instrument", 1.0, seen))
        wheel.add(WhiteNoise("confusion", 2.0, seen))
        wheel.add(RecordingSignal("late", 0.5, seen))
        assert seen["early"] == [None]  # registered before any noise block
        np.testing.assert_allclose(seen["late"][0], 3.0)
        wheel.run(2)
        assert len(seen["early"]) == len(seen["late"]) == 3
        for name in ("early", "late"):
            for received in seen[name][1:]:
                np.testing.assert_allclose(received, 3.0)

    def test_three_noise_blocks_each_receive_the_sum_of_the_other_two(self, observed):
        seen = {}
        wheel = Wheel(observed, num_frequency_divisions=8)
        for name, level in (("a", 1.0), ("b", 2.0), ("c", 4.0)):
            wheel.add(WhiteNoise(name, level, seen))
        np.testing.assert_allclose(seen["c"][0], 3.0)
        wheel.run(1)
        np.testing.assert_allclose(seen["a"][1], 6.0)
        np.testing.assert_allclose(seen["b"][1], 5.0)
        np.testing.assert_allclose(seen["c"][1], 3.0)
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 7.0))

    def test_a_republished_covariance_replaces_that_block_s_slot(self, observed):
        class Growing(WhiteNoise):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                self.level += 1.0
                return self.initial_sample(conditional_residual, covariance, rng=rng)

        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(Growing("growing", 1.0, {}))
        wheel.add(FixedNoiseBlock("fixed", white_covariance(observed, 10.0)))
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 11.0))
        wheel.run(1)
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 12.0))
        wheel.run(1)
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 13.0))
        np.testing.assert_array_equal(wheel.ledger["growing"].covariance.A, 3.0)
        np.testing.assert_array_equal(wheel.ledger["fixed"].covariance.A, 10.0)

    def test_noise_blocks_contribute_no_signal(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(WhiteNoise("a", 1.0, {}))
        wheel.add(WhiteNoise("b", 2.0, {}))
        wheel.run(2)
        for name in observed.channel_names:
            np.testing.assert_allclose(
                wheel.residual().channel_data[name],
                observed.channel_data[name],
                rtol=1e-12,
                atol=1e-12,
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

        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(WhiteNoise("a", 1.0, {}))
        wheel.add(WhiteNoise("b", 2.0, {}))
        wheel.add(Mutates("signal", 0.5))
        wheel.run(2)
        assert len(received) == 2
        assert received[0] is not received[1]
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
        total = wheel.covariance
        total.blocks.setflags(write=True)
        total.A[:] = 99.0
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
        assert type(wheel.covariance) is Covariance
        assert not wheel.covariance.blocks.flags.writeable

    def test_the_sum_describes_the_observations(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(WhiteNoise("a", 1.0, {}))
        wheel.add(WhiteNoise("b", 2.0, {}))
        wheel.covariance.check_compatible(
            transform(observed, "wdm", num_frequency_divisions=8)
        )
        transform(wheel.covariance, "time").check_compatible(observed)

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
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(WhiteNoise("a", 2.0, seen))
        wheel.add(Corrupting("b", 3.0, seen))
        wheel.run(1)
        np.testing.assert_allclose(seen["b"][0], 2.0)  # a's model at registration
        np.testing.assert_array_equal(seen["b"][1], 999.0)  # its copy was overwritten
        np.testing.assert_array_equal(wheel.ledger["a"].covariance.A, 2.0)
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 5.0))
        np.testing.assert_allclose(
            wheel._noise_sum.excluding(wheel._block_indices["b"]).A,
            home_level(observed, 2.0),
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
        wheel = Wheel(observed, num_frequency_divisions=8)
        wheel.add(Rising("a", 2.0, seen))
        wheel.add(TurnsSignal("b", 1.0, seen))
        with pytest.raises(
            ValueError,
            match=re.escape(
                "b.sample returned a template but 'b' declares role 'noise'"
            ),
        ):
            wheel.run(1)  # registration order: a is adopted, then b is refused
        assert len(seen["b"]) == 2
        np.testing.assert_allclose(seen["b"][1], 3.0)  # a's new leaf, handed to b
        np.testing.assert_array_equal(wheel.ledger["a"].covariance.A, 3.0)
        np.testing.assert_array_equal(wheel.ledger["b"].covariance.A, 1.0)
        assert wheel.ledger["b"].template is None
        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 4.0))


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
        wheel = Wheel(observed, num_frequency_divisions=8)
        assert not hasattr(wheel, "_covariance")
        assert not hasattr(wheel, "_noise_owner_block_name")

    def test_the_role_keyword_and_the_one_grid_rule_are_gone(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=8)
        with pytest.raises(TypeError, match="role"):
            wheel.add(EchoBlock("e"), **{"role": "signal"})  # the keyword is gone
        assert not hasattr(wheel, "_covariance_grid")
        assert not hasattr(wheel, "_canonical_state")
        assert not hasattr(wheel, "_observed_data")
