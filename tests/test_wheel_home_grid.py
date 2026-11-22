"""The WDM home grid: one campaign grid, blocks on any grid, the sum translated."""

import re
from dataclasses import dataclass, field

import numpy as np
import pytest

import enchilada.wheel
from conftest import white_covariance
from enchilada import (
    Block,
    BlockState,
    Covariance,
    FixedNoiseBlock,
    Grid,
    WDMGrid,
    Wheel,
    transform,
)
from enchilada.testing import EchoBlock, check_block

NF = 8  # home-grid rows for the 64-sample fixture (fs = 0.5, so dt = 2.0)
CLOSE = {"rtol": 1e-12, "atol": 1e-12}  # one exact WDM transform and back


def home_of(data):
    return transform(data, "wdm", num_frequency_divisions=NF)


def level_state(conditional_residual, value):
    """A template of `value` in every time sample, expressed on the residual's grid."""
    time_reference = transform(conditional_residual, "time")
    start = time_reference.block_state(
        {
            ch: np.full(time_reference.num_time_samples, value)
            for ch in time_reference.channel_names
        }
    )
    grid = conditional_residual.wdm_grid
    options = (
        {}
        if grid is None
        else {"num_frequency_divisions": grid.num_frequency_divisions}
    )
    return transform(
        start,
        conditional_residual.data_domain,
        reference_data=time_reference,
        **options,
    )


class LevelSignal:
    """A signal block adding `value` to every time sample; records its inputs."""

    role = "signal"

    @dataclass
    class State:
        received: list = field(default_factory=list)  # (residual, covariance)
        states: list = field(default_factory=list)  # current_block_state per sample

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self.state.received.append((conditional_residual, covariance))
        return level_state(conditional_residual, self.value)

    def state_from_parameters(self, parameters, reference_data):
        return level_state(reference_data, self.value)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.received.append((conditional_residual, covariance))
        self.state.states.append(current_block_state)
        return level_state(conditional_residual, self.value)


class WhiteNoise:
    """A noise block publishing white noise of `level`; records its inputs."""

    role = "noise"

    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.received = []  # (covariance, current_block_state or None) per call
        self.published = []  # the covariance returned by each call

    def _publish(self, conditional_residual):
        return white_covariance(conditional_residual, self.level)

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self.received.append((covariance, None))
        self.published.append(self._publish(conditional_residual))
        return BlockState(covariance=self.published[-1])

    def state_from_parameters(self, parameters, reference_data):
        return BlockState(covariance=self._publish(reference_data))

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.received.append((covariance, current_block_state))
        self.published.append(self._publish(conditional_residual))
        return BlockState(covariance=self.published[-1])


class ColouredNoise(WhiteNoise):
    """A noise block whose variance rises along its grid's last axis."""

    def _publish(self, conditional_residual):
        shape = conditional_residual.grid.channel_shape
        ramp = self.level * (1.0 + np.linspace(0.0, 1.0, shape[-1]))
        values = np.broadcast_to(ramp, shape).astype(np.float64)
        if conditional_residual.data_domain == "frequency":
            values = values.astype(np.complex128)
        return Covariance.like(
            conditional_residual,
            {ch: values.copy() for ch in conditional_residual.channel_names},
        )


def snapshot(wheel):
    """Everything a refused adoption must leave alone."""
    total = wheel.covariance
    return {
        "names": list(wheel.ledger),
        "roles": dict(wheel._roles),
        "grids": dict(wheel._block_domains),
        "blocks": list(wheel._blocks),
        "block_indices": dict(wheel._block_indices),
        "residual": wheel._working_residual.channel_data["A"].copy(),
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
    assert after["roles"] == before["roles"]
    assert after["grids"] == before["grids"]
    assert [name for name, _ in after["blocks"]] == [
        name for name, _ in before["blocks"]
    ]
    for (_, after_block), (_, before_block) in zip(
        after["blocks"], before["blocks"], strict=True
    ):
        assert after_block is before_block
    assert after["block_indices"] == before["block_indices"]
    assert after["rng"] == before["rng"]
    assert after["signal_sum"] is before["signal_sum"]
    assert after["noise_sum"] is before["noise_sum"]
    np.testing.assert_array_equal(after["residual"], before["residual"])
    _same_array(after["covariance"], before["covariance"])
    assert after["ledger"].keys() == before["ledger"].keys()
    for name, (template, covariance, parameters) in before["ledger"].items():
        after_template, after_covariance, after_parameters = after["ledger"][name]
        _same_array(after_template, template)
        _same_array(after_covariance, covariance)
        assert after_parameters == parameters


class TestConstruction:
    def test_time_or_frequency_observations_need_a_division_count(self, observed):
        for data in (observed, observed.to_frequency()):
            with pytest.raises(ValueError) as info:
                Wheel(data)
            message = str(info.value)
            assert "num_frequency_divisions" in message
            assert "num_time_divisions" in message
            assert f"data_domain={data.data_domain!r}" in message

    def test_the_home_grid_is_resolved_from_either_count(self, observed):
        expected = Grid(
            "wdm", 64, 0.5, WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
        )
        assert Wheel(observed, num_frequency_divisions=8).home_grid == expected
        assert Wheel(observed, num_time_divisions=8).home_grid == expected
        both = Wheel(observed, num_frequency_divisions=8, num_time_divisions=8)
        assert both.home_grid == expected
        by_columns = Wheel(observed.to_frequency(), num_time_divisions=16)
        assert by_columns.home_grid.wdm_grid.num_frequency_divisions == 4
        with pytest.raises(ValueError, match="product"):
            Wheel(observed, num_frequency_divisions=8, num_time_divisions=4)
        with pytest.raises(ValueError, match="positive even"):
            Wheel(observed, num_frequency_divisions=3)
        with pytest.raises(ValueError, match="divide"):
            Wheel(observed, num_frequency_divisions=6)

    def test_wdm_observations_bring_their_own_home_grid(self, observed):
        cells = home_of(observed)
        assert Wheel(cells).home_grid == cells.grid
        assert Wheel(cells, num_frequency_divisions=8).home_grid == cells.grid
        assert Wheel(cells, num_time_divisions=8).home_grid == cells.grid
        with pytest.raises(ValueError) as info:
            Wheel(cells, num_frequency_divisions=4)
        message = str(info.value)
        assert "num_frequency_divisions=8" in message
        assert "num_frequency_divisions=4" in message

    def test_the_old_signatures_stay_refused(self, observed):
        with pytest.raises(TypeError, match="positional"):
            Wheel(observed, 8)
        wheel = Wheel(observed, num_frequency_divisions=8)
        assert not hasattr(wheel, "_covariance_grid")
        assert not hasattr(wheel, "_observed_data")
        with pytest.raises(TypeError, match="role"):
            wheel.add(EchoBlock("echo"), role="signal")

    @pytest.mark.parametrize("domain", ["time", "frequency", "wdm"])
    def test_public_views_come_back_on_the_arrival_grid(self, observed, domain):
        data = (
            observed
            if domain == "time"
            else observed.to_frequency()
            if domain == "frequency"
            else home_of(observed)
        )
        wheel = Wheel(data, num_frequency_divisions=NF)
        assert wheel.observed_data.grid == data.grid
        np.testing.assert_array_equal(
            wheel.observed_data.channel_data["A"], data.channel_data["A"]
        )
        for view in (wheel.working_residual, wheel.residual()):
            assert view.grid == data.grid
            assert view.channel_data["A"].dtype == data.channel_data["A"].dtype
            np.testing.assert_allclose(
                view.channel_data["A"], data.channel_data["A"], **CLOSE
            )
            assert not np.shares_memory(
                view.channel_data["A"], wheel._working_residual.channel_data["A"]
            )
        if domain == "wdm":  # on the home grid the views are bit-exact copies
            np.testing.assert_array_equal(
                wheel.residual().channel_data["A"], data.channel_data["A"]
            )


class TestMixedGridCampaign:
    def campaign(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=1)
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        blocks = {
            "time": LevelSignal("time", 1.0),
            "spectral": LevelSignal("spectral", 2.0),
            "home": LevelSignal("home", 4.0),
            "coarse": LevelSignal("coarse", 8.0),
        }
        wheel.add(blocks["time"])
        wheel.add(blocks["spectral"], data_domain="frequency")
        wheel.add(blocks["home"], data_domain="wdm")
        wheel.add(blocks["coarse"], data_domain="wdm", num_frequency_divisions=4)
        return wheel, blocks

    def test_blocks_on_four_grids_share_one_campaign(self, observed):
        wheel, blocks = self.campaign(observed)
        wheel.run(2)
        total = 1.0 + 2.0 + 4.0 + 8.0
        np.testing.assert_allclose(
            wheel.residual().channel_data["A"],
            observed.channel_data["A"] - total,
            **CLOSE,
        )
        np.testing.assert_allclose(
            wheel.residual("coarse").channel_data["A"],
            observed.channel_data["A"] - (total - 8.0),
            **CLOSE,
        )
        for name, block in blocks.items():
            contribution = wheel.contribution(name)["E"]
            assert contribution.shape == (64,)  # the arrival grid
            np.testing.assert_allclose(contribution, block.value, rtol=1e-12)
            residual, _ = block.state.received[-1]
            expected_grid = wheel._block_domains[name]
            assert residual.grid == expected_grid
            np.testing.assert_allclose(
                transform(residual, "time").channel_data["A"],
                observed.channel_data["A"] - (total - block.value),
                **CLOSE,
            )
            current = block.state.states[-1]
            assert current.template["A"].shape == expected_grid.channel_shape
            np.testing.assert_allclose(
                current.template["A"],
                level_state(residual, block.value).template["A"],
                **CLOSE,
            )
        assert blocks["spectral"].state.received[-1][0].data_domain == "frequency"
        assert (
            blocks["coarse"].state.received[-1][0].wdm_grid.num_frequency_divisions == 4
        )
        assert blocks["home"].state.received[-1][0].grid == wheel.home_grid

    def test_the_ledger_keeps_templates_on_the_home_grid(self, observed):
        wheel, blocks = self.campaign(observed)
        for name, block in blocks.items():
            template = wheel.ledger[name].template["A"]
            assert template.shape == (NF, 64 // NF)
            np.testing.assert_allclose(
                template,
                level_state(home_of(observed), block.value).template["A"],
                **CLOSE,
            )
        assert wheel.ledger["floor"].template is None

    def test_every_block_receives_the_white_floor_on_its_grid(self, observed):
        wheel, blocks = self.campaign(observed)
        wheel.run(1)
        for block in blocks.values():
            for residual, covariance in block.state.received:
                assert type(covariance) is Covariance
                assert covariance.grid == residual.grid
                assert not covariance.blocks.flags.writeable
                np.testing.assert_allclose(
                    covariance.A, white_covariance(residual, 1.0).A, rtol=1e-12
                )
        assert wheel.covariance.grid == wheel.home_grid
        np.testing.assert_allclose(wheel.covariance.A, 1.0 / observed.fs, rtol=1e-12)


class TestNoiseSum:
    def test_a_signal_block_off_the_home_grid_receives_exactly_the_translated_sum(
        self, observed
    ):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        wheel.add(WhiteNoise("white", 1.0))
        wheel.add(ColouredNoise("coloured", 2.0), data_domain="frequency")
        spectral, coarse = LevelSignal("spectral", 1.0), LevelSignal("coarse", 1.0)
        wheel.add(spectral, data_domain="frequency")
        wheel.add(coarse, data_domain="wdm", num_frequency_divisions=4)
        wheel.run(1)
        total = wheel.covariance
        assert total.grid == wheel.home_grid
        for block, target, options in (
            (spectral, "frequency", {}),
            (coarse, "wdm", {"num_frequency_divisions": 4}),
        ):
            expected = transform(total, target, **options)
            for _, received in block.state.received:
                assert received.grid == expected.grid
                np.testing.assert_array_equal(received.blocks, expected.blocks)

    def test_a_noise_block_receives_the_others_translated_and_its_own_as_published(
        self, observed
    ):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        white = WhiteNoise("white", 1.0)
        coloured = ColouredNoise("coloured", 2.0)
        wheel.add(white)
        wheel.add(coloured, data_domain="frequency")
        wheel.run(2)
        spectrum = observed.to_frequency()
        assert white.received[0][0] is None  # alone at registration
        assert len(coloured.received) == len(coloured.published) == 3
        for received, _ in coloured.received:  # white's floor on its grid
            assert received.grid == spectrum.grid
            np.testing.assert_allclose(
                received.A, white_covariance(spectrum, 1.0).A, rtol=1e-12
            )
        for (_, current), published in zip(
            coloured.received[1:], coloured.published[:-1], strict=True
        ):
            assert current.template is None
            assert current.covariance.grid == spectrum.grid
            np.testing.assert_array_equal(current.covariance.blocks, published.blocks)
        for (received, _), published in zip(
            white.received[1:], coloured.published[:-1], strict=True
        ):  # coloured's accepted model: onto the home grid, then to the time grid
            assert received.grid == observed.grid
            via_home = transform(published, "wdm", num_frequency_divisions=NF)
            np.testing.assert_array_equal(
                received.blocks, transform(via_home, "time").blocks
            )

    def test_the_ledger_keeps_the_native_publication_and_the_sum_lives_at_home(
        self, observed
    ):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        wheel.add(WhiteNoise("white", 1.0))
        coloured = ColouredNoise("coloured", 2.0)
        wheel.add(coloured, data_domain="frequency")
        native = wheel.ledger["coloured"].covariance
        assert native.grid == observed.to_frequency().grid
        np.testing.assert_array_equal(native.blocks, coloured.published[-1].blocks)
        assert wheel.ledger["white"].covariance.grid == observed.grid
        assert wheel.covariance.grid == wheel.home_grid
        expected = transform(
            wheel.ledger["white"].covariance, "wdm", num_frequency_divisions=NF
        )
        expected = expected + transform(native, "wdm", num_frequency_divisions=NF)
        np.testing.assert_allclose(wheel.covariance.blocks, expected.blocks, rtol=1e-12)
        # the hand-over copies are independent of the stored sum
        total = wheel.covariance
        total.blocks.setflags(write=True)
        total.A[:] = 99.0
        np.testing.assert_allclose(wheel.covariance.blocks, expected.blocks, rtol=1e-12)

    def test_a_republished_covariance_replaces_its_leaf(self, observed):
        class Growing(WhiteNoise):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                self.level += 1.0
                return super().sample(
                    conditional_residual, covariance, current_block_state, rng=rng
                )

        wheel = Wheel(observed, num_frequency_divisions=NF)
        wheel.add(Growing("growing", 1.0), data_domain="frequency")
        wheel.add(FixedNoiseBlock("fixed", white_covariance(observed, 10.0)))
        per_cell = 1.0 / observed.fs
        np.testing.assert_allclose(wheel.covariance.A, 11.0 * per_cell, rtol=1e-12)
        wheel.run(1)
        np.testing.assert_allclose(wheel.covariance.A, 12.0 * per_cell, rtol=1e-12)
        wheel.run(1)
        np.testing.assert_allclose(wheel.covariance.A, 13.0 * per_cell, rtol=1e-12)
        np.testing.assert_allclose(
            wheel.ledger["growing"].covariance.A,
            white_covariance(observed.to_frequency(), 3.0).A,
        )

    def test_wdm_observations_with_home_grid_blocks_are_bit_exact(self, observed):
        cells = home_of(observed)
        wheel = Wheel(cells)
        noise, signal = WhiteNoise("noise", 1.0), LevelSignal("signal", 1.0)
        wheel.add(noise)
        wheel.add(signal, data_domain="wdm")
        wheel.run(1)
        residual, covariance = signal.state.received[-1]
        assert residual.grid == cells.grid
        template = signal.state.states[-1].template["A"]
        np.testing.assert_array_equal(
            residual.channel_data["A"], cells.channel_data["A"]
        )  # its own template is left in
        np.testing.assert_array_equal(
            wheel.residual().channel_data["A"], cells.channel_data["A"] - template
        )
        np.testing.assert_array_equal(covariance.blocks, noise.published[-1].blocks)
        np.testing.assert_array_equal(
            wheel.covariance.blocks, noise.published[-1].blocks
        )
        np.testing.assert_array_equal(wheel.contribution("signal")["A"], template)

    def test_a_noise_block_on_another_wdm_grid_is_kept_native_and_summed_at_home(
        self, observed
    ):
        coarse = transform(observed, "wdm", num_frequency_divisions=4)
        wheel = Wheel(observed, num_frequency_divisions=NF)
        wheel.add(
            FixedNoiseBlock("floor", white_covariance(coarse, 2.0)),
            data_domain="wdm",
            num_frequency_divisions=4,
        )
        signal = LevelSignal("coarse", 1.0)
        wheel.add(signal, data_domain="wdm", num_frequency_divisions=4)
        wheel.run(1)
        native = wheel.ledger["floor"].covariance
        assert native.grid == coarse.grid
        np.testing.assert_array_equal(
            native.blocks, white_covariance(coarse, 2.0).blocks
        )
        assert wheel.covariance.grid == wheel.home_grid
        np.testing.assert_allclose(
            wheel.covariance.A, white_covariance(home_of(observed), 2.0).A, rtol=1e-12
        )
        assert len(signal.state.received) == 2
        for residual, received in signal.state.received:
            assert residual.grid == received.grid == coarse.grid
            np.testing.assert_allclose(
                received.A, white_covariance(coarse, 2.0).A, rtol=1e-12
            )


class TestHomeGridHandOver:
    """A block on the home grid receives copies, never the stored objects."""

    def test_a_home_grid_noise_block_cannot_corrupt_the_stored_covariance(
        self, observed
    ):
        class Corrupting(WhiteNoise):
            """Overwrites every array of the covariance it receives, then publishes."""

            @staticmethod
            def _corrupt(covariance):
                if covariance is not None:
                    covariance.blocks.flags.writeable = True
                    covariance.blocks[...] = 999.0

            def initial_sample(self, conditional_residual, covariance, *, rng):
                self._corrupt(covariance)
                return super().initial_sample(conditional_residual, covariance, rng=rng)

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                self._corrupt(covariance)
                return super().sample(
                    conditional_residual, covariance, current_block_state, rng=rng
                )

        cells = home_of(observed)
        wheel = Wheel(observed, num_frequency_divisions=NF)
        first, second = WhiteNoise("first", 1.0), Corrupting("second", 3.0)
        wheel.add(first, data_domain="wdm")
        first_published = wheel.ledger["first"].covariance.blocks.copy()
        total = (white_covariance(cells, 1.0) + white_covariance(cells, 3.0)).blocks
        wheel.add(second, data_domain="wdm")
        wheel.run(2)
        assert len(second.received) == 3
        for received, _ in second.received:  # each was its own copy to overwrite
            assert received.grid == wheel.home_grid
            np.testing.assert_array_equal(received.blocks, 999.0)
        np.testing.assert_array_equal(
            wheel.ledger["first"].covariance.blocks, first_published
        )
        np.testing.assert_array_equal(wheel.covariance.blocks, total)

    def test_a_home_grid_block_cannot_corrupt_the_residual_through_a_failed_start(
        self, observed
    ):
        class Scribbler(LevelSignal):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                values = conditional_residual.channel_data["A"]
                values.flags.writeable = True
                values[...] = 999.0
                raise RuntimeError("scribbled on its residual, then failed")

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        wheel.add(LevelSignal("time", 1.0))
        before = snapshot(wheel)
        with pytest.raises(RuntimeError, match="scribbled on its residual"):
            wheel.add(Scribbler("scribbler", 1.0), data_domain="wdm")
        assert_unchanged(wheel, before)
        assert "scribbler" not in wheel._roles

    def test_a_hand_over_that_does_not_translate_is_refused_before_the_block_runs(
        self, observed
    ):
        cells = home_of(observed)
        huge = Covariance.like(
            cells,
            {
                ch: np.full(cells.grid.channel_shape, 5e306)
                for ch in cells.channel_names
            },
        )
        spectrum = observed.to_frequency().grid

        def refusal(name):
            return pytest.raises(
                ValueError,
                match=re.escape(
                    f"covariance for block {name!r} does not translate onto its "
                    f"grid {spectrum}: "
                )
                + ".*finite",
            )

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        early = LevelSignal("early", 1.0)
        wheel.add(early, data_domain="frequency")  # before any noise: receives None
        wheel.add(FixedNoiseBlock("floor", huge), data_domain="wdm")
        before = snapshot(wheel)
        late = LevelSignal("late", 1.0)
        # The home-grid floor overflows on the frequency grid's bins.
        with np.errstate(over="ignore", invalid="ignore"), refusal("late"):
            wheel.add(late, data_domain="frequency")
        assert late.state.received == []
        assert_unchanged(wheel, before)
        with np.errstate(over="ignore", invalid="ignore"), refusal("early"):
            wheel.run(1)
        assert len(early.state.received) == 1  # its registration only
        assert_unchanged(wheel, before)


class TestDeclaredRoles:
    def test_the_role_is_read_once_at_add(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        block = LevelSignal("turncoat", 1.0)
        wheel.add(block)
        block.role = "noise"
        assert wheel._roles == {"turncoat": "signal"}
        wheel.run(1)  # still adopted as a signal block
        np.testing.assert_allclose(wheel.contribution("turncoat")["A"], 1.0, rtol=1e-12)
        block.role = "nosie"
        wheel.run(1)
        assert len(block.state.states) == 2

    def test_a_block_without_a_role_is_refused_after_the_protocol_checks(
        self, observed
    ):
        class NoRole:
            name = "plain"

            def initial_sample(self, conditional_residual, covariance, *, rng):
                raise AssertionError(
                    "add must refuse the role before starting the block"
                )

            def state_from_parameters(self, parameters, reference_data):
                # Stub: no test rebuilds this block's state.
                return BlockState(parameters=dict(parameters))

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                raise AssertionError("unreachable")

        class NoneRole(NoRole):
            role = None

        class NoSample:
            name = "nosample"

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        before = snapshot(wheel)
        for block in (NoRole(), NoneRole()):
            with pytest.raises(TypeError) as info:
                wheel.add(block)
            message = str(info.value)
            assert message.startswith("block 'plain' declares no role; ")
            assert "role ('signal' or 'noise')" in message
            assert "initial_sample(conditional_residual, covariance, *, rng)" in message
            assert message.endswith(
                "state_from_parameters(parameters, reference_data)."
            )
        with pytest.raises(TypeError, match="does not implement sample"):
            wheel.add(NoSample())  # the protocol checks still come first
        assert not isinstance(NoRole(), Block)
        assert isinstance(LevelSignal("x", 1.0), Block)
        assert_unchanged(wheel, before)

    @pytest.mark.parametrize("declared", ["nosie", "Signal", "noise ", "", 1])
    def test_an_invalid_role_is_refused_with_its_value(self, observed, declared):
        block = LevelSignal("typo", 1.0)
        block.role = declared
        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        before = snapshot(wheel)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"block 'typo' declares role {declared!r}; "
                "a role is 'signal' or 'noise'"
            ),
        ):
            wheel.add(block)
        assert block.state.received == []
        assert_unchanged(wheel, before)

    def test_check_block_refuses_a_signal_block_among_the_noise_blocks(self, observed):
        class NeverStarted(LevelSignal):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                raise AssertionError("check_block must refuse noise_blocks first")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "noise_blocks entry 'echo' declares role 'signal'; every block in "
                "noise_blocks must declare role 'noise'"
            ),
        ):
            check_block(
                NeverStarted("under_test", 1.0),
                observed,
                noise_blocks=(EchoBlock("echo"),),
                num_frequency_divisions=NF,
            )

    def test_check_block_passes_the_division_counts_to_the_wheel(self, observed):
        check_block(EchoBlock("echo"), observed, num_frequency_divisions=NF)
        check_block(EchoBlock("echo"), observed, num_time_divisions=16)
        check_block(EchoBlock("echo"), home_of(observed))
        with pytest.raises(ValueError, match="num_frequency_divisions or num_time"):
            check_block(EchoBlock("echo"), observed)
        with pytest.raises(ValueError, match="num_frequency_divisions=4"):
            check_block(EchoBlock("echo"), home_of(observed), num_frequency_divisions=4)


class TestRefusals:
    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_noise_block_publishing_off_its_grid_is_refused_with_rollback(
        self, observed, method
    ):
        class OffGrid(WhiteNoise):
            def __init__(self, name, level, offend_in):
                super().__init__(name, level)
                self.offend_in = offend_in

            def _publish(self, conditional_residual, offending):
                data = (
                    transform(conditional_residual, "wdm", num_frequency_divisions=4)
                    if offending
                    else conditional_residual
                )
                return BlockState(covariance=white_covariance(data, self.level))

            def initial_sample(self, conditional_residual, covariance, *, rng):
                return self._publish(
                    conditional_residual, self.offend_in == "initial_sample"
                )

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return self._publish(conditional_residual, self.offend_in == "sample")

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        block = OffGrid("spectral", 1.0, method)
        if method == "sample":
            wheel.add(block, data_domain="frequency")
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        before = snapshot(wheel)
        with pytest.raises(ValueError) as info:
            if method == "initial_sample":
                wheel.add(block, data_domain="frequency")
            else:
                wheel.run(1)
        message = str(info.value)
        assert message.startswith(f"spectral.{method} covariance is on grid ")
        assert "data_domain='wdm'" in message and "data_domain='frequency'" in message
        assert (
            "publish the covariance on the grid of the data the block received"
            in message
        )
        assert_unchanged(wheel, before)

    def test_a_covariance_that_fails_to_translate_is_refused_with_rollback(
        self, observed, monkeypatch
    ):
        real_transform = enchilada.wheel.transform

        def failing(value, target_domain, **options):
            if isinstance(value, Covariance) and target_domain == "wdm":
                raise ValueError("synthetic: the translated blocks must be finite")
            return real_transform(value, target_domain, **options)

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(
            FixedNoiseBlock("floor", white_covariance(observed, 1.0))
        )  # time grid
        before = snapshot(wheel)
        monkeypatch.setattr(enchilada.wheel, "transform", failing)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "spectral.initial_sample covariance does not translate onto the "
                f"home grid {wheel.home_grid}: synthetic: the translated blocks "
                "must be finite"
            ),
        ):
            wheel.add(WhiteNoise("spectral", 1.0), data_domain="frequency")
        assert_unchanged(wheel, before)
        assert "spectral" not in wheel._roles
        with pytest.raises(
            ValueError, match="floor.sample covariance does not translate"
        ):
            wheel.run(1)  # floor republishes on the time grid: translated at adoption
        assert_unchanged(wheel, before)

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_template_is_validated_on_the_block_grid(self, observed, method):
        class WrongShape(LevelSignal):
            def __init__(self, name, offend_in):
                super().__init__(name, 1.0)
                self.offend_in = offend_in

            def _publish(self, conditional_residual, offending):
                if not offending:
                    return level_state(conditional_residual, 1.0)
                return BlockState(
                    template={
                        ch: np.zeros(64) for ch in conditional_residual.channel_names
                    }
                )

            def initial_sample(self, conditional_residual, covariance, *, rng):
                return self._publish(
                    conditional_residual, self.offend_in == "initial_sample"
                )

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return self._publish(conditional_residual, self.offend_in == "sample")

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        block = WrongShape("spectral", method)
        if method == "sample":
            wheel.add(block, data_domain="frequency")
        wheel.add(LevelSignal("time", 1.0))
        before = snapshot(wheel)
        with pytest.raises(
            ValueError,
            match=re.escape(f"spectral.{method} template") + ".*length 64, expected 33",
        ):
            if method == "initial_sample":
                wheel.add(block, data_domain="frequency")
            else:
                wheel.run(1)
        assert_unchanged(wheel, before)

    def test_a_non_finite_template_is_refused_before_it_is_transformed(self, observed):
        class NaNSignal(LevelSignal):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                state = level_state(conditional_residual, 1.0)
                state.template["A"][0] = np.nan
                return state

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(NaNSignal("spectral", 1.0), data_domain="frequency")
        before = snapshot(wheel)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "spectral.sample returned non-finite samples in channel 'A'"
            ),
        ):
            wheel.run(1)
        assert_unchanged(wheel, before)

    @pytest.mark.parametrize("domain", ["time", "frequency"])
    def test_division_counts_off_wdm_are_refused_before_the_block_runs(
        self, observed, domain
    ):
        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        wheel.add(LevelSignal("time", 1.0))
        before = snapshot(wheel)
        block = LevelSignal("spectral", 1.0)
        with pytest.raises(
            ValueError,
            match=re.escape("division counts apply only to target_domain='wdm'"),
        ):
            wheel.add(block, data_domain=domain, num_frequency_divisions=4)
        assert block.state.received == []
        assert "spectral" not in wheel.ledger
        assert_unchanged(wheel, before)
