"""The campaign output file: the recorder, the reader and the Wheel that writes it."""

import re
from dataclasses import replace

import h5py
import numpy as np
import pytest

from conftest import make_observed, white_covariance
from enchilada import BlockState, Grid, L2Data, WDMGrid, Wheel, transform
from enchilada.l2 import (
    _block_grid_from_attrs,
    _grid_from_attrs,
    _Recorder,
    _Schema,
)


def recorder(tmp_path, observed, name="campaign.h5"):
    return _Recorder.create(tmp_path / name, observed)


class TestRecorder:
    def test_create_writes_the_format_and_the_observation_metadata(
        self, tmp_path, observed
    ):
        rec = recorder(tmp_path, observed)
        rec.close()
        with h5py.File(tmp_path / "campaign.h5", "r") as file:
            assert file.attrs["format"] == "enchilada-l2"
            assert int(file.attrs["format_version"]) == 1
            assert isinstance(file.attrs["enchilada_version"], str)
            assert int(file.attrs["num_steps"]) == 0
            obs = file["observation"].attrs
            assert obs["data_domain"] == "time"
            assert int(obs["num_time_samples"]) == 64
            assert float(obs["sample_rate_hz"]) == 0.5
            assert float(obs["start_time_gps"]) == 0.0
            assert obs["tdi_generation"] == "2.0"
            assert obs["physical_observable"] == "fractional_frequency"
            assert list(obs["channel_names"]) == ["A", "E", "T"]
            assert "num_frequency_divisions" not in obs
            assert list(file["blocks"]) == []

    def test_a_wdm_observation_records_its_divisions(self, tmp_path, observed):
        wdm = transform(observed, "wdm", num_frequency_divisions=8)
        rec = recorder(tmp_path, wdm)
        rec.close()
        with h5py.File(tmp_path / "campaign.h5", "r") as file:
            obs = file["observation"].attrs
            assert obs["data_domain"] == "wdm"
            assert int(obs["num_frequency_divisions"]) == 8
            assert int(obs["num_time_divisions"]) == 8

    def test_an_existing_file_is_refused(self, tmp_path, observed):
        recorder(tmp_path, observed).close()
        with pytest.raises(FileExistsError, match="name a new path"):
            recorder(tmp_path, observed)

    def test_a_failed_create_leaves_no_file(self, tmp_path, observed, monkeypatch):
        def refuse(file, observed):
            raise RuntimeError("boom")

        monkeypatch.setattr(_Recorder, "initialize", refuse)
        with pytest.raises(RuntimeError, match="boom"):
            recorder(tmp_path, observed)
        assert not (tmp_path / "campaign.h5").exists()

    def test_encode_fixes_the_schema_at_registration(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        encoded = rec.encode(
            "b",
            "initial_sample",
            {
                "amplitude": 1.5,
                "count": 3,
                "flags": np.array([True, False]),
                "sources": np.zeros((2, 3), dtype=np.float32),
                "phase": 1j,
            },
            registering=True,
        )
        assert encoded.schema.keys == (
            "amplitude",
            "count",
            "flags",
            "sources",
            "phase",
        )
        assert encoded.schema.kinds == ("f", "i", "b", "f", "c")
        assert encoded.schema.ndims == (0, 0, 1, 2, 0)
        assert [v.dtype for v in encoded.values] == [
            np.dtype(np.float64),
            np.dtype(np.int64),
            np.dtype(bool),
            np.dtype(np.float64),
            np.dtype(np.complex128),
        ]
        assert encoded.values[3].shape == (2, 3)
        assert encoded.values[0].shape == ()

    @pytest.mark.parametrize(
        ("parameters", "error", "message"),
        [
            ({1: 2.0}, TypeError, "non-string key 1"),
            ({"a/b": 2.0}, ValueError, "parameter name 'a/b' contains '/'"),
            ({"": 2.0}, ValueError, "parameter name '' cannot name"),
            (
                {"label": "two"},
                TypeError,
                "dtype <U3; recorded parameters must be bool, integer, float or "
                "complex",
            ),
            ({"nested": {"a": 1}}, TypeError, "recorded parameters must be"),
            ({"nothing": None}, TypeError, "recorded parameters must be"),
            ({"ragged": [[1], [1, 2]]}, TypeError, "is not a regular numeric array"),
            ({"a\x00b": 1.0}, ValueError, "NUL character"),
        ],
    )
    def test_encode_refuses_unrecordable_values(
        self, tmp_path, observed, parameters, error, message
    ):
        rec = recorder(tmp_path, observed)
        with pytest.raises(error, match=re.escape(message)):
            rec.encode("b", "initial_sample", parameters, registering=True)

    def test_encode_holds_later_entries_to_the_schema(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        start = rec.encode(
            "b",
            "initial_sample",
            {"amplitude": 1.0, "sources": np.zeros(2)},
            registering=True,
        )
        rec.add_block("b", role="signal", index=0, grid=observed.grid, encoded=start)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "b.sample parameters have keys ['amplitude'] but 'b' registered "
                "['amplitude', 'sources']; the key set is fixed at registration"
            ),
        ):
            rec.encode("b", "sample", {"amplitude": 2.0}, registering=False)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "b.sample parameters['amplitude'] has kind 'i' but was registered with "
                "kind 'f'; publish the same kind every call"
            ),
        ):
            rec.encode(
                "b",
                "sample",
                {"amplitude": 2, "sources": np.zeros(2)},
                registering=False,
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "b.sample parameters['sources'] has 2 dimensions but was registered "
                "with 1; shapes may change between calls, the number of dimensions "
                "may not"
            ),
        ):
            rec.encode(
                "b",
                "sample",
                {"amplitude": 2.0, "sources": np.zeros((2, 2))},
                registering=False,
            )
        narrow_and_empty = rec.encode(
            "b",
            "sample",
            {"amplitude": np.float32(2.0), "sources": np.zeros(0)},
            registering=False,
        )
        assert narrow_and_empty.values[0].dtype == np.float64
        assert narrow_and_empty.values[1].shape == (0,)
        rec.encode(
            "b",
            "sample",
            {"amplitude": np.nan, "sources": np.full(5, np.inf)},
            registering=False,
        )

    def test_add_block_and_append_step_write_the_layout(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "a",
            role="signal",
            index=0,
            grid=observed.grid,
            encoded=rec.encode("a", "initial_sample", {"x": 1.0}, registering=True),
        )
        rec.add_block(
            "pop",
            role="noise",
            index=1,
            grid=Grid(
                "wdm", 64, 0.5, WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
            ),
            encoded=rec.encode(
                "pop",
                "initial_sample",
                {"f": np.array([1.0, 2.0]), "m": np.ones((1, 2))},
                registering=True,
            ),
        )
        rec.append_step(
            {
                "a": rec.encode("a", "sample", {"x": 2.0}, registering=False),
                "pop": rec.encode(
                    "pop",
                    "sample",
                    {"f": np.zeros(0), "m": np.ones((2, 2))},
                    registering=False,
                ),
            }
        )
        rec.close()
        with h5py.File(tmp_path / "campaign.h5", "r") as file:
            assert int(file.attrs["num_steps"]) == 1
            a = file["blocks/a"]
            assert a.attrs["role"] == "signal" and int(a.attrs["index"]) == 0
            assert int(a.attrs["first_step"]) == 0 and int(a.attrs["num_entries"]) == 2
            assert a.attrs["data_domain"] == "time"
            assert list(a.attrs["keys"]) == ["x"]
            assert list(a.attrs["kinds"]) == ["f"]
            assert list(a.attrs["ndims"]) == [0]
            assert "shapes" not in a
            assert a["parameters/x"].maxshape == (None,)
            assert a["parameters/x"].chunks == (1024,)
            np.testing.assert_array_equal(a["parameters/x"][:], [1.0, 2.0])
            pop = file["blocks/pop"]
            assert pop.attrs["data_domain"] == "wdm"
            assert int(pop.attrs["num_frequency_divisions"]) == 8
            assert int(pop.attrs["num_time_divisions"]) == 8
            np.testing.assert_array_equal(pop["shapes"][:], [[2, 1, 2], [0, 2, 2]])
            assert pop["shapes"].dtype == np.int64
            np.testing.assert_array_equal(pop["parameters/f"][:], [1.0, 2.0])
            np.testing.assert_array_equal(
                pop["parameters/m"][:], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )

    def test_first_step_is_the_step_count_at_registration(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "a",
            role="signal",
            index=0,
            grid=observed.grid,
            encoded=rec.encode("a", "initial_sample", {}, registering=True),
        )
        for _ in range(3):
            rec.append_step({"a": rec.encode("a", "sample", {}, registering=False)})
        rec.add_block(
            "late",
            role="signal",
            index=1,
            grid=observed.grid,
            encoded=rec.encode("late", "initial_sample", {}, registering=True),
        )
        rec.close()
        with h5py.File(tmp_path / "campaign.h5", "r") as file:
            assert int(file["blocks/late"].attrs["first_step"]) == 3
            assert int(file["blocks/a"].attrs["num_entries"]) == 4
            assert int(file["blocks/late"].attrs["num_entries"]) == 1

    def test_append_step_requires_every_registered_block(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "a",
            role="signal",
            index=0,
            grid=observed.grid,
            encoded=rec.encode("a", "initial_sample", {}, registering=True),
        )
        with pytest.raises(RuntimeError, match="every registered block"):
            rec.append_step({})

    def test_close_is_idempotent_and_reported(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        assert not rec.closed
        rec.close()
        rec.close()
        assert rec.closed

    def test_initialize_takes_an_open_handle_and_leaves_it_to_the_caller(
        self, tmp_path, observed
    ):
        path = tmp_path / "shared.h5"
        with h5py.File(path, "x") as file:
            rec = _Recorder.initialize(file, observed)
            assert rec.file is file
            rec.add_block(
                "a",
                role="signal",
                index=0,
                grid=observed.grid,
                encoded=rec.encode("a", "initial_sample", {"x": 1.0}, registering=True),
            )
            file.create_group("simulation").attrs["truth"] = "a"
            file.attrs["producer"] = "test"
        assert rec.closed
        with h5py.File(path, "r") as file:
            assert file.attrs["format"] == "enchilada-l2"
            assert int(file.attrs["num_steps"]) == 0
            assert list(file["blocks"]) == ["a"]
            assert int(file["blocks/a"].attrs["num_entries"]) == 1
            assert file["simulation"].attrs["truth"] == "a"

    def test_stored_dtypes_follow_the_kinds(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "a",
            role="signal",
            index=0,
            grid=observed.grid,
            encoded=rec.encode(
                "a",
                "initial_sample",
                {"i": np.int8(3), "c": np.complex64(1j), "b": True, "u": np.uint8(2)},
                registering=True,
            ),
        )
        rec.close()
        with h5py.File(tmp_path / "campaign.h5", "r") as file:
            parameters = file["blocks/a/parameters"]
            assert parameters["i"].dtype == np.int64
            assert parameters["c"].dtype == np.complex128
            assert parameters["b"].dtype == np.dtype(bool)
            assert parameters["u"].dtype == np.uint64

    def test_encode_refuses_an_unregistered_block(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        with pytest.raises(RuntimeError, match="not registered"):
            rec.encode("nope", "sample", {}, registering=False)

    def test_a_failed_write_latches_the_recorder(self, tmp_path, observed, monkeypatch):
        rec = recorder(tmp_path, observed)
        for name, index in (("a", 0), ("b", 1)):
            rec.add_block(
                name,
                role="signal",
                index=index,
                grid=observed.grid,
                encoded=rec.encode(
                    name, "initial_sample", {"x": 1.0}, registering=True
                ),
            )
        pending = {
            name: rec.encode(name, "sample", {"x": 2.0}, registering=False)
            for name in ("a", "b")
        }
        original = _Recorder._append_entry
        calls = []

        def flaky(group, encoded):
            calls.append(group)
            if len(calls) == 2:
                raise OSError("disk full")
            original(group, encoded)

        monkeypatch.setattr(_Recorder, "_append_entry", staticmethod(flaky))
        with pytest.raises(OSError, match="disk full"):
            rec.append_step(pending)
        with pytest.raises(RuntimeError, match="failed part-way"):
            rec.encode("a", "sample", {"x": 2.0}, registering=False)
        with pytest.raises(RuntimeError, match="failed part-way"):
            rec.append_step(pending)
        rec.close()
        with h5py.File(tmp_path / "campaign.h5", "r") as file:
            assert int(file.attrs["num_steps"]) == 0

    def test_a_failed_flush_names_the_step_before_the_failed_one(
        self, tmp_path, observed, monkeypatch
    ):
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "a",
            role="signal",
            index=0,
            grid=observed.grid,
            encoded=rec.encode("a", "initial_sample", {"x": 1.0}, registering=True),
        )
        rec.append_step({"a": rec.encode("a", "sample", {"x": 2.0}, registering=False)})

        def failing_flush(self):
            raise OSError("flush failed")

        monkeypatch.setattr(h5py.File, "flush", failing_flush)
        with pytest.raises(OSError, match="flush failed"):
            rec.append_step(
                {"a": rec.encode("a", "sample", {"x": 3.0}, registering=False)}
            )
        # num_steps was already set to 2 in memory; only step 1 is surely flushed.
        with pytest.raises(RuntimeError, match="consistent up to step 1 "):
            rec.encode("a", "sample", {"x": 4.0}, registering=False)
        monkeypatch.undo()
        rec.close()

    def test_a_failed_add_block_leaves_no_partial_group(
        self, tmp_path, observed, monkeypatch
    ):
        rec = recorder(tmp_path, observed)
        encoded = rec.encode("a", "initial_sample", {"x": 1.0}, registering=True)

        def refuse(group, encoded):
            raise OSError("disk full")

        monkeypatch.setattr(_Recorder, "_append_entry", staticmethod(refuse))
        with pytest.raises(OSError, match="disk full"):
            rec.add_block(
                "a", role="signal", index=0, grid=observed.grid, encoded=encoded
            )
        with pytest.raises(RuntimeError, match="failed part-way"):
            rec.encode("a", "sample", {"x": 1.0}, registering=False)
        assert list(rec.file["blocks"]) == []
        rec.close()
        with pytest.raises(RuntimeError, match="failed part-way"):
            rec.encode("a", "sample", {"x": 1.0}, registering=False)

    def test_append_step_refuses_an_entry_for_another_schema(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        for index, name in enumerate(("a", "b")):
            rec.add_block(
                name,
                role="signal",
                index=index,
                grid=observed.grid,
                encoded=rec.encode(
                    name, "initial_sample", {"x": 1.0}, registering=True
                ),
            )
        with pytest.raises(RuntimeError, match="another block's schema"):
            rec.append_step(
                {
                    "a": rec.encode("b", "sample", {"x": 1.0}, registering=False),
                    "b": rec.encode("b", "sample", {"x": 1.0}, registering=False),
                }
            )


class TestAttrsRoundTrip:
    @pytest.mark.parametrize(
        "schema",
        [
            _Schema((), (), ()),
            _Schema(("a", "b"), ("f", "c"), (0, 2)),
        ],
    )
    def test_schema(self, tmp_path, schema):
        with h5py.File(tmp_path / "s.h5", "w") as file:
            schema.write(file.attrs)
            assert _Schema.read(file.attrs) == schema

    def test_grids(self, tmp_path, observed):
        wdm = transform(observed, "wdm", num_frequency_divisions=8)
        for data in (observed, wdm):
            rec = _Recorder.create(tmp_path / f"{data.data_domain}.h5", data)
            attrs = rec.file["observation"].attrs
            assert _grid_from_attrs(attrs) == data.grid
            assert _block_grid_from_attrs(attrs, observed.grid) == Grid(
                data.data_domain,
                observed.num_time_samples,
                observed.sample_rate_hz,
                data.wdm_grid,
            )
            rec.close()


class Amplitude:
    """Signal block: a constant template of `amplitude` per channel on its grid."""

    def __init__(self, name, start, step=0.5):
        self.name, self.start, self.step = name, start, step

    def _state(self, data, amplitude):
        return data.block_state(
            {ch: np.full_like(arr, amplitude) for ch, arr in data.channel_data.items()},
            parameters={"amplitude": float(amplitude)},
        )

    def state_from_parameters(self, parameters, reference_data):
        return self._state(reference_data, parameters["amplitude"])

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self._state(conditional_residual, self.start)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return self._state(
            conditional_residual,
            current_block_state.parameters["amplitude"] + self.step,
        )


class Level:
    """Noise block: white noise of variance `parameters['variance']`, +1 per sample."""

    def __init__(self, name, start):
        self.name, self.start = name, start

    def state_from_parameters(self, parameters, reference_data):
        variance = float(parameters["variance"])
        return BlockState(
            covariance=white_covariance(reference_data, variance),
            parameters={"variance": variance},
        )

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self.state_from_parameters(
            {"variance": self.start}, conditional_residual
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return self.state_from_parameters(
            {"variance": current_block_state.parameters["variance"] + 1.0},
            conditional_residual,
        )


class Population:
    """Reversible-jump stand-in: the `frequencies` array changes length each call."""

    name = "pop"

    def state_from_parameters(self, parameters, reference_data):
        return BlockState(
            parameters={
                "frequencies": np.asarray(parameters["frequencies"], dtype=float)
            }
        )

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self.state_from_parameters(
            {"frequencies": np.array([1.0, 2.0])}, conditional_residual
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        count = (len(current_block_state.parameters["frequencies"]) + 1) % 4
        return self.state_from_parameters(
            {"frequencies": np.arange(count, dtype=float)}, conditional_residual
        )


def written_file(tmp_path, observed):
    """A hand-written file: 'a' from step 0, 'late' from step 2, 3 steps."""
    rec = recorder(tmp_path, observed, "written.h5")
    a = Amplitude("a", 1.0)
    rec.add_block(
        "a",
        role="signal",
        index=0,
        grid=observed.grid,
        encoded=rec.encode("a", "initial_sample", {"amplitude": 1.0}, registering=True),
    )
    for step in (1, 2):
        rec.append_step(
            {
                "a": rec.encode(
                    "a", "sample", {"amplitude": 1.0 + 0.5 * step}, registering=False
                )
            }
        )
    rec.add_block(
        "late",
        role="noise",
        index=1,
        grid=observed.grid,
        encoded=rec.encode(
            "late", "initial_sample", {"variance": 2.0}, registering=True
        ),
    )
    rec.append_step(
        {
            "a": rec.encode("a", "sample", {"amplitude": 2.5}, registering=False),
            "late": rec.encode("late", "sample", {"variance": 3.0}, registering=False),
        }
    )
    rec.close()
    return tmp_path / "written.h5", a, Level("late", 2.0)


class TestL2Data:
    def test_open_reads_the_metadata(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            assert l2.num_steps == 3
            assert l2.block_names == ("a", "late")
            assert l2.observation_grid == observed.grid
            assert l2.channel_names == ("A", "E", "T")
            assert l2.role("a") == "signal" and l2.role("late") == "noise"
            assert l2.first_step("a") == 0 and l2.first_step("late") == 2
            assert l2.block_grid("late") == observed.grid
        with pytest.raises(ValueError, match="L2Data is closed"):
            _ = l2.num_steps

    def test_open_refuses_other_files_and_mismatched_observations(
        self, tmp_path, observed
    ):
        with h5py.File(tmp_path / "other.h5", "w") as file:
            file.attrs["format"] = "something-else"
        with pytest.raises(ValueError, match="not an enchilada L2 file"):
            L2Data.open(tmp_path / "other.h5")
        path, _, _ = written_file(tmp_path, observed)
        other = make_observed(np.random.default_rng(1), channel_names=("A", "E"))
        with pytest.raises(ValueError, match="channel names"):
            L2Data.open(path, observed_data=other)
        with pytest.raises(ValueError, match="grid"):
            L2Data.open(path, observed_data=observed.to_frequency())

    def test_groups_and_attributes_the_reader_does_not_own_are_ignored(
        self, tmp_path, observed
    ):
        path, _, _ = written_file(tmp_path, observed)
        with h5py.File(path, "r+") as file:
            file.attrs["producer"] = "simulation"
            file.create_group("tdis").create_dataset("A", data=np.zeros(4))
            file.create_group("orbits")
            file.create_group("simulation").attrs["seed"] = 7
        with L2Data.open(path) as l2:
            assert l2.num_steps == 3
            assert l2.block_names == ("a", "late")
            np.testing.assert_array_equal(
                l2.chain("a", "amplitude"), [1.0, 1.5, 2.0, 2.5]
            )

    def test_parameters_and_chains(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            assert l2.parameters("a", 0) == {"amplitude": 1.0}
            assert l2.parameters("a") == {"amplitude": 2.5}
            assert l2.parameters("a", -2) == {"amplitude": 2.0}
            assert isinstance(l2.parameters("a", 1)["amplitude"], np.floating)
            np.testing.assert_array_equal(
                l2.chain("a", "amplitude"), [1.0, 1.5, 2.0, 2.5]
            )
            np.testing.assert_array_equal(l2.chain("late", "variance"), [2.0, 3.0])
            np.testing.assert_array_equal(
                l2.entries("a", "amplitude"), [1.0, 1.5, 2.0, 2.5]
            )
            with pytest.raises(
                IndexError, match="step 4 is out of range; the file holds steps 0 to 3"
            ):
                l2.parameters("a", 4)
            with pytest.raises(IndexError, match="out of range"):
                l2.parameters("a", -5)
            with pytest.raises(
                ValueError,
                match=(
                    "block 'late' was registered at step 2; it has no state at step 1"
                ),
            ):
                l2.parameters("late", 1)
            with pytest.raises(ValueError, match="no parameter 'phase'"):
                l2.chain("a", "phase")
            with pytest.raises(ValueError, match="unknown block"):
                l2.role("nobody")

    def test_ragged_parameters(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "pop",
            role="signal",
            index=0,
            grid=observed.grid,
            encoded=rec.encode(
                "pop",
                "initial_sample",
                {"frequencies": np.array([1.0, 2.0])},
                registering=True,
            ),
        )
        rec.append_step(
            {
                "pop": rec.encode(
                    "pop", "sample", {"frequencies": np.zeros(0)}, registering=False
                )
            }
        )
        rec.append_step(
            {
                "pop": rec.encode(
                    "pop", "sample", {"frequencies": np.array([5.0])}, registering=False
                )
            }
        )
        rec.close()
        with L2Data.open(tmp_path / "campaign.h5") as l2:
            entries = l2.entries("pop", "frequencies")
            assert [e.shape for e in entries] == [(2,), (0,), (1,)]
            np.testing.assert_array_equal(entries[2], [5.0])
            np.testing.assert_array_equal(
                l2.parameters("pop", 1)["frequencies"], np.zeros(0)
            )
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "'pop'.'frequencies' has entries of different shapes; "
                    "use entries() for a list"
                ),
            ):
                l2.chain("pop", "frequencies")

    def test_data_beyond_the_counters_is_ignored(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with h5py.File(path, "r+") as file:  # a record interrupted before its counters
            values = file["blocks/a/parameters/amplitude"]
            values.resize((values.shape[0] + 1,))
            values[-1] = 99.0
        with L2Data.open(path) as l2:
            np.testing.assert_array_equal(
                l2.chain("a", "amplitude"), [1.0, 1.5, 2.0, 2.5]
            )

    def test_an_entry_ahead_of_the_step_counter_is_ignored(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with h5py.File(path, "r+") as file:  # a step that failed after block "a"
            values = file["blocks/a/parameters/amplitude"]
            values.resize((values.shape[0] + 1,))
            values[-1] = 99.0
            file["blocks/a"].attrs["num_entries"] = (
                int(file["blocks/a"].attrs["num_entries"]) + 1
            )
        with L2Data.open(path) as l2:
            assert l2.num_steps == 3
            np.testing.assert_array_equal(
                l2.chain("a", "amplitude"), [1.0, 1.5, 2.0, 2.5]
            )
            assert l2.parameters("a") == {"amplitude": 2.5}

    def test_states_covariance_and_residual_rebuild_through_the_blocks(
        self, tmp_path, observed
    ):
        path, a, late = written_file(tmp_path, observed)
        # No observed data: states and covariance still work.
        with L2Data.open(path) as l2:
            states = l2.states(3, [a, late])
            np.testing.assert_array_equal(states["a"].template["A"], 2.5)
            assert states["late"].template is None
            np.testing.assert_array_equal(states["late"].covariance.A, 3.0)
            np.testing.assert_array_equal(l2.states(0, [a])["a"].template["E"], 1.0)
            assert l2.covariance(1, [a]) is None
            np.testing.assert_array_equal(l2.covariance(2, [late]).A, 2.0)
            with pytest.raises(
                ValueError,
                match="no observed data attached; pass observed_data to L2Data.open",
            ):
                l2.residual(3, [a])
            with pytest.raises(ValueError, match=re.escape("missing ['late']")):
                l2.covariance(3, [a])
            with pytest.raises(ValueError, match="unknown block"):
                l2.states(3, [Amplitude("ghost", 0.0)])
            with pytest.raises(ValueError, match="repeated"):
                l2.states(3, [a, a])
        with L2Data.open(path, observed_data=observed) as l2:
            residual = l2.residual(3, [a, late])
            for ch in observed.channel_names:
                np.testing.assert_allclose(
                    residual.channel_data[ch], observed.channel_data[ch] - 2.5
                )
            with pytest.raises(ValueError, match=re.escape("missing ['a']")):
                l2.residual(3, [late])

    def test_states_translate_a_frequency_block_back_to_the_observation_grid(
        self, tmp_path, observed
    ):
        spectral = observed.to_frequency()
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "f",
            role="signal",
            index=0,
            grid=spectral.grid,
            encoded=rec.encode(
                "f", "initial_sample", {"amplitude": 0.25}, registering=True
            ),
        )
        rec.close()
        block = Amplitude("f", 0.25)
        expected = transform(
            block.state_from_parameters({"amplitude": 0.25}, spectral),
            "time",
            reference_data=spectral,
        )
        with L2Data.open(tmp_path / "campaign.h5") as l2:
            assert l2.block_grid("f") == spectral.grid
            state = l2.states(0, [block])["f"]
            assert state.template["A"].shape == (64,)
            np.testing.assert_allclose(state.template["A"], expected.template["A"])

    def test_bad_rebuilds_are_reported(self, tmp_path, observed):
        path, a, _ = written_file(tmp_path, observed)

        class Wrong(Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                return {"not": "a state"}

        class OffGrid(Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                return BlockState(
                    template={ch: np.zeros(3) for ch in reference_data.channel_names}
                )

        with L2Data.open(path) as l2:
            with pytest.raises(
                TypeError, match="a.state_from_parameters must return a BlockState"
            ):
                l2.states(0, [Wrong("a", 1.0)])
            with pytest.raises(ValueError, match="length 3, expected 64"):
                l2.states(0, [OffGrid("a", 1.0)])


class Raises(Amplitude):
    def state_from_parameters(self, parameters, reference_data):
        raise RuntimeError("needs an orbit")


class NoiseWithTemplate(Level):
    def state_from_parameters(self, parameters, reference_data):
        state = super().state_from_parameters(parameters, reference_data)
        return replace(
            state, template={ch: 0 * v for ch, v in reference_data.channel_data.items()}
        )


class NoiseBare(Level):
    def state_from_parameters(self, parameters, reference_data):
        return BlockState(parameters={"variance": 1.0})


class SignalWithCovariance(Amplitude):
    def state_from_parameters(self, parameters, reference_data):
        state = super().state_from_parameters(parameters, reference_data)
        return replace(state, covariance=white_covariance(reference_data, 1.0))


def two_noise_file(tmp_path, observed):
    rec = recorder(tmp_path, observed, "two.h5")
    for index, (name, variance) in enumerate((("n1", 2.0), ("n2", 5.0))):
        rec.add_block(
            name,
            role="noise",
            index=index,
            grid=observed.grid,
            encoded=rec.encode(
                name, "initial_sample", {"variance": variance}, registering=True
            ),
        )
    rec.close()
    return tmp_path / "two.h5"


class TestL2DataReview:
    def test_two_noise_blocks_sum(self, tmp_path, observed):
        path = two_noise_file(tmp_path, observed)
        n1, n2 = Level("n1", 2.0), Level("n2", 5.0)
        with L2Data.open(path) as l2:
            total = l2.covariance(0, [n1, n2])
            states = l2.states(0, [n2, n1])
            assert list(states) == ["n2", "n1"]
            expected = states["n1"].covariance + states["n2"].covariance
            np.testing.assert_array_equal(total.blocks, expected.blocks)
            np.testing.assert_array_equal(total.A, 7.0)

    def test_format_version_is_refused(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with h5py.File(path, "r+") as file:
            file.attrs["format_version"] = 2
        with pytest.raises(
            ValueError, match="has format version 2; this enchilada reads version 1"
        ):
            L2Data.open(path)

    def test_role_rules_on_rebuilt_states(self, tmp_path, observed):
        path, a, late = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            with pytest.raises(ValueError, match="returned a template but 'late'"):
                l2.states(3, [NoiseWithTemplate("late", 2.0)])
            with pytest.raises(ValueError, match="returned a covariance but 'a'"):
                l2.states(3, [SignalWithCovariance("a", 1.0)])
            with pytest.raises(
                ValueError, match="must return a covariance because 'late'"
            ):
                l2.states(3, [NoiseBare("late", 1.0)])

    def test_a_raising_rebuild_gains_a_note(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            with pytest.raises(RuntimeError, match="needs an orbit") as info:
                l2.states(1, [Raises("a", 1.0)])
        assert any(
            "while L2Data rebuilt 'a' at step 1; attach observed_data" in note
            for note in info.value.__notes__
        )

    def test_names_are_refused(self, tmp_path, observed):
        path, a, _ = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            with pytest.raises(TypeError, match="Block objects, not names"):
                l2.states(0, ["a"])
            with pytest.raises(TypeError, match="Block objects, not names"):
                l2.covariance(0, ["a"])

    def test_a_late_block_fails_before_earlier_rebuilds(self, tmp_path, observed):
        path, a, late = written_file(tmp_path, observed)
        calls = []

        class Spy(Amplitude):
            def state_from_parameters(self, parameters, reference_data):
                calls.append(1)
                return super().state_from_parameters(parameters, reference_data)

        with L2Data.open(path) as l2:
            with pytest.raises(ValueError, match="registered at step 2"):
                l2.states(1, [Spy("a", 1.0), late])
        assert calls == []

    def test_registration_order_follows_index_not_name(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        for index, name in enumerate(("zeta", "alpha")):
            rec.add_block(
                name,
                role="signal",
                index=index,
                grid=observed.grid,
                encoded=rec.encode(
                    name, "initial_sample", {"amplitude": 1.0}, registering=True
                ),
            )
        rec.close()
        with L2Data.open(tmp_path / "campaign.h5") as l2:
            assert l2.block_names == ("zeta", "alpha")

    def test_chain_of_a_fixed_shape_array(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        rec.add_block(
            "m",
            role="signal",
            index=0,
            grid=observed.grid,
            encoded=rec.encode(
                "m",
                "initial_sample",
                {"w": np.arange(6.0).reshape(2, 3)},
                registering=True,
            ),
        )
        rec.append_step(
            {"m": rec.encode("m", "sample", {"w": np.ones((2, 3))}, registering=False)}
        )
        rec.close()
        with L2Data.open(tmp_path / "campaign.h5") as l2:
            assert l2.chain("m", "w").shape == (2, 2, 3)
            np.testing.assert_array_equal(
                l2.parameters("m", 0)["w"], np.arange(6.0).reshape(2, 3)
            )

    def test_states_keep_the_order_given(self, tmp_path, observed):
        path, a, late = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            assert list(l2.states(3, [late, a])) == ["late", "a"]

    def test_a_step_must_be_an_integer(self, tmp_path, observed):
        path, a, _ = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            with pytest.raises(TypeError, match="step must be an integer"):
                l2.parameters("a", 1.5)
            with pytest.raises(TypeError, match="step must be an integer"):
                l2.states(True, [a])

    def test_a_borrowed_handle_stays_open(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with h5py.File(path, "r") as file:
            l2 = L2Data(file, owns_file=False)
            l2.close()
            assert l2.num_steps == 3
            assert file
        with pytest.raises(ValueError, match="is not an enchilada L2 file"):
            with h5py.File(tmp_path / "other.h5", "w") as other:
                L2Data(other)

    def test_observed_data_metadata_must_match(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        shifted = replace(observed, start_time_gps=10.0)
        with pytest.raises(
            ValueError, match=r"start_time_gps 10\.0 does not match the file's 0\.0"
        ):
            L2Data.open(path, observed_data=shifted)

    def test_a_group_without_entries_is_not_listed(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with h5py.File(path, "r+") as file:
            group = file.create_group("blocks/ghost")
            group.attrs["index"] = 5
            group.attrs["num_entries"] = 0
            file.create_group("blocks/bare").attrs["role"] = "signal"
        with L2Data.open(path) as l2:
            assert l2.block_names == ("a", "late")
            with pytest.raises(ValueError, match="unknown block"):
                l2.first_step("bare")
            with pytest.raises(ValueError, match="unknown block"):
                l2.role("ghost")


class TestL2DataFollowUps:
    def test_residual_is_float64_for_a_float32_observation(self, tmp_path, observed):
        path, a, late = written_file(tmp_path, observed)
        single = replace(
            observed,
            channel_data={
                ch: v.astype(np.float32) for ch, v in observed.channel_data.items()
            },
        )

        class Wide(Amplitude):
            def _state(self, data, amplitude):
                return data.block_state(
                    {
                        ch: np.full(arr.shape, amplitude)
                        for ch, arr in data.channel_data.items()
                    },
                    parameters={"amplitude": float(amplitude)},
                )

        with L2Data.open(path, observed_data=single) as l2:
            result = l2.residual(3, [Wide("a", 1.0), late])
        assert all(v.dtype == np.float64 for v in result.channel_data.values())

    def test_an_unrecognised_role_is_refused_everywhere(self, tmp_path, observed):
        path, a, late = written_file(tmp_path, observed)
        with h5py.File(path, "r+") as file:
            file["blocks/late"].attrs["role"] = "mystery"
        with L2Data.open(path, observed_data=observed) as l2:
            with pytest.raises(ValueError, match="unknown role 'mystery'"):
                l2.covariance(3, [a, late])
            with pytest.raises(ValueError, match="unknown role 'mystery'"):
                l2.residual(3, [a, late])

    def test_add_block_refuses_an_unknown_role(self, tmp_path, observed):
        rec = recorder(tmp_path, observed)
        encoded = rec.encode(
            "x", "initial_sample", {"amplitude": 1.0}, registering=True
        )
        with pytest.raises(ValueError, match="block 'x' has role 'loud'"):
            rec.add_block(
                "x", role="loud", index=0, grid=observed.grid, encoded=encoded
            )
        rec.close()

    def test_the_orbit_hint_only_shows_without_observed_data(self, tmp_path, observed):
        path, _, _ = written_file(tmp_path, observed)
        with L2Data.open(path) as l2:
            with pytest.raises(RuntimeError) as bare:
                l2.states(1, [Raises("a", 1.0)])
        with L2Data.open(path, observed_data=observed) as l2:
            with pytest.raises(RuntimeError) as attached:
                l2.states(1, [Raises("a", 1.0)])
        assert "attach observed_data" in bare.value.__notes__[0]
        assert "attach observed_data" not in attached.value.__notes__[0]


def raw_entries(wheel):
    """Every block's raw num_entries, which the reader's num_steps bound hides."""
    blocks = wheel._recorder.file["blocks"]
    return {name: int(blocks[name].attrs["num_entries"]) for name in blocks}


def assert_rebuilds_wheel(wheel, blocks, *, atol=0):
    """The file's rebuilt states, covariance and residual agree with the Wheel's."""
    channels = wheel.observed_data.channel_names
    states = wheel.l2.states(-1, blocks)
    for block in blocks:
        entry = wheel.ledger[block.name]
        if entry.template is None:
            assert states[block.name].template is None
        else:
            for ch in channels:
                np.testing.assert_allclose(
                    states[block.name].template[ch], entry.template[ch], atol=atol
                )
    rebuilt = wheel.l2.covariance(-1, blocks)
    if wheel.covariance is None:
        assert rebuilt is None
    else:
        np.testing.assert_allclose(rebuilt.blocks, wheel.covariance.blocks, atol=atol)
    residual = wheel.l2.residual(-1, blocks)
    for ch in channels:
        np.testing.assert_allclose(
            residual.channel_data[ch],
            wheel.working_residual.channel_data[ch],
            atol=atol,
        )


class Draws(Amplitude):
    """Signal block that draws from its RNG, so a leaked candidate RNG shows."""

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        rng.standard_normal()
        return super().sample(
            conditional_residual, covariance, current_block_state, rng=rng
        )


class Relabels(Draws):
    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        state = super().sample(
            conditional_residual, covariance, current_block_state, rng=rng
        )
        state.parameters["label"] = 1.0
        return state


class Counts(Amplitude):
    """Signal block that counts its sample calls."""

    def __init__(self, name, start):
        super().__init__(name, start)
        self.calls = 0
        self.initial_calls = 0

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self.initial_calls += 1
        return super().initial_sample(conditional_residual, covariance, rng=rng)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.calls += 1
        return super().sample(
            conditional_residual, covariance, current_block_state, rng=rng
        )


class TestWheelRecords:
    def test_without_a_path_nothing_changes(self, observed):
        wheel = Wheel(observed, random_seed=0)
        assert wheel.l2 is None
        wheel.close()
        wheel.add(Amplitude("a", 1.0), role="signal")
        wheel.run(1)
        wheel.add(Amplitude("later", 1.0, step=0.1), role="signal")

    def test_the_file_is_created_with_the_observation_and_refused_when_present(
        self, tmp_path, observed
    ):
        path = tmp_path / "campaign.h5"
        wheel = Wheel(observed, random_seed=0, output_path=path)
        assert path.exists()
        assert isinstance(wheel.l2, L2Data)
        assert wheel.l2 is wheel.l2
        assert wheel.l2.num_steps == 0 and wheel.l2.block_names == ()
        assert wheel.l2.observation_grid == observed.grid
        with pytest.raises(FileExistsError):
            Wheel(observed, output_path=path)
        wheel.close()

    def test_a_failure_while_opening_the_view_removes_the_file(
        self, tmp_path, observed, monkeypatch
    ):
        path = tmp_path / "campaign.h5"
        original = L2Data.__init__

        def refuses(self, *args, **kwargs):
            raise RuntimeError("no view")

        monkeypatch.setattr(L2Data, "__init__", refuses)
        with pytest.raises(RuntimeError, match="no view"):
            Wheel(observed, output_path=path)
        assert not path.exists()
        monkeypatch.setattr(L2Data, "__init__", original)
        Wheel(observed, output_path=path).close()

    def test_registration_and_cycles_are_recorded(self, tmp_path, observed):
        with Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5") as wheel:
            wheel.add(Level("noise", 1.0), role="noise")
            wheel.add(Amplitude("a", 1.0), role="signal")
            assert wheel.l2.block_names == ("noise", "a")
            assert wheel.l2.num_steps == 0
            np.testing.assert_array_equal(wheel.l2.chain("a", "amplitude"), [1.0])
            wheel.run(3)
            assert wheel.l2.num_steps == 3
            np.testing.assert_array_equal(
                wheel.l2.chain("a", "amplitude"), [1.0, 1.5, 2.0, 2.5]
            )
            np.testing.assert_array_equal(
                wheel.l2.chain("noise", "variance"), [1.0, 2.0, 3.0, 4.0]
            )
            assert wheel.l2.parameters("a") == {"amplitude": 2.5}
            wheel.run(2)
            assert wheel.l2.num_steps == 5
            assert wheel.l2.chain("a", "amplitude").shape == (6,)

    def test_the_record_precedes_the_callback(self, tmp_path, observed):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        wheel.add(Amplitude("a", 1.0), role="signal")
        seen = []

        def callback(cycle_index, campaign):
            seen.append((cycle_index, campaign.l2.num_steps))
            raise RuntimeError("stop")

        with pytest.raises(RuntimeError, match="stop"):
            wheel.run(2, on_cycle_complete=callback)
        assert seen == [(0, 1)]
        assert wheel.l2.num_steps == 1

    def test_a_partial_cycle_records_nothing(self, tmp_path, observed):
        class Explodes(Amplitude):
            armed = False

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                if self.armed:
                    raise RuntimeError("boom")
                return super().sample(
                    conditional_residual, covariance, current_block_state, rng=rng
                )

        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        first, second = Amplitude("first", 1.0), Explodes("second", 1.0)
        wheel.add(first, role="signal")
        wheel.add(second, role="signal")
        wheel.run(1)
        second.armed = True
        with pytest.raises(RuntimeError, match="boom"):
            wheel.run(1)
        # committed in memory
        assert wheel.ledger["first"].parameters == {"amplitude": 2.0}
        assert wheel.l2.num_steps == 1
        assert raw_entries(wheel) == {"first": 2, "second": 2}
        np.testing.assert_array_equal(wheel.l2.chain("first", "amplitude"), [1.0, 1.5])
        second.armed = False
        wheel.run(1)
        np.testing.assert_array_equal(
            wheel.l2.chain("first", "amplitude"), [1.0, 1.5, 2.5]
        )
        np.testing.assert_array_equal(
            wheel.l2.chain("second", "amplitude"), [1.0, 1.5, 2.0]
        )

    def test_a_block_added_after_cycles_starts_at_the_current_step(
        self, tmp_path, observed
    ):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        early, late = Amplitude("early", 1.0), Amplitude("late", 10.0)
        wheel.add(early, role="signal")
        wheel.run(2)
        wheel.add(late, role="signal")
        assert wheel.l2.first_step("late") == 2
        np.testing.assert_array_equal(wheel.l2.chain("late", "amplitude"), [10.0])
        wheel.run(1)
        np.testing.assert_array_equal(wheel.l2.chain("late", "amplitude"), [10.0, 10.5])
        assert wheel.l2.chain("early", "amplitude").shape == (4,)
        with pytest.raises(ValueError, match="registered at step 2"):
            wheel.l2.states(1, [late])
        assert wheel.l2.parameters("late", 2) == {"amplitude": 10.0}

    def test_a_drawing_cycle_advances_the_rng(self, tmp_path, observed):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        wheel.add(Draws("a", 1.0), role="signal")
        before = wheel._rng.bit_generator.state
        wheel.run(1)
        assert wheel._rng.bit_generator.state != before

    def test_an_unrecordable_sample_changes_nothing(self, tmp_path, observed):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        wheel.add(Relabels("a", 1.0), role="signal")  # first in the cycle
        wheel.add(Level("noise", 1.0), role="noise")
        rng_before = wheel._rng.bit_generator.state
        residual_before = wheel.working_residual.channel_data["A"].copy()
        covariance_before = wheel.covariance.blocks.copy()
        entries_before = raw_entries(wheel)
        with pytest.raises(ValueError, match="the key set is fixed at registration"):
            wheel.run(1)
        assert wheel.ledger["a"].parameters == {"amplitude": 1.0}
        assert wheel.ledger["noise"].parameters == {"variance": 1.0}
        assert wheel._rng.bit_generator.state == rng_before
        np.testing.assert_array_equal(
            wheel.working_residual.channel_data["A"], residual_before
        )
        np.testing.assert_array_equal(wheel.covariance.blocks, covariance_before)
        assert raw_entries(wheel) == entries_before == {"a": 1, "noise": 1}
        assert wheel.l2.num_steps == 0
        np.testing.assert_array_equal(wheel.l2.chain("a", "amplitude"), [1.0])

    def test_an_unrecordable_start_is_refused_at_registration(self, tmp_path, observed):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        wheel.add(Amplitude("a", 1.0), role="signal")

        class Strings(Amplitude):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                state = super().initial_sample(
                    conditional_residual, covariance, rng=rng
                )
                state.parameters["label"] = "text"
                return state

        with pytest.raises(
            TypeError, match="recorded parameters must be bool, integer"
        ):
            wheel.add(Strings("s", 1.0), role="signal")
        assert "s" not in wheel.ledger
        assert wheel.l2.block_names == ("a",)
        assert raw_entries(wheel) == {"a": 1}
        wheel.add(Amplitude("s", 1.0), role="signal")
        assert wheel.l2.block_names == ("a", "s")

    def test_a_name_that_cannot_name_a_group_is_refused_before_initial_sample(
        self, tmp_path, observed
    ):
        calls = []

        class Loud(Amplitude):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                calls.append(True)
                return super().initial_sample(conditional_residual, covariance, rng=rng)

        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        with pytest.raises(
            ValueError, match=re.escape("block name 'a/b' contains '/'")
        ):
            wheel.add(Loud("a/b", 1.0), role="signal")
        with pytest.raises(ValueError, match="is not valid UTF-8"):
            wheel.add(Loud("bad\ud800", 1.0), role="signal")
        assert calls == [] and len(wheel.ledger) == 0
        Wheel(observed).add(Loud("a/b", 1.0), role="signal")  # fine without a file
        assert calls == [True]

    def test_a_key_that_is_not_utf8_is_refused(self, tmp_path, observed):
        class BadKey(Amplitude):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                state = super().initial_sample(
                    conditional_residual, covariance, rng=rng
                )
                state.parameters["bad\ud800"] = 1.0
                return state

        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        with pytest.raises(ValueError, match="is not valid UTF-8"):
            wheel.add(BadKey("k", 1.0), role="signal")
        assert len(wheel.ledger) == 0

    def test_rebuilt_states_match_the_wheel(self, tmp_path, observed):
        with Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5") as wheel:
            blocks = [Level("noise", 1.0), Amplitude("t", 1.0), Population()]
            wheel.add(blocks[0], role="noise")
            wheel.add(blocks[1], role="signal")
            wheel.add(blocks[2], role="signal")
            wheel.run(3)
            assert_rebuilds_wheel(wheel, blocks)

    def test_a_ragged_parameter_keeps_every_entry(self, tmp_path, observed):
        with Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5") as wheel:
            wheel.add(Population(), role="signal")
            wheel.run(3)
            assert [e.shape for e in wheel.l2.entries("pop", "frequencies")] == [
                (2,),
                (3,),
                (0,),
                (1,),
            ]

    def test_rebuilt_states_match_the_wheel_across_grids(self, tmp_path, observed):
        with Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5") as wheel:
            blocks = [Amplitude("f", 0.5), Amplitude("w", 0.25)]
            wheel.add(blocks[0], role="signal", data_domain="frequency")
            wheel.add(
                blocks[1], role="signal", data_domain="wdm", num_frequency_divisions=8
            )
            wheel.run(2)
            assert wheel.l2.block_grid("f") == observed.to_frequency().grid
            assert wheel.l2.block_grid("w").wdm_grid == WDMGrid(
                num_frequency_divisions=8, num_time_divisions=8
            )
            assert_rebuilds_wheel(wheel, blocks, atol=1e-12)

    @pytest.mark.parametrize(
        ("data_domain", "divisions"),
        [("frequency", {}), ("wdm", {"num_frequency_divisions": 8})],
    )
    def test_a_noise_block_off_the_observation_grid_rebuilds_the_covariance(
        self, tmp_path, observed, data_domain, divisions
    ):
        with Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5") as wheel:
            blocks = [Level("noise", 1.0), Amplitude("s", 0.5)]
            wheel.add(blocks[0], role="noise", data_domain=data_domain, **divisions)
            wheel.add(blocks[1], role="signal", data_domain=data_domain, **divisions)
            wheel.run(3)
            assert wheel.covariance.grid != observed.grid
            assert wheel.l2.covariance(-1, blocks).grid == wheel.covariance.grid
            assert_rebuilds_wheel(wheel, blocks, atol=1e-12)

    def test_a_failed_write_refuses_the_next_run_before_sampling(
        self, tmp_path, observed, monkeypatch
    ):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        block = Counts("a", 1.0)
        wheel.add(block, role="signal")
        starts = Counts("b", 1.0)

        def full_disk(group, encoded):
            raise OSError("disk full")

        monkeypatch.setattr(_Recorder, "_append_entry", staticmethod(full_disk))
        with pytest.raises(OSError, match="disk full"):
            wheel.run(1)
        monkeypatch.undo()
        assert wheel.ledger["a"].parameters == {"amplitude": 1.5}  # memory one ahead
        assert wheel.l2.num_steps == 0
        assert block.calls == 1
        with pytest.raises(RuntimeError, match="failed part-way"):
            wheel.run(1)
        with pytest.raises(RuntimeError, match="failed part-way"):
            wheel.add(starts, role="signal")
        assert starts.initial_calls == 0
        assert block.calls == 1

    def test_an_add_interrupted_before_the_write_refuses_the_next_run(
        self, tmp_path, observed, monkeypatch
    ):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        block = Counts("a", 1.0)
        wheel.add(block, role="signal")

        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(_Recorder, "add_block", interrupted)
        with pytest.raises(KeyboardInterrupt):
            wheel.add(Amplitude("b", 1.0), role="signal")
        monkeypatch.undo()
        with pytest.raises(RuntimeError, match="add was interrupted"):
            wheel.run(1)
        assert block.calls == 0

    def test_close_ends_recording_and_the_file_reopens(self, tmp_path, observed):
        path = tmp_path / "c.h5"
        wheel = Wheel(observed, random_seed=0, output_path=path)
        block = Amplitude("a", 1.0)
        wheel.add(block, role="signal")
        wheel.run(2)
        wheel.close()
        wheel.close()
        with pytest.raises(ValueError, match="the campaign output file is closed"):
            wheel.run(1)
        with pytest.raises(ValueError, match="the campaign output file is closed"):
            wheel.run(0)
        with pytest.raises(ValueError, match="the campaign output file is closed"):
            wheel.add(Amplitude("b", 1.0), role="signal")
        with pytest.raises(ValueError, match="L2Data is closed"):
            wheel.l2.chain("a", "amplitude")
        with L2Data.open(path, observed_data=observed) as l2:
            assert l2.num_steps == 2
            np.testing.assert_array_equal(l2.chain("a", "amplitude"), [1.0, 1.5, 2.0])
            np.testing.assert_allclose(
                l2.residual(-1, [block]).channel_data["A"],
                observed.channel_data["A"] - 2.0,
            )

    def test_close_inside_the_callback_stops_the_next_cycle_before_sampling(
        self, tmp_path, observed
    ):
        wheel = Wheel(observed, random_seed=0, output_path=tmp_path / "c.h5")
        block = Counts("a", 1.0)
        wheel.add(block, role="signal")
        with pytest.raises(
            ValueError,
            match="the campaign output file is closed; construct a new Wheel",
        ):
            wheel.run(3, on_cycle_complete=lambda index, campaign: campaign.close())
        assert block.calls == 1

    def test_the_context_manager_closes(self, tmp_path, observed):
        path = tmp_path / "c.h5"
        with Wheel(observed, random_seed=0, output_path=path) as wheel:
            wheel.add(Amplitude("a", 1.0), role="signal")
        with pytest.raises(
            ValueError,
            match="the campaign output file is closed; construct a new Wheel",
        ):
            wheel.run(1)

    def test_the_context_manager_closes_on_an_exception(self, tmp_path, observed):
        path = tmp_path / "c.h5"
        with pytest.raises(KeyError):
            with Wheel(observed, random_seed=0, output_path=path) as wheel:
                wheel.add(Amplitude("a", 1.0), role="signal")
                raise KeyError("inside")
        assert wheel._recorder.closed
        with pytest.raises(ValueError, match="the campaign output file is closed"):
            wheel.run(1)
        with L2Data.open(path, observed_data=observed) as l2:
            assert l2.block_names == ("a",)
