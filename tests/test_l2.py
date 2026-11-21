"""The campaign output file: the recorder, the reader and the Wheel that writes it."""

import re

import h5py
import numpy as np
import pytest

from enchilada import Grid, WDMGrid, transform
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
