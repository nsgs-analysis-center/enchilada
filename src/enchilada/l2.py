"""The campaign output file: every block's parameters at every step, and its reader.

The Wheel writes the file through `_Recorder`; `L2Data` reads it back. Layout,
format version 1:

    /                              attrs: format, format_version, enchilada_version,
                                   num_steps (committed complete cycles)
    /observation                   attrs: the observation grid and conventions
    /blocks/<name>                 attrs: role, index, first_step, data_domain,
                                   [num_frequency_divisions, num_time_divisions],
                                   keys, kinds, ndims, num_entries
    /blocks/<name>/shapes          int64 (num_entries, sum(ndims)), when sum(ndims) > 0
    /blocks/<name>/parameters/<k>  one packed, resizable dataset per parameter

Entry i of a block is its state at step first_step + i; entry 0 is its start.
Counters (`num_entries`, `num_steps`) are written after the data they count and
the file is flushed after every record, so a reader trusts the counters and a
crash loses at most the record in progress. `num_steps` bounds every block: a
reader takes at most `num_steps - first_step + 1` entries, whatever `num_entries`
says.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import h5py
import numpy as np

from enchilada.data import L1Data
from enchilada.domains import Grid, WDMGrid

FORMAT = "enchilada-l2"
FORMAT_VERSION = 1
STORED_DTYPES: dict[str, np.dtype[Any]] = {
    "b": np.dtype(bool),
    "i": np.dtype(np.int64),
    "u": np.dtype(np.uint64),
    "f": np.dtype(np.float64),
    "c": np.dtype(np.complex128),
}
_CHUNK = 1024


def _enchilada_version() -> str:
    try:
        return version("enchilada")
    except PackageNotFoundError:  # pragma: no cover - source tree without install
        return "0+unknown"


def _check_hdf5_name(value: str, what: str, kind: str) -> None:
    """Refuse a name HDF5 cannot take.

    `kind` is "group" for a block name or "dataset" for a parameter key.
    """
    if "/" in value or "\x00" in value:
        raise ValueError(
            f"{what} {value!r} contains '/' or a NUL character; it cannot name a "
            f"{kind} in the output file"
        )
    if value in ("", ".", ".."):
        raise ValueError(f"{what} {value!r} cannot name a {kind} in the output file")


@dataclass(frozen=True)
class _Schema:
    """A block's parameter layout, fixed at registration."""

    keys: tuple[str, ...]
    kinds: tuple[str, ...]
    ndims: tuple[int, ...]

    def write(self, attrs: MutableMapping[str, Any]) -> None:
        attrs["keys"] = np.array(list(self.keys), dtype=h5py.string_dtype())
        attrs["kinds"] = np.array(list(self.kinds), dtype=h5py.string_dtype())
        attrs["ndims"] = np.array(list(self.ndims), dtype=np.int64)

    @classmethod
    def read(cls, attrs: Mapping[str, Any]) -> _Schema:
        return cls(
            tuple(str(key) for key in attrs["keys"]),
            tuple(str(kind) for kind in attrs["kinds"]),
            tuple(int(ndim) for ndim in attrs["ndims"]),
        )


@dataclass(frozen=True)
class _Encoded:
    """One block's parameters converted to their stored dtypes, in schema order."""

    schema: _Schema
    values: tuple[np.ndarray, ...]


def _write_observation(group: h5py.Group, observed: L1Data) -> None:
    group.attrs["data_domain"] = observed.data_domain
    group.attrs["num_time_samples"] = int(observed.num_time_samples)
    group.attrs["sample_rate_hz"] = float(observed.sample_rate_hz)
    group.attrs["start_time_gps"] = float(observed.start_time_gps)
    group.attrs["tdi_generation"] = observed.tdi_generation
    group.attrs["physical_observable"] = observed.physical_observable
    group.attrs["channel_names"] = np.array(
        list(observed.channel_names), dtype=h5py.string_dtype()
    )
    _write_wdm_grid(group.attrs, observed.wdm_grid)


def _write_wdm_grid(attrs: MutableMapping[str, Any], wdm_grid: WDMGrid | None) -> None:
    if wdm_grid is not None:
        attrs["num_frequency_divisions"] = wdm_grid.num_frequency_divisions
        attrs["num_time_divisions"] = wdm_grid.num_time_divisions


def _wdm_grid_from_attrs(attrs: Mapping[str, Any]) -> WDMGrid | None:
    if attrs["data_domain"] != "wdm":
        return None
    return WDMGrid(
        num_frequency_divisions=int(attrs["num_frequency_divisions"]),
        num_time_divisions=int(attrs["num_time_divisions"]),
    )


def _grid_from_attrs(attrs: Mapping[str, Any]) -> Grid:
    return Grid(
        str(attrs["data_domain"]),
        int(attrs["num_time_samples"]),
        float(attrs["sample_rate_hz"]),
        _wdm_grid_from_attrs(attrs),
    )


def _block_grid_from_attrs(attrs: Mapping[str, Any], observation_grid: Grid) -> Grid:
    """A block group's grid: its domain and divisions on the observation's samples."""
    return Grid(
        str(attrs["data_domain"]),
        observation_grid.num_time_samples,
        observation_grid.sample_rate_hz,
        _wdm_grid_from_attrs(attrs),
    )


def _recordable_array(label: str, key: str, value: Any) -> np.ndarray:
    """`value` as an array of a recordable kind, or TypeError naming block and key."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{label} parameters[{key!r}] is not a regular numeric array ({error}); "
            "recorded parameters must be bool, integer, float or complex scalars or "
            "arrays"
        ) from error
    if array.dtype.kind not in STORED_DTYPES:
        raise TypeError(
            f"{label} parameters[{key!r}] has dtype {array.dtype}; recorded "
            "parameters must be bool, integer, float or complex scalars or arrays"
        )
    return array


class _Recorder:
    """Owns the open campaign file and appends to it; the Wheel's writer.

    `close` closes the file whether or not `create` opened it.
    """

    def __init__(self, file: h5py.File) -> None:
        self._file = file
        self._failed: BaseException | None = None
        self._schemas: dict[str, _Schema] = {}

    @classmethod
    def create(cls, path: str | os.PathLike[str], observed: L1Data) -> _Recorder:
        """Create the file (refusing an existing one) with the observation metadata."""
        name = os.fspath(path)
        # Checked explicitly: HDF5 reports "file exists" for a path this process
        # still holds open as a plain OSError, not FileExistsError.
        if os.path.exists(name):
            raise FileExistsError(
                f"{name!r} exists; the Wheel does not overwrite or resume a campaign "
                "file, so name a new path"
            )
        file = h5py.File(name, "x")
        try:
            return cls.initialize(file, observed)
        except BaseException:
            # Mode "x" proves this call made the file, so it is ours to remove.
            file.close()
            with contextlib.suppress(OSError):
                os.remove(name)
            raise

    @classmethod
    def initialize(cls, file: h5py.File, observed: L1Data) -> _Recorder:
        """Write the format and observation metadata into an open, writable `file`.

        For a writer that opens the file itself to put other groups beside
        `blocks` (a simulation's truth and data, say); `create` uses it too.
        """
        file.attrs["format"] = FORMAT
        file.attrs["format_version"] = FORMAT_VERSION
        file.attrs["enchilada_version"] = _enchilada_version()
        file.attrs["num_steps"] = 0
        _write_observation(file.create_group("observation"), observed)
        file.create_group("blocks")
        file.flush()
        return cls(file)

    @property
    def file(self) -> h5py.File:
        return self._file

    @property
    def closed(self) -> bool:
        return not bool(self._file)

    def close(self) -> None:
        if self._file:
            self._file.close()

    def _require_writable(self) -> None:
        if self._failed is not None:
            steps = int(self._file.attrs["num_steps"])
            raise RuntimeError(
                "an earlier write to the campaign file failed part-way; the file is "
                f"consistent up to step {steps} but cannot be appended to; construct "
                "a new Wheel to record another campaign"
            ) from self._failed

    def encode(
        self,
        block_name: str,
        method_name: str,
        parameters: dict[str, Any],
        *,
        registering: bool,
    ) -> _Encoded:
        """Validate `parameters` against the recordable rules and convert them.

        A registering block fixes its schema from these parameters; a
        registered block must match its schema's keys, kinds and dimension
        counts. Raises TypeError or ValueError and stores nothing.

        `longdouble` and `clongdouble` values are narrowed to float64 and
        complex128. The returned values may alias the caller's arrays, which
        must not be mutated before the write.
        """
        self._require_writable()
        label = f"{block_name}.{method_name}"
        arrays: dict[str, np.ndarray] = {}
        for key, value in parameters.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{label} parameters have a non-string key {key!r}; recorded "
                    "parameter names must be strings"
                )
            _check_hdf5_name(key, f"{label} parameter name", "dataset")
            array = _recordable_array(label, key, value)
            arrays[key] = array
        if registering:
            schema = _Schema(
                tuple(arrays),
                tuple(array.dtype.kind for array in arrays.values()),
                tuple(array.ndim for array in arrays.values()),
            )
        else:
            if block_name not in self._schemas:
                raise RuntimeError(
                    f"{label}: block {block_name!r} is not registered with the recorder"
                )
            schema = self._schemas[block_name]
            if set(arrays) != set(schema.keys):
                raise ValueError(
                    f"{label} parameters have keys {sorted(arrays)} but "
                    f"{block_name!r} registered {sorted(schema.keys)}; the key set is "
                    "fixed at registration"
                )
            for key, kind, ndim in zip(
                schema.keys, schema.kinds, schema.ndims, strict=True
            ):
                array = arrays[key]
                if array.dtype.kind != kind:
                    raise ValueError(
                        f"{label} parameters[{key!r}] has kind {array.dtype.kind!r} "
                        f"but was registered with kind {kind!r}; publish the same kind "
                        "every call"
                    )
                if array.ndim != ndim:
                    raise ValueError(
                        f"{label} parameters[{key!r}] has {array.ndim} dimensions but "
                        f"was registered with {ndim}; shapes may change between calls, "
                        "the number of dimensions may not"
                    )
        values = tuple(
            np.asarray(arrays[key], dtype=STORED_DTYPES[kind])
            for key, kind in zip(schema.keys, schema.kinds, strict=True)
        )
        return _Encoded(schema, values)

    def add_block(
        self, name: str, *, role: str, index: int, grid: Grid, encoded: _Encoded
    ) -> None:
        """Create the block's group and datasets and write its start as entry 0."""
        self._require_writable()
        try:
            group = self._file["blocks"].create_group(name)
            group.attrs["role"] = role
            group.attrs["index"] = int(index)
            group.attrs["first_step"] = int(self._file.attrs["num_steps"])
            group.attrs["data_domain"] = grid.data_domain
            _write_wdm_grid(group.attrs, grid.wdm_grid)
            encoded.schema.write(group.attrs)
            group.attrs["num_entries"] = 0
            parameters = group.create_group("parameters")
            for key, kind in zip(
                encoded.schema.keys, encoded.schema.kinds, strict=True
            ):
                parameters.create_dataset(
                    key,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=STORED_DTYPES[kind],
                    chunks=(_CHUNK,),
                )
            width = sum(encoded.schema.ndims)
            if width:
                group.create_dataset(
                    "shapes",
                    shape=(0, width),
                    maxshape=(None, width),
                    dtype=np.int64,
                    chunks=(_CHUNK, width),
                )
            self._schemas[name] = encoded.schema
            self._append_entry(group, encoded)
            self._file.flush()
        except BaseException as error:
            self._failed = error
            raise

    def append_step(self, pending: dict[str, _Encoded]) -> None:
        """Append one entry per registered block and commit the step."""
        self._require_writable()
        if set(pending) != set(self._schemas):
            raise RuntimeError(
                "a step needs an entry for every registered block; got "
                f"{sorted(pending)} for {sorted(self._schemas)}"
            )
        for name, encoded in pending.items():
            if encoded.schema is not self._schemas[name]:
                raise RuntimeError(
                    f"{name}: the entry was encoded for another block's schema"
                )
        try:
            blocks = self._file["blocks"]
            for name, encoded in pending.items():
                self._append_entry(blocks[name], encoded)
            self._file.attrs["num_steps"] = int(self._file.attrs["num_steps"]) + 1
            self._file.flush()
        except BaseException as error:
            self._failed = error
            raise

    @staticmethod
    def _append_entry(group: h5py.Group, encoded: _Encoded) -> None:
        count = int(group.attrs["num_entries"])
        parameters = group["parameters"]
        for key, array in zip(encoded.schema.keys, encoded.values, strict=True):
            dataset = parameters[key]
            start = dataset.shape[0]
            flat = array.reshape(-1)
            dataset.resize((start + flat.size,))
            if flat.size:
                dataset[start:] = flat
        if "shapes" in group:
            shapes = group["shapes"]
            shapes.resize((count + 1, shapes.shape[1]))
            shapes[count] = np.concatenate(
                [np.asarray(array.shape, dtype=np.int64) for array in encoded.values]
            )
        group.attrs["num_entries"] = count + 1
