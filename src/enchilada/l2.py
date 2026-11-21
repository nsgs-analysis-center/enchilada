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
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import h5py
import numpy as np

from enchilada.block import Block
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid, WDMGrid
from enchilada.translation import _data_on_grid, _state_on_grid

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
    if "/" in value:
        raise ValueError(
            f"{what} {value!r} contains '/'; it cannot name a {kind} in the output file"
        )
    if "\x00" in value:
        raise ValueError(
            f"{what} {value!r} contains a NUL character; it cannot name a {kind} in "
            "the output file"
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
        self._failed_at_step = 0
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
            steps = self._failed_at_step
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
        group = None
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
            self._latch(error)
            self._schemas.pop(name, None)
            if group is not None:
                with contextlib.suppress(Exception):
                    del self._file["blocks"][name]
            raise

    def _latch(self, error: BaseException) -> None:
        self._failed = error
        with contextlib.suppress(Exception):
            self._failed_at_step = int(self._file.attrs["num_steps"])

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
            self._latch(error)
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


class L2Data:
    """A read-only view of a campaign file: chains, parameters and rebuilt states.

    `wheel.l2` is this view over the Wheel's open file while a campaign runs;
    `L2Data.open(path)` reopens a finished file. Steps are global: step 0 is
    the state after the first registrations, step s the state after the s-th
    complete cycle across every `run` call; a block added later has
    `first_step(name) > 0` and no state before it. Negative steps count from
    the latest, so -1 is the current state.

    The file holds parameters, not templates. `states`, `covariance` and
    `residual` take the block instances and call their `state_from_parameters`
    on each block's registered grid, then translate templates to the
    observation grid the way the Wheel does. `residual` needs the observed
    data: attach it with `observed_data` at `open` (it also carries the orbit
    blocks may need); the live view has the Wheel's.
    """

    def __init__(
        self, file: Any, *, observed_data: L1Data | None = None, owns_file: bool = True
    ) -> None:
        self._file = file
        self._owns_file = owns_file
        self._observed: L1Data | None = None
        if observed_data is not None:
            if tuple(observed_data.channel_names) != self.channel_names:
                raise ValueError(
                    f"observed_data channel names {tuple(observed_data.channel_names)} "
                    f"do not match the file's {self.channel_names}"
                )
            if observed_data.grid != self.observation_grid:
                raise ValueError(
                    f"observed_data grid {observed_data.grid} does not match the "
                    f"file's observation grid {self.observation_grid}"
                )
            self._observed = observed_data

    @classmethod
    def open(
        cls, path: str | os.PathLike[str], *, observed_data: L1Data | None = None
    ) -> L2Data:
        """Open a campaign file read-only, optionally attaching the observation."""
        file = h5py.File(os.fspath(path), "r")
        try:
            if file.attrs.get("format") != FORMAT:
                raise ValueError(f"{os.fspath(path)} is not an enchilada L2 file")
            found = int(file.attrs["format_version"])
            if found != FORMAT_VERSION:
                raise ValueError(
                    f"{os.fspath(path)} has format version {found}; this enchilada "
                    f"reads version {FORMAT_VERSION}"
                )
            return cls(file, observed_data=observed_data, owns_file=True)
        except Exception:
            file.close()
            raise

    def close(self) -> None:
        """Close a file this view opened; a Wheel-owned file is left alone."""
        if self._owns_file and self._file:
            self._file.close()

    def __enter__(self) -> L2Data:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- metadata --------------------------------------------------------------

    def _require_open(self) -> Any:
        if not self._file:
            raise ValueError("L2Data is closed")
        return self._file

    @property
    def num_steps(self) -> int:
        """Committed complete cycles; the file holds steps 0 to num_steps."""
        return int(self._require_open().attrs["num_steps"])

    @property
    def block_names(self) -> tuple[str, ...]:
        """Recorded blocks in registration order."""
        blocks = self._require_open()["blocks"]
        return tuple(
            name
            for _, name in sorted(
                (int(group.attrs["index"]), name) for name, group in blocks.items()
            )
        )

    @property
    def observation_grid(self) -> Grid:
        return _grid_from_attrs(self._require_open()["observation"].attrs)

    @property
    def channel_names(self) -> tuple[str, ...]:
        names = self._require_open()["observation"].attrs["channel_names"]
        return tuple(str(name) for name in names)

    def _group(self, name: str) -> Any:
        blocks = self._require_open()["blocks"]
        if name not in blocks:
            raise ValueError(
                f"unknown block {name!r}; recorded: {list(self.block_names)}"
            )
        return blocks[name]

    def role(self, name: str) -> str:
        """`"signal"` or `"noise"`."""
        return str(self._group(name).attrs["role"])

    def first_step(self, name: str) -> int:
        """The step at which the block was registered; its entry 0 is that state."""
        return int(self._group(name).attrs["first_step"])

    def block_grid(self, name: str) -> Grid:
        """The grid the block was registered on and renders on."""
        return _block_grid_from_attrs(self._group(name).attrs, self.observation_grid)

    # ---- entries ---------------------------------------------------------------

    def _resolve_step(self, step: int) -> int:
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise TypeError(f"step must be an integer, got {step!r}")
        num_steps = self.num_steps
        resolved = int(step) + (num_steps + 1 if step < 0 else 0)
        if not 0 <= resolved <= num_steps:
            raise IndexError(
                f"step {step} is out of range; the file holds steps 0 to {num_steps}"
            )
        return resolved

    def _entry_index(self, name: str, resolved_step: int) -> int:
        first = self.first_step(name)
        if resolved_step < first:
            raise ValueError(
                f"block {name!r} was registered at step {first}; it has no state at "
                f"step {resolved_step}"
            )
        return resolved_step - first

    @staticmethod
    def _schema(group: Any) -> _Schema:
        return _Schema.read(group.attrs)

    def _key_layout(self, name: str, key: str) -> tuple[Any, np.ndarray, np.ndarray]:
        """The key's packed dataset, entry shapes (n, ndim) and offsets (n + 1,)."""
        group = self._group(name)
        schema = self._schema(group)
        if key not in schema.keys:
            raise ValueError(
                f"block {name!r} has no parameter {key!r}; "
                f"recorded: {list(schema.keys)}"
            )
        # num_steps bounds every block: an entry a failed step wrote ahead of the
        # committed step counter is not data.
        count = min(
            int(group.attrs["num_entries"]),
            self.num_steps - int(group.attrs["first_step"]) + 1,
        )
        position = schema.keys.index(key)
        ndim = schema.ndims[position]
        if ndim == 0:
            shapes = np.zeros((count, 0), dtype=np.int64)
        else:
            column = sum(schema.ndims[:position])
            shapes = np.asarray(
                group["shapes"][:count, column : column + ndim], dtype=np.int64
            )
        sizes = np.prod(shapes, axis=1, dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes)])
        return group["parameters"][key], shapes, offsets

    @staticmethod
    def _value(
        flat: np.ndarray, shapes: np.ndarray, offsets: np.ndarray, index: int
    ) -> Any:
        if shapes.shape[1] == 0:
            return flat[offsets[index]]
        return flat[offsets[index] : offsets[index + 1]].reshape(tuple(shapes[index]))

    def entries(self, name: str, key: str) -> list[Any]:
        """Every recorded value of one parameter, a list in step order."""
        dataset, shapes, offsets = self._key_layout(name, key)
        flat = dataset[: offsets[-1]]
        return [self._value(flat, shapes, offsets, i) for i in range(shapes.shape[0])]

    def chain(self, name: str, key: str) -> np.ndarray:
        """One dense array `(num_entries, *shape)` of a parameter with a fixed shape."""
        dataset, shapes, offsets = self._key_layout(name, key)
        count = shapes.shape[0]
        if count and not (shapes == shapes[0]).all():
            raise ValueError(
                f"{name!r}.{key!r} has entries of different shapes; use entries() "
                "for a list"
            )
        flat = dataset[: offsets[-1]]
        return flat.reshape((count, *(tuple(shapes[0]) if count else ())))

    def parameters(self, name: str, step: int = -1) -> dict[str, Any]:
        """A block's parameter set at `step`: numpy scalars and arrays."""
        index = self._entry_index(name, self._resolve_step(step))
        return self._parameters_at(name, index)

    def _parameters_at(self, name: str, index: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in self._schema(self._group(name)).keys:
            dataset, shapes, offsets = self._key_layout(name, key)
            flat = dataset[offsets[index] : offsets[index + 1]]
            result[key] = (
                flat[0] if shapes.shape[1] == 0 else flat.reshape(tuple(shapes[index]))
            )
        return result

    # ---- rebuilding --------------------------------------------------------------

    def _zero_observation(self) -> L1Data:
        attrs = self._require_open()["observation"].attrs
        grid = self.observation_grid
        dtype = np.complex128 if grid.data_domain == "frequency" else np.float64
        return L1Data(
            {name: np.zeros(grid.channel_shape, dtype) for name in self.channel_names},
            sample_rate_hz=grid.sample_rate_hz,
            tdi_generation=str(attrs["tdi_generation"]),
            physical_observable=str(attrs["physical_observable"]),
            channel_names=self.channel_names,
            num_time_samples=grid.num_time_samples,
            start_time_gps=float(attrs["start_time_gps"]),
            data_domain=grid.data_domain,
        )

    def _reference(self, name: str) -> L1Data:
        observed = (
            self._observed if self._observed is not None else self._zero_observation()
        )
        return _data_on_grid(observed, self.block_grid(name))

    def _named(self, blocks: Iterable[Block]) -> dict[str, Block]:
        named: dict[str, Block] = {}
        for block in blocks:
            if block.name in named:
                raise ValueError(f"blocks have repeated names: {block.name!r}")
            named[block.name] = block
        recorded = self.block_names
        unknown = [name for name in named if name not in recorded]
        if unknown:
            raise ValueError(f"unknown blocks {unknown}; recorded: {list(recorded)}")
        return named

    def states(self, step: int, blocks: Iterable[Block]) -> dict[str, BlockState]:
        """Each given block's state at `step`, rebuilt on the observation grid."""
        resolved = self._resolve_step(step)
        named = self._named(blocks)
        result: dict[str, BlockState] = {}
        for name, block in named.items():
            index = self._entry_index(name, resolved)
            reference = self._reference(name)
            state = block.state_from_parameters(
                self._parameters_at(name, index), reference
            )
            if not isinstance(state, BlockState):
                raise TypeError(
                    f"{name}.state_from_parameters must return a BlockState, got "
                    f"{type(state).__name__}"
                )
            if state.template is not None:
                check_on_grid(
                    state.template,
                    channel_names=reference.channel_names,
                    num_time_samples=reference.num_time_samples,
                    data_domain=reference.data_domain,
                    wdm_grid=reference.wdm_grid,
                    context_label=f"{name}.state_from_parameters template",
                )
            result[name] = _state_on_grid(
                state, self.observation_grid, reference_data=reference
            )
        return result

    def _recorded_with_role(self, role: str, resolved_step: int) -> list[str]:
        return [
            name
            for name in self.block_names
            if self.role(name) == role and self.first_step(name) <= resolved_step
        ]

    def covariance(self, step: int, blocks: Iterable[Block]) -> Covariance | None:
        """The sum of the noise blocks' covariances at `step`; None without any."""
        resolved = self._resolve_step(step)
        named = self._named(blocks)
        needed = self._recorded_with_role("noise", resolved)
        missing = [name for name in needed if name not in named]
        if missing:
            raise ValueError(
                f"covariance at step {resolved} needs every noise block recorded "
                f"there; missing {missing}"
            )
        total: Covariance | None = None
        for name, state in self.states(
            resolved, [named[name] for name in needed]
        ).items():
            assert state.covariance is not None, name
            total = state.covariance if total is None else total + state.covariance
        return total

    def residual(self, step: int, blocks: Iterable[Block]) -> L1Data:
        """The observation minus every signal block's rebuilt template at `step`."""
        if self._observed is None:
            raise ValueError(
                "no observed data attached; pass observed_data to L2Data.open"
            )
        resolved = self._resolve_step(step)
        named = self._named(blocks)
        needed = self._recorded_with_role("signal", resolved)
        missing = [name for name in needed if name not in named]
        if missing:
            raise ValueError(
                f"residual at step {resolved} needs every signal block recorded "
                f"there; missing {missing}"
            )
        channel_data = {
            name: array.copy() for name, array in self._observed.channel_data.items()
        }
        for state in self.states(resolved, [named[name] for name in needed]).values():
            if state.template is None:
                continue
            for name in channel_data:
                channel_data[name] = channel_data[name] - state.template[name]
        return replace(self._observed, channel_data=channel_data)
