# Campaign output: `state_from_parameters` and the L2 file

**Date:** 2026-11-20. **Branch:** `l2data`, cut from `main` at c669da8 (after
PR #12), in the worktree `.claude/worktrees/l2data`. **Status:** design approved
in conversation, implementation not started.

## 1. Problem

`Ledger` keeps each block's latest accepted state and nothing else. Posterior
history exists only through `on_cycle_complete`, where every user writes the
same collection loop by hand, and nothing reaches disk unless they write that
too. The README lists "No checkpoint file format or lifecycle end" as a known
limitation.

A block's template is a deterministic function of its parameters and the grid,
so the history worth keeping is the parameter set of every block at every
step. Templates are large and derivable; parameters are small and primary. This
design gives the Wheel an output file that it writes itself, a protocol method
that rebuilds any block's state from recorded parameters, and a reader that
turns the file back into states, covariances and residuals.

## 2. Decisions made with the user (2026-11-20)

| Question | Decision |
|---|---|
| What the file holds | The parameter set of every block at every step, plus the metadata needed to rebuild: observation grid and channel names, each block's role, grid, registration index and first step. No templates, no observation arrays. |
| How a residual comes back | Each block rebuilds its state from a parameter set through a new required protocol method, `state_from_parameters(parameters, reference_data)`. "Render" was rejected as a name. |
| Parameter shapes | Ragged: a key's shape may differ at every step, so reversible-jump blocks record a changing number of sources. |
| When data reaches disk | Write-through: every record is appended to an open HDF5 file and flushed. Memory stays flat. |
| Who writes | The Wheel. `Wheel(..., output_path=...)` records registration and every complete cycle with no user call. Reading is through `wheel.l2` while the campaign runs and `L2Data.open(path)` afterwards. |
| Step numbering | A global step counter across `run` calls, since `on_cycle_complete`'s index restarts per call. |
| Existing file at the path | Refused. There is no restart format. |
| h5py | Becomes a core dependency; the Wheel writes HDF5. |

## 3. The protocol: `state_from_parameters`

### 3.1 Signature and contract

```python
def state_from_parameters(
    self, parameters: dict[str, Any], reference_data: L1Data
) -> BlockState: ...
```

Rebuild this block's complete physical state from one parameter set: the
template for a signal block, the covariance for a noise block, with
`parameters` echoed on the result.

- **Purity.** The result depends only on `parameters`, the block's own
  configuration (constructor arguments) and the grid metadata of
  `reference_data`: domain, sample count, rate, start time, channel names,
  WDM grid, orbit. It never reads `reference_data`'s channel values and never
  draws from a random stream.
- **Grid.** `reference_data` is on the block's registered grid, the grid
  `sample` receives. The returned template is on that grid; `Wheel` and
  `L2Data` translate it to the observation grid.
- **Value types.** Parameters read back from a file arrive as numpy scalars
  and arrays, not the Python `float`s a block may have published. The method
  accepts both.
- **Determinism to tolerance.** Rebuilding from the published parameters must
  reproduce the published template and covariance to a relative tolerance of
  1e-9 (section 3.3). Publishing exactly what was rendered is the simplest way
  to meet this: `initial_sample` and `sample` end by calling
  `self.state_from_parameters({...}, conditional_residual)`, so the private
  `_render` helpers in the examples become this public method.

`Block`'s docstring describes the method beside the other two. `Wheel.add`
refuses a block without a callable `state_from_parameters`, before
`initial_sample` is called, with

```
TypeError: block 'name' does not implement state_from_parameters; a Block needs
name, initial_sample(conditional_residual, covariance, *, rng),
sample(conditional_residual, covariance, current_block_state, *, rng) and
state_from_parameters(parameters, reference_data).
```

and the two existing messages for `sample` and `initial_sample` list all four
members the same way. The check runs in every campaign, recording or not, so
every block in a campaign can be rebuilt from its file later.

### 3.2 Core blocks

- `FixedNoiseBlock.state_from_parameters` returns the configured covariance
  (its parameters are `{}`), like its other two methods.
- `EchoBlock.state_from_parameters` returns `BlockState()`.

### 3.3 `check_block` round trip

After registration and after every cycle, `check_block` rebuilds the block
under test from its own ledger entry:

```python
entry = wheel.ledger[name]
rebuilt = block_under_test.state_from_parameters(entry.parameters, wheel.observed_data)
```

The observation grid is the block's grid in `check_block`, so no translation is
needed. `rebuilt` must be a `BlockState` (`TypeError` naming the block). Its
template must be None exactly when the entry's is, have the same channel keys,
and satisfy per channel
`np.allclose(rebuilt, published, rtol=1e-9, atol=1e-9 * max(abs(published)))`
(`atol` is 0 for an all-zero channel). Its covariance must be None exactly
when the entry's is, have the same `channel_names` and `grid`, and satisfy the
same closeness on the covariance arrays. A failure raises

```
ValueError: name.state_from_parameters does not reproduce the published
template in channel 'A' at step 2: parameters must determine the template
```

(or "... the published covariance at step 2 ..."). Step 0 is registration;
step k is after cycle k. The rebuilt state's own `parameters` are not
compared. The per-cycle check runs through `on_cycle_complete` on the scratch
Wheel; `check_block`'s signature is unchanged.

### 3.4 Migration

Every block gains the method: the two core blocks; `SineBlock`,
`WhiteNoiseBlock`, `ExcessNoiseBlock`, `GBBlock`, `DomainEcho`, `ForgeBlock`,
the benchmark's `FixedSignal`; the block classes defined in
`examples/create_block.ipynb` and `examples/domain_translation.ipynb`; and
about sixty block classes in `tests/`, most of which return `BlockState()` or
reuse an existing render helper. The ten `isinstance(..., Block)` checks in
tests and notebooks hold once the method exists, because `Block` is a
`runtime_checkable` protocol. This migration is its own stage and the suite is
green before any file code lands.

## 4. `Wheel`

### 4.1 Construction and lifecycle

```python
Wheel(observed_data, *, random_seed=None, output_path=None)
```

`output_path` is `str | os.PathLike[str] | None`. With a path the Wheel creates
the HDF5 file during construction, with mode `"x"`: an existing file raises
`FileExistsError` and a missing parent directory raises the `OSError` h5py
gives. The root attributes and the observation metadata (section 5) are written
and flushed before `__init__` returns. With None, nothing in the Wheel's
behaviour changes from today.

| Member | Meaning |
|---|---|
| `wheel.l2` | The live `L2Data` view over the open file (section 6), with the Wheel's pristine observation attached so `residual` works; None without a path. The same object on every access. |
| `wheel.close()` | Closes the file. Idempotent. Afterwards `add` and `run` raise `ValueError("the campaign output file is closed; construct a new Wheel to record another campaign")`, and reads through `wheel.l2` raise `ValueError("L2Data is closed")`. Without a path, a no-op. |
| `with Wheel(...) as wheel:` | `__enter__` returns the Wheel; `__exit__` calls `close()`. |

### 4.2 What is recorded, and when

| Event | Record |
|---|---|
| `add` succeeds | A block group with the block's role, registration index, grid and first step, and entry 0: the parameters of the adopted start. |
| A cycle in `run` completes | One entry per registered block, the parameters of its current ledger state; then the committed step count advances by one. Written before `on_cycle_complete` is called, so a callback that raises does not lose the record. |
| A cycle fails part-way | Nothing. Earlier blocks in that cycle stay committed in memory, as today; the next complete cycle records every block's then-current parameters. |

Steps are global. Step 0 is the state after the first registrations; step s is
the state after the s-th complete cycle, counted across every `run` call. A
block registered after c complete cycles has `first_step = c`: its entry 0 is
its start and its entry i is its state at step c + i. The invariant at every
committed step s is that each block with `first_step <= s` has exactly
`s - first_step + 1` entries. "The state at step s" therefore means the ledger
as it stood when the step counter read s, after any registrations made at that
count.

### 4.3 Failure rules

The Wheel's guarantee stands: a refused state leaves everything unchanged.

- Parameters are *encoded* in `_adopt`, after the existing `BlockState`, grid,
  finiteness and role checks and before the commit block. Encoding validates
  the rules of section 4.4 and converts each value to its stored dtype. A
  value that cannot be recorded is refused with `TypeError` or `ValueError`
  exactly like an off-grid template: ledger, sums, residual, RNG and file
  untouched.
- The disk write happens after the commit block. A registration writes
  immediately. A sampling adoption parks its encoding in a per-cycle pending
  map that `run` appends to the file once the cycle completes; the map is
  cleared at the start of every cycle, so a failed cycle's encodings are
  dropped.
- An `OSError` from h5py propagates from `add` or `run`. The campaign in
  memory is intact and one record ahead of the file; the file is consistent
  up to its last committed counters (section 5.2). Continuing a campaign after
  such an error is unsupported.

### 4.4 Recordable parameters

These rules apply only when `output_path` is set. Without a file, parameters
stay as unconstrained as today.

| Rule | On violation |
|---|---|
| Keys are non-empty strings, not `"."` or `".."`, without `"/"` (they name HDF5 datasets). | `TypeError` for a non-string key; `ValueError` naming the key otherwise. |
| Values are numeric: `np.asarray(value).dtype.kind` in `b i u f c`. Python `bool`, `int`, `float`, `complex`, numpy scalars and arrays qualify; strings, None, nested dicts, lists of dicts and object arrays do not. Non-finite values are allowed. | `TypeError: name.sample parameters['label'] has dtype <U5; recorded parameters must be bool, integer, float or complex scalars or arrays` |
| The key set is fixed at registration. | `ValueError: name.sample parameters have keys [...] but 'name' registered [...]; the key set is fixed at registration` |
| Each key's dtype kind is fixed at registration. | `ValueError: name.sample parameters['amplitude'] has kind 'i' but was registered with kind 'f'; publish the same kind every call` |
| Each key's number of dimensions is fixed at registration. Shapes are free, including zero length. | `ValueError: name.sample parameters['frequency'] has 2 dimensions but was registered with 1; shapes may change between calls, the number of dimensions may not` |
| The block name is a non-empty string, not `"."` or `".."`, without `"/"` (it names an HDF5 group). Checked in `add` before `initial_sample`. | `ValueError: block name 'a/b' contains '/'; it cannot name a group in the output file` |

Stored dtypes by kind: `b` → bool, `i` → int64, `u` → uint64, `f` → float64,
`c` → complex128. Narrower values are widened; the kind never changes.

### 4.5 Inside `Wheel`

New module `src/enchilada/l2.py` holds `L2Data` (section 6) and a private
`_Recorder` that owns the h5py file: `create(path, observed_data)`,
`encode(block_name, method_name, parameters, *, registering)`,
`add_block(...)`, `append_step(pending)`, `close()`. `wheel.py` imports the
recorder and never imports h5py itself. The Wheel gains `_recorder`
(`_Recorder | None`), `_pending` (`dict[str, _Encoded]`) and `_l2`.

`_adopt` calls `encode` when recording and, after the commit block, either
`add_block` (registration) or stores the encoding in `_pending`. `run` clears
`_pending` at the top of each cycle and calls `append_step` after the inner
loop, before `on_cycle_complete`. The translation calls in `_block_inputs` and
`_canonical_state` are factored into two small helpers that `L2Data` reuses,
so a rebuilt residual equals the Wheel's to round-off by construction.

## 5. The file

### 5.1 Layout, format version 1

```
/                              attrs: format="enchilada-l2", format_version=1,
                               enchilada_version, num_steps
/observation                   attrs: data_domain, num_time_samples, sample_rate_hz,
                               start_time_gps, tdi_generation, physical_observable,
                               channel_names; for wdm also num_frequency_divisions,
                               num_time_divisions
/blocks/<name>                 attrs: role, index, first_step, data_domain,
                               [num_frequency_divisions, num_time_divisions],
                               keys, kinds, ndims, num_entries
/blocks/<name>/shapes          int64, shape (num_entries, sum(ndims)); present only
                               when sum(ndims) > 0
/blocks/<name>/parameters/<k>  1-D packed values for key k, dtype by kind,
                               resizable, chunked
```

`keys` is the parameter order; `kinds` and `ndims` are parallel to it. Row i of
`shapes` is the concatenation of every key's shape at entry i in key order; a
scalar key contributes no columns. A reader derives each key's entry sizes
(1 for a scalar, the product of its shape otherwise), takes the cumulative sum
for offsets, and slices the packed values. The observation group holds grid
metadata only; the observed arrays live in the L1 file they came from.

A file with one entry per block is also a complete truth record for a
simulation: `L2Data.states(0, blocks)` would return the injected states. That
is the first step toward the injection-recovery work and is out of scope here.

### 5.2 Consistency and concurrency

Within one record the packed values are appended first, then the `shapes` row,
then `num_entries`; after every block of a cycle, `num_steps`; then
`File.flush()`. Readers trust the counters and ignore anything beyond them.
Probed on the development machine (h5py 3.16, HDF5 2.0): a writer killed with
SIGKILL at random points in 12 trials left 12 files that reopened with every
counter consistent with its data.

A second process cannot open the file while the Wheel holds it, because HDF5
file locking refuses it (`BlockingIOError`, errno 35). Live reading is
in-process through `wheel.l2`; cross-process live reading is a non-goal
(section 10). A second handle in the same process works, so `L2Data.open` on
a file the Wheel has already closed, or has open, is fine.

### 5.3 Cost

One cycle's appends and flush for 1000 blocks with 8 scalar parameters each,
same machine:

| Layout | Per cycle |
|---|---|
| one values and one offsets dataset per key | 0.42 s |
| one values dataset per key, one shapes index per block (this design) | 0.24 s |
| one packed dataset per block, one dtype | 0.09 s |

The chosen layout keeps each chain in one dataset with its own dtype at a
quarter of a millisecond per block per cycle. `benchmarks/benchmark_wheel.py`
gains `--record` to track it.

## 6. `L2Data`

A read-only view of a campaign file. The same class serves `wheel.l2` (over
the Wheel's open handle) and offline reading.

```python
class L2Data:
    @classmethod
    def open(cls, path, *, observed_data: L1Data | None = None) -> L2Data
    def close(self) -> None                    # also a context manager
    num_steps: int                             # committed steps
    block_names: tuple[str, ...]               # registration order
    observation_grid: Grid
    channel_names: tuple[str, ...]
    def role(self, name) -> Role
    def first_step(self, name) -> int
    def block_grid(self, name) -> Grid
    def parameters(self, name, step=-1) -> dict[str, Any]
    def chain(self, name, key) -> np.ndarray
    def entries(self, name, key) -> list[np.ndarray]
    def states(self, step, blocks) -> dict[str, BlockState]
    def covariance(self, step, blocks) -> Covariance | None
    def residual(self, step, blocks) -> L1Data
```

- `open` opens read-only. `observed_data`, when given, must match the stored
  observation grid and channel names (`ValueError` naming both) and is what
  `residual` subtracts from; it carries the orbit blocks may need.
- **Steps.** `step` is an int in `[-(num_steps + 1), num_steps]`; negative
  values count from the latest, so `-1` is the current state. Out of range
  raises `IndexError("step 7 is out of range; the file holds steps 0 to 5")`.
  A block not yet registered at `step` raises
  `ValueError("block 'late' was registered at step 3; it has no state at step 1")`.
- `parameters(name, step)` returns a fresh dict in key order: numpy scalars
  for scalar keys, arrays with the recorded shape otherwise.
- `chain(name, key)` returns a dense array of shape `(num_entries, *shape)`
  when every entry has the same shape, else
  `ValueError("'pop'.'frequency' has entries of different shapes; use entries() for a list")`.
  `entries` always returns a list of arrays.
- `states(step, blocks)`: `blocks` is an iterable of `Block`s with distinct
  names that are all in the file (`ValueError` naming an unknown or repeated
  name); a subset of the recorded blocks is allowed. For each block the
  reference data is the attached observation or, without one, a zero-valued
  `L1Data` built from the stored observation metadata (no orbit), translated
  to the block's grid with the Wheel's helper. The block's
  `state_from_parameters(self.parameters(name, step), reference)` must return
  a `BlockState` (`TypeError` naming the block) whose template, if any, passes
  `check_on_grid` for the block's grid; the state is translated back to the
  observation grid with the Wheel's other helper. Returns a dict in the order
  the blocks were given.
- `covariance(step, blocks)` sums the covariances of the noise blocks recorded
  at `step`, all of which must be supplied (`ValueError` listing the missing
  names); signal blocks in `blocks` are ignored. None when no noise block was
  recorded at that step.
- `residual(step, blocks)` requires attached observed data
  (`ValueError("no observed data attached; pass observed_data to L2Data.open")`)
  and every signal block recorded at `step` (`ValueError` listing the missing
  names); noise blocks in `blocks` are ignored. Returns the observation minus
  the sum of the rebuilt templates (None contributes zero), as a new `L1Data`
  on the observation grid.

## 7. Examples, forge, notebooks, benchmark

| File | Change |
|---|---|
| `examples/toy_fit.py` | `_render` in both blocks becomes `state_from_parameters`; `initial_sample` and `sample` call it. `run_toy_fit(n_cycles=300, burn_in=100, seed=0, *, output_path=None)`: None writes to a temporary directory removed at return. The Wheel is used as a context manager; the RMS progress print stays in `on_cycle_complete`; the chains are read with `wheel.l2.chain(name, key)` after the run. Docstring describes the file. |
| `examples/noise_floor.py` | `ExcessNoiseBlock` gains the method (its render helper becomes it). |
| `examples/gb_model.py` | `GBBlock._render` becomes `state_from_parameters`, taking the parameter dict; `initial_sample` and `sample` build the dict and call it. |
| `examples/domain_translation.py`, `.ipynb` | `DomainEcho.state_from_parameters` returns `BlockState()`. |
| `examples/create_block.ipynb` | Both block classes gain the method (their `_render` and `_result` become it); the contract table and prose list four methods; the known-injection campaign records to a temporary file and reads the amplitude chain from `wheel.l2`. |
| `examples/demo.py`, `demo.ipynb`, `gb_block_eryn.ipynb`, `mojito_wdm_wheel.py` | Only where a block class is defined does it gain the method; files that import blocks are unchanged. |
| `forge/block.py` | `_template` becomes `state_from_parameters(parameters, reference_data)`; still three `# EDIT` methods. |
| `forge/wheel.ipynb` | Records to a temporary file; the chain comes from `wheel.l2.chain("forge", "amplitude")`; printed lines unchanged. |
| `forge/README.md`, `forge/block.ipynb` | Prose names the three methods and the output file. |
| `benchmarks/benchmark_wheel.py` | `FixedSignal` gains the method; `--record` writes to a temporary file and the JSON gains `"recorded"`. |

## 8. Tests

| Behaviour | Test |
|---|---|
| `add` refuses a block without `state_from_parameters` before `initial_sample`; all three error messages list four members; a block lacking the method is not an `isinstance` `Block` | `tests/test_wheel.py`, `tests/test_typing.py` |
| `check_block` round trip: passes for the core and example blocks; fails with the template message for a block whose rebuild differs, and with the covariance message for a noise block; registration and cycles are both checked | `tests/test_testing.py` |
| File created with root and observation attributes; existing path refused; `l2` None and `close` a no-op without a path | `tests/test_l2.py` |
| Registration writes entry 0; each complete cycle writes one entry per block and advances `num_steps`; the record precedes `on_cycle_complete`; a failed cycle records nothing; a refused state leaves the file unchanged | `tests/test_l2.py` |
| `add` after `run`: `first_step`, shorter chain, `states` at an earlier step refuses it | `tests/test_l2.py` |
| Recordable rules: each row of section 4.4 at registration and at `sample`, each leaving ledger, residual, sums, RNG and file unchanged; widening of narrow dtypes; non-finite allowed; zero-length arrays allowed | `tests/test_l2.py` |
| Ragged: changing shapes per step, `chain` refuses with the message, `entries` returns the list; fixed shapes read back dense with the recorded shape; scalars read back as numpy scalars | `tests/test_l2.py` |
| `states`, `covariance`, `residual` at the last step equal `wheel.ledger`, `wheel.covariance` and `wheel.working_residual` to `assert_allclose`, including a block on a WDM grid and a frequency-grid noise block; the error cases of section 6 | `tests/test_l2.py` |
| `L2Data.open` after `close`, with and without `observed_data`; mismatched observed data refused; context managers; reads after close raise | `tests/test_l2.py` |
| Public surface pin gains `L2Data`; mypy accepts the protocol with four members and `Wheel(output_path=...)` | `tests/test_typing.py` |
| `run_toy_fit` with and without `output_path`; the forge notebooks still print their lines | `tests/test_examples.py`, `tests/test_forge.py` |

Gates: full suite with coverage at or above 95 %, `ruff check`,
`ruff format --check`, `mypy`; notebooks never formatted.

## 9. Packaging and documentation

- `pyproject.toml`: `dependencies` gains `"h5py>=3.12.1"` with a comment;
  `uv.lock` regenerated. The `mojito` and `numeric-orbits` extras are
  unchanged; the CHANGELOG notes that `[mojito]` is no longer needed.
- CI (`ci.yml`, `release.yml`): the core-only lane's comment and the
  core-minimum assertion loop stop naming h5py; core-minimum installs
  `h5py==3.12.1` beside the other floors (its interpreter is 3.12).
- `enchilada.__all__` gains `L2Data`; `enchilada/__init__.py` imports it.
- README: "Plugging in your sampler" describes the fourth method and the
  round-trip check; a new "Campaign output" section after it shows
  `Wheel(..., output_path=...)`, `wheel.l2.chain`, `L2Data.open` and
  `residual`; API rows for `Wheel(output_path=)`, `wheel.l2`, `close`,
  `L2Data` and `state_from_parameters`; the Known limitations entry becomes
  "No restart: the output file records parameters, not block sampler states
  or the RNG, so a campaign cannot resume from it".
- CHANGELOG `## [0.4.0] — Unreleased`. Breaking: `state_from_parameters`
  required of every block and checked at `Wheel.add`; h5py is a core
  dependency. Added: `Wheel(output_path=...)`, `wheel.l2`, `Wheel.close` and
  context-manager support, `L2Data`, the `check_block` round trip,
  `run_toy_fit(output_path=)`, `benchmark_wheel.py --record`.
- Docstrings: `Block`, `BlockState.parameters` (recordable values), `Wheel`
  class, `add`, `run`, `check_block`.

## 10. Non-goals

- Checkpoint or restart: block sampler states and the RNG are not recorded,
  and an existing file is refused rather than appended to.
- Storing templates, residuals or the observation arrays in the file.
- The orbit in the file; it arrives with `observed_data` at `open`.
- Cross-process reading while the Wheel holds the file (SWMR or lock-free
  opening), and any buffering or periodic flush policy.
- Writing simulation truth. The layout supports it as one entry per block;
  the writer comes with the simulation work.
- Non-numeric parameters, nested parameter dicts, or a key set that changes
  after registration.
