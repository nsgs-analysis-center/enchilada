# Covariance simplification design

Date: 2026-11-11
Status: approved design (five sections approved in discussion 2026-11-11; the PSD
and variance helpers removed on review 2026-11-12; sections 2.8, 3 and 3.5
withdrawn on 2026-11-12 after domainia's 3.0 spec, user decision "core now,
components later"); approved for planning
Branch: `covariance-simplification` (from `main` at a827b69)
Supersedes: sections 1, 3.2 to 3.4, 4 and 6 of
`2026-11-10-covariance-object-design.md`, and
section 4 (the enchilada adapter) of domainia's
`docs/superpowers/specs/2026-10-31-domainia-general-library-design.md`, which still
names `TranslatedCovariance` and `DataCovariance` as they were before 0.3.0.
Coordinated with domainia's
`docs/superpowers/specs/2026-11-12-domainia-translation-core-design.md` (3.0),
whose section 8 asks for the withdrawals recorded here. The noise-components
layer it describes is a separate enchilada spec.

## Goal

Make `Covariance` smaller and easier to maintain without removing a capability
anything uses (a convenience that is one line of `like` away is not a
capability), and remove the covariance translation seam left in 0.3.0. That seam
assumed covariance arrays would come back on another grid, and four of its
seven strict xfails assert a diagonal does. domainia 3.0 (spec 2026-11-12) will
instead evaluate noise models per grid and never moves a covariance array
between domains. This design keeps every block on the grid of the covariance it
reads; delivering a noise model to a block on another grid is the components
layer, a separate spec that follows domainia 3.0.

## Findings that drove the design

Census of `src/enchilada/covariance.py` at a827b69 (764 lines, 74 tests), by
callers outside the module and its own test file:

| Feature | Lines (approx.) | Callers elsewhere |
|---|---|---|
| `form="full"`: shape rule, axis-swapping conjugate transpose, block assembly, Hermitian symmetrisation, point-diagonal extraction, the over-64-channel grouping | 150 | one test proving a block rejects it |
| XYZ to AET derivation, its cache, the bidirectional link, `__getstate__`/`__setstate__` | 110 | none |
| Cross-channel pairs and the per-point positive-definiteness check | 60 | two blocks asserting the cross terms are zero |
| Attribute access (`cov.AE`), the reserved-name and ambiguity rules | 70 | `cov.A` in one notebook |
| `from_psd`, `from_variance`, `noise_psd`, `noise_variance` and the `psd(freqs[, channel])` model protocol | 120 | tests, examples and the README; nothing in the package calls them |

Every example and benchmark uses A, E, T data with independent channels and
reads per-point variances through `noise_variance` or `noise_psd`. The only
call to a model's `psd` inside the package is in `from_psd`: the framework
never needs a PSD, only a covariance on a grid, so PSD knowledge belongs to
the noise blocks.

domainia facts:

- Its 3.0 spec (2026-11-12, approved) deletes the operator and lazy-translation
  layer (`translate.py`, `Operator`, `TranslatedOperator`, `translate_operator`,
  `domains.py`) and every likelihood. It keeps `transform` on arrays and adds
  `frequency_covariance` and `wdm_covariance`, which evaluate a K-by-K noise
  model on a frequency or WDM grid. No function moves a covariance array between
  domains, so nothing in domainia can back the 0.3.0 seam.
- Both packages use `X = dt * rfft(x)` for frequency data, the dense
  `(Nf + 1, Nt)` WDM layout, the same active-pixel rule (edge bands active at
  even time index) and the Meyer window. No units shim is needed.
- The domainia repository is private and its PyPI name belongs to an unrelated
  package.

## Decisions made in discussion

| Question | Decision |
|---|---|
| What a block on another grid than the covariance receives | Nothing yet: the Wheel refuses the run with `ValueError` naming both grids. Delivery through noise components is the separate spec that follows domainia 3.0. (Withdrawn 2026-11-12: an operator backed by `domainia.translate_operator`, which domainia 3.0 deletes.) |
| Channel features | Keep cross terms, the positive-definiteness check and the XYZ to AET derivation |
| Grid on the object | Yes; reverses the 2026-11-10 grid-free decision |
| Storage layout | One stacked `(K, K, *S)` Hermitian array; the friendly dict stays the input; reverses the 2026-11-10 one-array-per-pair decision |
| The full form | Removed |
| PSD and variance helpers | Removed: `from_psd`, `from_variance`, `noise_psd`, `noise_variance` and the `psd(...)` model protocol; a noise block builds its covariance with `like` and reads `pair` |
| domainia dependency | None in this release; the core stays NumPy-only |
| Where the Wheel keeps the covariance | One native `Covariance` on its owner's grid; nothing is translated back |
| Version | 0.4.0 |

## 1. `Grid`

Module `src/enchilada/domains.py`. A frozen dataclass with the four fields that
fix a channel's array layout:

```python
Grid(data_domain: str, num_time_samples: int, sample_rate_hz: float, wdm_grid: WDMGrid | None = None)
```

Validation (`ValueError` unless noted): `data_domain` in `DOMAINS`, which moves
to `domains.py` and `L1Data.DOMAINS` re-exports; `num_time_samples` a positive
integer (`bool` excluded); `sample_rate_hz` positive and finite; `wdm_grid` a
`WDMGrid` (`TypeError` otherwise) exactly when `data_domain == "wdm"`, with
`wdm_grid.num_time_samples == num_time_samples`.

Members:

| Member | Value |
|---|---|
| `channel_shape` | `(N,)` time, `(N // 2 + 1,)` frequency, `wdm_grid.array_shape` WDM |
| `active_mask` | all true, except `wdm_grid.active_mask` on WDM |
| `observation_duration_s` | `num_time_samples / sample_rate_hz` |
| `same_record(other)` | `num_time_samples` and `sample_rate_hz` equal |

Equality is dataclass equality over the four fields, so two grids of one record
in different domains compare unequal. `L1Data.grid` is a property returning the
data's `Grid`. The epoch `start_time_gps` is not part of a grid: a stationary
noise model has no epoch, and `transform` does not shift records.

## 2. `Covariance`

Module `src/enchilada/covariance.py`. `@dataclass(frozen=True, eq=False)`.

### 2.1 Fields

```python
blocks: np.ndarray                 # (K, K, *grid.channel_shape), read-only, Hermitian at every point
channel_names: tuple[str, ...]     # keyword-only
grid: Grid                         # keyword-only
```

`blocks[i, j]` is the covariance between channels `i` and `j` at every point of
the grid, `E[x_i conj(x_j)]`; `blocks[j, i]` is its complex conjugate. There is
one form: one value per point. A covariance between different points is out of
scope (it was the removed full form).

### 2.2 Construction and validation

`Covariance(blocks, *, channel_names, grid)` validates in this order:

1. `channel_names`: a tuple or list of unique non-empty strings (`TypeError`
   for a non-string, `ValueError` otherwise), at least one.
2. `grid`: a `Grid` (`TypeError`).
3. `blocks`: an `np.ndarray` of floating or complex dtype (`TypeError`), shape
   `(K, K) + grid.channel_shape`, every entry finite (`ValueError`).
4. Hermitian exactly: `blocks` equals `conj(swapaxes(blocks, 0, 1))` element for
   element. `like` (section 2.3) and `in_basis` produce exact symmetry; a caller
   assembling a product symmetrises first.
5. Real where the domain requires it. Time and WDM grids: a complex dtype
   raises `ValueError`, because complex blocks are valid on a frequency grid,
   so the dtype is a wrong value, not a wrong kind. Frequency
   grids: every diagonal entry
   `blocks[i, i]` has zero imaginary part at every bin, and every entry at DC
   and at Nyquist for even `N` has zero imaginary part (`ValueError`).
6. Variances `blocks[i, i].real` are positive at every active point. At an
   inactive WDM pixel the whole `K`-by-`K` block is zero (`ValueError`).
7. Positive definite at every active point: one batched
   `np.linalg.cholesky` over the `(P, K, K)` stack of active points;
   `LinAlgError` becomes `ValueError("covariance must be positive definite at
   every active point")`.
8. The array is stored as a fresh float64 or complex128 copy with
   `writeable=False`.

There is no reserved-name check on channel names: a channel whose name equals
an attribute of the class is reachable through `pair()` only (section 2.4).

### 2.3 Constructor from a mapping

`Covariance.like(data, pairs)` is the one classmethod. It takes the `L1Data`
whose grid and channel order the covariance adopts. Keys are a channel name (`"A"`), a two-channel
concatenation resolved by `split_pair_name` (`"AE"`), or a `(first, second)`
tuple. An unknown name or an unresolvable string raises `ValueError`. Any key
`(a, b)` sets `blocks[a, b] = value` and `blocks[b, a] = conj(value)`; two keys
naming one pair in either orientation raise `ValueError`. Every channel's own
entry is required; an absent off-diagonal pair is zero. Each value must have
`grid.channel_shape` (`ValueError`). The result is validated by the constructor.

`from_psd` and `from_variance` are removed. A noise model, white or coloured,
is a block's business: the block computes its per-point variances on the grid
of the data it received and passes them to `like`. The package never calls a
model's `psd` method and defines no protocol for one. Section 2.6 records the
one convention a block needs.

### 2.4 Access

`pair(first, second)` returns `blocks[i, j]`, a read-only view; an unknown
channel raises `ValueError`.

`__getattr__(name)` runs only when normal lookup fails. Names starting with
`_` and the field names raise `AttributeError`. Otherwise
`split_pair_name(name, channel_names)` resolves a channel or a concatenation of
two to `pair(...)`; an ambiguous split raises `AttributeError` with the message
from `split_pair_name`. When the channel set is exactly `{X, Y, Z}` or exactly
`{A, E, T}` and the name resolves in the other basis, the value is read from
`in_basis(other)`. Anything else raises `AttributeError` listing the valid
names: the channel names, their ordered concatenations, and the other basis's
when it exists.

### 2.5 Derived basis

`in_basis(basis)`, `basis` in `("XYZ", "AET")` (`ValueError` otherwise), needs
the channel set to be exactly one triple (`ValueError` naming the channels).
The current basis returns `self`. Otherwise, with `M` the orthonormal matrix of
the 2026-11-10 spec section 3.5 (`AET_FROM_XYZ`, or its transpose for AET to
XYZ) and `B` the blocks reordered to `("X", "Y", "Z")` or `("A", "E", "T")`:

```python
rotated = np.einsum("ai,bj,ij...->ab...", M, M, B)
rotated = 0.5 * (rotated + np.conj(np.swapaxes(rotated, 0, 1)))   # exact Hermitian
rotated[i, i] = rotated[i, i].real  for every i                   # real variances
return Covariance(rotated, channel_names=target, grid=self.grid)
```

There is no cache and no back-link; the caller keeps the result. A
`ValueError` from the constructor is re-raised as `ValueError` explaining that
the rotated covariance is not numerically positive definite (one channel's
variance below float64 roundoff of the others, for example T far below A at the
lowest LISA frequencies), chaining the original. `hasattr(cov, "X")` can
therefore raise `ValueError`, as in 0.3.0.

### 2.6 Units

`blocks` holds variances in the squared units of the data's coordinates on its
grid: `x_n**2` in time, `(dt * rfft(x))**2` in frequency, squared WDM
coefficients in WDM. The package converts neither to nor from a one-sided PSD,
so `noise_psd` and `noise_variance` are removed; `pair(ch, ch)` is the
read-back. The one convention a noise block needs follows from
`X = dt * rfft(x)`: stationary noise with one-sided PSD `S` has bin variance
`C_k = (Tobs / 2) * S(f_k)` at every rfft bin, DC and Nyquist included, with
`Tobs = N / fs`. White noise of variance `sigma**2` per sample therefore has
`C_k = N * dt**2 * sigma**2` at every bin and, on a WDM grid, `dt * sigma**2`
at every active pixel. The README states these beside the Fourier convention,
where a block author looks for them.

### 2.7 Copies

Two copy paths, each with one job:

- `copy()` is `Covariance(self.blocks, channel_names=self.channel_names,
  grid=self.grid)`: a fresh buffer, revalidated, including the batched
  Cholesky. It is the trust boundary: the Wheel calls `Covariance.copy(x)`
  unbound on every covariance it receives (construction, seeds, block returns),
  so a subclass cannot skip validation and an array whose write flag a caller
  flipped is caught.
- `deepcopy` and pickle restore without validating. `__reduce__` returns a
  module-level restore function with the three fields; it builds the object
  with `object.__new__`, sets the fields, and marks the array read-only. One
  array copy, no Cholesky. The Ledger snapshots, the pre-call state snapshot,
  `transform(block_state)` and the Wheel's handoffs of the covariance it
  already validated (`_block_inputs`, `Wheel.covariance`) all use this path,
  which replaces the 0.3.0 `_copy_validated` fast path and its two pickle
  hooks with one method. The cost of a snapshot is one array copy, as in
  0.3.0; `benchmarks/benchmark_wheel.py` gates it.

### 2.8 No likelihood methods

`Covariance` has no `apply`, `solve`, `project`, `quadratic_form` or
`log_determinant`. Likelihoods, their real-coordinate weights (an interior
frequency bin is two real coordinates; DC and the even-`N` Nyquist bin are one)
and any Jacobian between domains belong to blocks, as domainia's 3.0 spec puts
it (its section 1). The 0.3.0 removal of those methods stands. (Withdrawn
2026-11-12: NumPy implementations of domainia's `Operator` protocol on
`Covariance`.)

## 3. The seam

(Withdrawn 2026-11-12: `_domainia.py`, `NativeOperator`, `TranslatedCovariance`,
`NoiseModel`, the covariance overload of `transform` returning a translated
view, and the domainia CI lane. domainia 3.0 deletes the layer they were built
on, and its section 8 asks for these withdrawals.)

### 3.1 No covariance translation

`transform` no longer accepts a `Covariance`: its overloads become
`L1Data | BlockState`, and a `Covariance` argument raises `TypeError` saying
that a covariance carries its grid and does not translate, and that `copy()`
makes an independent copy. `reference_data` keeps its one role, the source grid
of a `BlockState` template. `transform(block_state, ...)` translates the
template as today and carries `covariance` through unchanged apart from the
deep copy, since it knows its own grid.

### 3.2 Block contract

`Block.sample(conditional_residual, covariance: Covariance | None,
current_block_state, *, rng)` is unchanged. The covariance a block receives is
always a plain `Covariance` on the grid of `conditional_residual`.
`BlockState.covariance` stays `Covariance | None`.

### 3.3 Packaging and CI

No domainia, no new workflow, no new marker. The `wdm.yml` lane keeps running
the WDM tests, including the former xfails that need the backend for their
data transforms; its file list drops `tests/test_covariance_translation.py`,
which 0.3.0 deleted.

## 4. Wheel and the block contract

The Wheel holds one native `Covariance`, on whatever grid its owner published
it. The 0.3.0 rule that it lives on the observation grid cannot survive without
translation: a WDM publication has no array form on a time grid.

- `Wheel(observed, initial_covariance=None, *, random_seed=None)`: a native
  `Covariance` (`TypeError` otherwise) whose grid is the same record as the
  observations and whose channel names match (`ValueError`); any domain is
  allowed. Stored as `Covariance.copy(...)`.
- A block registered with `data_domain=None` is assigned the observation
  `Grid`, so "the block's grid" is defined for every block, and the comparison
  below stays meaningful after a noise block in another domain has published.
- `_block_inputs` returns `deepcopy(self._covariance)` when the block's grid
  equals the covariance's grid. When they differ it raises `ValueError` naming
  the block, both grids and the fix (register the block on the covariance's
  grid, or have the noise block publish on the block's grid); the ledger, RNG
  and covariance are unchanged. Every example, benchmark and former xfail runs
  on equal grids.
- On return, a published covariance must be on the block's own grid: `_adopt`
  requires `covariance.grid` to equal the block's `Grid` and its channel names
  to equal the observations' (`ValueError` naming both grids), then stores
  `Covariance.copy(...)`. Nothing is translated back. The noise-owner rule,
  `NoiseOverwrittenWarning` and the atomic commit are unchanged.
- A seed covariance in `add` goes through the same `_adopt` check against the
  selected `Grid`, so the `_empty_channel` helper and the empty-reference
  construction are deleted.
- `Wheel.covariance` returns `deepcopy(self._covariance)`; callers read its
  `grid`. The stored object is always a plain `Covariance` that passed
  `copy()` at adoption, so the unvalidated restore is safe here.
- The internal `DomainSelection` tuple `(domain, wdm_grid)` becomes a `Grid`
  built once in `add`; `_target_grid` returns the `WDMGrid` it does today and
  `add` wraps it.

What a block sees: a plain `Covariance` on its own grid, read through
`covariance.A`, `pair`, `blocks` and `in_basis`.

## 5. Errors

`TypeError` for a wrong kind of argument (a non-array, a non-`Grid`, a
`Covariance` passed to `transform`);
`ValueError` for a wrong value (a shape, a non-finite entry, a complex array on
a time or WDM grid, a non-Hermitian block, a non-positive variance, a grid or
channel mismatch, an unknown channel or basis); `AttributeError` for
an unknown attribute name. `NotImplementedError` disappears from the package,
and `exclude_also` in the coverage configuration drops it.

## 6. Testing

- `tests/test_covariance.py` is rewritten around the new layout: construction
  rules one by one, `like`, `pair` and the attribute sugar, `in_basis` against
  the explicit `M C M^T` product and its round trip, the T-below-roundoff
  error, copy, deepcopy and pickle, and `check_compatible`. About 35 tests.
- `tests/conftest.py` gains `white_covariance(data, variance)`: `like` with the
  section 2.6 constants for the data's domain (`sigma**2` per time sample,
  `N * dt**2 * sigma**2` per frequency bin, `dt * sigma**2` at active WDM
  pixels and zero elsewhere). It replaces the `from_variance` and `from_psd`
  calls in tests and the `FlatPSD`, `UnitPSD` and `White` test doubles.
- `tests/test_grid.py` covers `Grid` validation and members and `L1Data.grid`.
- `tests/test_translation.py`: a `Covariance` passed to `transform` raises
  `TypeError`; a `BlockState` with a covariance translates its template and
  carries the covariance unchanged and unaliased; the former seam tests that
  expected `NotImplementedError` go.
- `tests/test_wheel_translation.py`: a block registered on another grid than
  the Wheel's covariance is refused at `run` with `ValueError` naming both
  grids, with the ledger, RNG and covariance unchanged; a seed covariance or a
  returned covariance on another grid than the block's is refused at `add` and
  at `run` the same way. These replace the three seam tests there.
- The seven strict xfails become passing tests, all under section 4 and none
  needing domainia: the toy time blocks on frequency and WDM observations (the
  noise block publishes on the time grid they sample on), the GB probe seeded
  with its native frequency covariance on time and WDM observations, the
  noise-only frequency seed, the block-state round trip, which now builds its
  WDM covariance with `like` on the WDM reference since `transform` no longer
  takes a covariance, and the WDM publication, whose only block reads the
  covariance on the WDM grid it published on. The WDM cases keep `requires_wdm`
  for the data transforms.
- The isolation test in `tests/test_state.py` keeps passing: every handoff is a
  `copy()` or a `deepcopy`.
- Coverage: the gate stays at its current threshold.

## 7. Migration

All mechanical:

- `noise_variance(ch, data)` and `noise_psd(ch, data)` calls (about 40 sites in
  tests, examples and the README) become `pair(ch, ch)` reads in the units of
  section 2.6, or go where a test only exercised the read-back. Two blocks
  consume the value: the toy block in `examples/toy_fit.py` divides by the
  per-sample array instead of a scalar, and the GB block in
  `examples/gb_model.py` feeds a narrowband likelihood written in PSD units,
  so it converts the bin variances back with `2 * C / Tobs` from
  `conditional_residual.Tobs` rather than rewriting the likelihood.
- `transform(cov, ...)` calls (14 files) go: a same-grid call becomes
  `copy()` or is dropped where the Wheel now hands the covariance over
  directly, and a cross-grid call is rewritten as section 6 says. The
  `assert_type` line for it in `tests/test_typing.py` becomes one for `copy()`.
- `covariance.form != "diagonal"` guards in `examples/toy_fit.py`,
  `examples/gb_model.py` and `examples/create_block.ipynb` are deleted: there
  is one form. The cross-term checks those blocks make stay.
- `from_variance` (about 60 sites, nearly all tests) and `from_psd` (about 22)
  become `like` calls: tests through `white_covariance`;
  `examples/toy_fit.py`, `benchmarks/benchmark_wheel.py`,
  `examples/create_block.ipynb` and `examples/domain_translation.ipynb` build
  the constant arrays in place; `examples/gb_model.py` keeps
  `FixedLISANoise.psd` as user code, and both its noise publication and the
  initial covariance in `examples/gb_block_eryn.ipynb` pass
  `(Tobs / 2) * noise.psd(freqs)` per channel to `like`, with the model
  floored at low frequency as today. The test PSD doubles are deleted.
- `Covariance.like` call sites are unchanged.
- README: the attribute API, the shape rule (one form), `Grid`, and the
  section 2.6 units beside the Fourier convention in place of the `from_psd`
  passage and the `psd(frequencies[, channel])` protocol; the translation
  section says a covariance does not translate and that a block reads it on
  its own grid; the Known limitations bullet "No covariance translation yet"
  becomes "No covariance delivery across grids yet": a campaign that carries a
  covariance keeps the noise owner and every block that reads it on one grid,
  until the components layer that follows domainia 3.0.
- CHANGELOG 0.4.0: Added `Grid`, `L1Data.grid`; Changed the storage layout,
  `like`'s grid capture, the Wheel's storage rule and its refusal of a block on
  another grid; Removed the full form, `form`, the derived-basis cache,
  `from_psd`, `from_variance`, `noise_psd`, `noise_variance`, the covariance
  overload of `transform` with its `reference_data`, `NotImplementedError`
  from the seam.
- `pyproject.toml` version 0.4.0.
- `docs/wheel.svg` is the user's and is left alone.

## 8. Non-goals

- Delivering a covariance to a block on another grid, noise components with
  several owners, and domainia 3.0's `transform`, `frequency_covariance` and
  `wdm_covariance`: the components spec that follows domainia 3.0.
- Moving the data transforms (`_transforms.py`, the `wdm` backend) to domainia.
- Band-limited or partial-grid covariances.
- A WDM window parameter on `WDMGrid`.
- Non-stationary or evolutionary noise models.
- Likelihood methods on `Covariance`.

Each is a separate spec.
