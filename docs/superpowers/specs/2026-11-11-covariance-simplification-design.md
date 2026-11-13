# Covariance simplification design

Date: 2026-11-11
Status: approved design (five sections approved in discussion 2026-11-11; the PSD
and variance helpers removed on review 2026-11-12); awaiting the user's review
of the written spec
Branch: `covariance-simplification` (from `main` at a827b69)
Supersedes: sections 1, 3.2 to 3.4, 4 and 6 of
`2026-11-10-covariance-object-design.md`, and
section 4 (the enchilada adapter) of domainia's
`docs/superpowers/specs/2026-10-31-domainia-general-library-design.md`, which still
names `TranslatedCovariance` and `DataCovariance` as they were before 0.3.0.

## Goal

Make `Covariance` smaller and easier to maintain without removing a capability
anything uses (a convenience that is one line of `like` away is not a
capability), and connect it to the translation layer that exists. `domainia`
(`~/Documents/lisa/domainia`, 2.0.0) translates a covariance lazily as an
*operator* seen from another domain; it never produces covariance arrays on the
target grid. The seam left in 0.3.0 assumed arrays would come back, and four of
its seven strict xfails assert a diagonal does. This design replaces that seam.

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

domainia facts checked against its source:

- `translate_operator(op, target)` returns a `TranslatedOperator` holding a
  reference to the native operator; every call moves `{channel: array}` dicts
  with `transform` and applies the coordinate Jacobian. No dense matrix is built.
- The `Operator` protocol is `domain`, `channel_names`, `degrees_of_freedom`,
  `apply`, `solve`, `project`, `quadratic_form`, `log_determinant`, acting on
  the stacked real coordinates: time samples; active WDM cells; for Fourier
  `Re X_0`, `sqrt(2) Re X_k`, `sqrt(2) Im X_k` and `Re X_{N/2}` for even `N`.
- Both packages use `X = dt * rfft(x)` for frequency data, the dense
  `(Nf + 1, Nt)` WDM layout, the same active-pixel rule (edge bands active at
  even time index) and the Meyer window. No units shim is needed.
- domainia has no frequency-native per-bin operator (`PointwiseOperator` takes
  time or WDM; `DenseOperator` is `(KN, KN)`), so the native operator for an
  enchilada covariance lives in enchilada.
- The domainia repository is private and its PyPI name belongs to an unrelated
  package.

## Decisions made in discussion

| Question | Decision |
|---|---|
| What an off-grid block receives | An operator it queries, backed by `domainia.translate_operator` |
| Channel features | Keep cross terms, the positive-definiteness check and the XYZ to AET derivation |
| Grid on the object | Yes; reverses the 2026-11-10 grid-free decision |
| Storage layout | One stacked `(K, K, *S)` Hermitian array; the friendly dict stays the input; reverses the 2026-11-10 one-array-per-pair decision |
| The full form | Removed |
| PSD and variance helpers | Removed: `from_psd`, `from_variance`, `noise_psd`, `noise_variance` and the `psd(...)` model protocol; a noise block builds its covariance with `like` and reads `pair` |
| domainia dependency | Optional and private; no PyPI extra; the core stays NumPy-only |
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
scope (it was the removed full form, and domainia's `DenseOperator` covers it).

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
   raises `ValueError`, as in domainia, because complex blocks are valid on a
   frequency grid so the dtype is a wrong value, not a wrong kind. Frequency
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

### 2.8 Operator methods

`Covariance` implements the five methods of domainia's `Operator` protocol in
NumPy, on its own grid, with no domainia import. `values` is a
`{channel: array}` mapping or an `L1Data`; its channels must be exactly
`channel_names` (`ValueError`), each array `grid.channel_shape` and finite
(`ValueError`), complex only on a frequency grid (`ValueError`), and real at DC
and even-`N` Nyquist on a frequency grid (`ValueError`). `apply`, `solve` and
`project` return dicts in channel order with exact zeros at inactive WDM pixels.

| Method | Value |
|---|---|
| `apply(values)` | `C_p x_p` at every point: `einsum("ij...,j...->i...", blocks, X)` |
| `solve(values)` | `C_p^{-1} x_p` at every active point: one batched `np.linalg.solve` |
| `project(values)` | `x_p` at active points, zero elsewhere |
| `quadratic_form(values)` | `sum_p w_p * Re(x_p^H C_p^{-1} x_p)` |
| `log_determinant()` | `sum over active p of w_p * log det C_p`, from the Cholesky factors |
| `degrees_of_freedom` | `K * num_time_samples` in every domain |

`w_p` is the real-coordinate weight that makes a translated quadratic form
equal the native one. Time samples and WDM pixels weigh 1 (0 at inactive
pixels). On a frequency grid the interior bins `1 .. (N - 1) // 2` weigh 2,
because each complex coefficient is two real coordinates
`sqrt(2) Re X_k, sqrt(2) Im X_k`; DC and the even-`N` Nyquist bin weigh 1. By
Parseval for `X = dt * rfft(x)`, `sum_n x_n^2 = (X_0^2 + 2 sum_int |X_k|^2 +
X_{N/2}^2) / (N dt^2)`, and white noise has `C_k = N dt^2 sigma^2` at every bin,
so the weighted frequency form equals the time form `sum_n x_n^2 / sigma^2`
exactly, which the unweighted sum does not. Half the quadratic form is the
Whittle exponent `sum_int |X_k|^2 / C_k + 0.5 sum_end X_k^2 / C_k`. The
real-coordinate covariance of one interior bin with `K` correlated channels is
the `2K`-by-`2K` embedding `[[Re C, -Im C], [Im C, Re C]]`, whose determinant
is `|det C|^2`, which is where the weight 2 on `log det C_k` comes from. A block
writes its Gaussian log-likelihood as
`-0.5 * (cov.quadratic_form(residual) + cov.log_determinant())` in any domain.

`apply`, `solve` and `project` set the DC and even-`N` Nyquist entries of a
frequency result to their real part explicitly, as domainia's `transform` does
for its own output, so a `TranslatedCovariance` never refuses a native result
over a `1e-17` imaginary residue from the complex solver.

## 3. The adapter and the seam

### 3.1 `_domainia.py`

A private module and the only place that imports domainia, inside functions:

- `require_domainia()` returns the module or raises `ImportError` saying that
  translating a covariance between domains needs domainia and how to install
  it from its repository.
- `to_domain(grid)` maps a `Grid` to `TimeDomain(N, dt)`,
  `FourierDomain.full(N, dt)` or `WDMDomain.full(wilson.Grid(Nf, Nt, dt))` with
  the default Meyer window. The time domain's `t0` is left at its default; a
  `TranslatedCovariance` never compares two time domains, because equal grids
  never reach the seam.
- `NativeOperator(covariance)` presents a `Covariance` as a domainia
  `Operator`: `domain` is `to_domain(covariance.grid)`; the other members
  delegate.

### 3.2 `TranslatedCovariance`

Public, in `src/enchilada/translation.py`. `@dataclass(frozen=True,
eq=False)` with fields `native: Covariance` and `grid: Grid`. Construction
requires both kinds (`TypeError`), `native.grid.same_record(grid)` and
`grid != native.grid` (`ValueError`), then builds
`domainia.translate_operator(NativeOperator(native), to_domain(grid))` once and
keeps it privately.

Members: `channel_names`, `degrees_of_freedom`, `check_compatible(data)` (grid
equality and channel names, as the native), and `apply`, `solve`, `project`,
`quadratic_form`, `log_determinant` with the same input rules as section 2.8,
answered on `grid` through the translated operator. It has no `blocks`,
`pair` or `in_basis`: a block that needs
per-point arrays tests `isinstance(covariance, Covariance)` and raises
otherwise, which is what the `form` guards in `examples/toy_fit.py`,
`examples/gb_model.py` and `examples/create_block.ipynb` do today.

### 3.3 `NoiseModel`

A `runtime_checkable` `Protocol` in `src/enchilada/block.py` naming the shared
surface: `grid`, `channel_names`, `degrees_of_freedom`, `check_compatible`,
`apply`, `solve`, `project`, `quadratic_form`, `log_determinant`.
`Block.sample(conditional_residual, covariance: NoiseModel | None,
current_block_state, *, rng)`. `BlockState.covariance` stays
`Covariance | None`: a block publishes arrays on its own grid, never a
translated view (`TypeError`).

### 3.4 `transform`

```python
transform(covariance, target_domain, *, num_frequency_divisions=None, num_time_divisions=None)
    -> Covariance | TranslatedCovariance
```

`reference_data` is removed from the covariance overload and raises
`TypeError` if passed with one. The target `Grid` is built from the native's
record and `_target_grid(...)` (a WDM target without counts reuses the native's
WDM grid when it has one, else `resolve_wdm_grid` raises as for data). A target
equal to the native grid returns `native.copy()`; another grid returns
`TranslatedCovariance(native, target)`. A `TranslatedCovariance` input is
re-targeted from its `native`, never nested, as domainia does.

`transform(block_state, ...)` translates the template as today, with
`reference_data` for the template's source grid, and carries `covariance`
through unchanged apart from the deep copy, since it knows its own grid.

### 3.5 Packaging and CI

domainia is not declared in `pyproject.toml`: it is private, and a bare
`domainia` requirement would resolve to the unrelated PyPI package. The README
documents installing it from its repository into the environment. Tests that
need it skip with a `requires_domainia` marker mirroring `requires_wdm`. CI
gains `.github/workflows/domainia.yml` modelled on `wdm.yml`: `workflow_call`,
a read-only deploy key secret, a pinned revision, `uv pip install --no-deps`
into the locked environment, and a step that fails if the import is missing so
the skip cannot hide an install problem. domainia requires `numpy>=2.1` and
`scipy`, so the core-minimum lane never installs it. The core stays NumPy-only.

## 4. Wheel and the block contract

The Wheel holds one native `Covariance`, on whatever grid its owner published
it. The 0.3.0 rule that it lives on the observation grid cannot survive the
operator design: a WDM publication has no array form on a time grid.

- `Wheel(observed, initial_covariance=None, *, random_seed=None)`: a native
  `Covariance` (`TypeError` otherwise) whose grid is the same record as the
  observations and whose channel names match (`ValueError`); any domain is
  allowed. Stored as `Covariance.copy(...)`.
- A block registered with `data_domain=None` is assigned the observation
  `Grid`, so "the block's grid" is defined for every block, and the comparison
  below stays meaningful after a noise block in another domain has published.
- `_block_inputs` returns `deepcopy(self._covariance)` when the block's grid
  equals the covariance's grid, else
  `transform(self._covariance, domain, num_frequency_divisions=...)`. The seam
  is reached only when the two grids differ, so every example and benchmark
  runs without domainia.
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

What a block sees: `covariance.quadratic_form(residual)` and
`log_determinant()` always; `covariance.A`, `pair` and `in_basis` when it is a
native `Covariance`.

## 5. Errors

`TypeError` for a wrong kind of argument (a non-array, a non-`Grid`, a
`TranslatedCovariance` in a `BlockState`);
`ValueError` for a wrong value (a shape, a non-finite entry, a complex array on
a time or WDM grid, a non-Hermitian block, a non-positive variance, a grid or
channel mismatch, an unknown channel or basis); `AttributeError` for
an unknown attribute name; `ImportError` from the seam without domainia.
`NotImplementedError` disappears from the package, and `exclude_also` in the
coverage configuration drops it.

## 6. Testing

- `tests/test_covariance.py` is rewritten around the new layout: construction
  rules one by one, `like`, `pair` and the attribute sugar, `in_basis` against
  the explicit `M C M^T` product and its round trip, the T-below-roundoff
  error, copy, deepcopy and pickle,
  `check_compatible`, and the five operator methods against dense
  real-coordinate matrices on small grids in time, frequency and WDM, including
  a complex Hermitian cross term in frequency. About 40 tests.
- `tests/conftest.py` gains `white_covariance(data, variance)`: `like` with the
  section 2.6 constants for the data's domain (`sigma**2` per time sample,
  `N * dt**2 * sigma**2` per frequency bin, `dt * sigma**2` at active WDM
  pixels and zero elsewhere). It replaces the `from_variance` and `from_psd`
  calls in tests and the `FlatPSD`, `UnitPSD` and `White` test doubles.
- The weights of section 2.8 are pinned natively, with no domainia: the same
  white noise built with `white_covariance` on time data and on its spectrum
  gives equal `quadratic_form` on `x` and on `dt * rfft(x)`, and
  `log_determinant` values that differ by exactly
  `2 * K * N * (log dt + 0.5 * log N)`, domainia's coordinate Jacobian. The
  WDM covariance on a zero-valued WDM `L1Data` pins the WDM leg of
  `log_determinant`, `K * N * log dt` above the time value, without the WDM
  backend.
- `tests/test_grid.py` covers `Grid` validation and members and `L1Data.grid`.
- `tests/test_translated_covariance.py` skips without domainia and pins: same
  grid returns a copy; another grid returns a `TranslatedCovariance`;
  `quadratic_form` and `log_determinant` of a translated white covariance agree
  with the native across time, frequency and WDM; `solve` round-trips through
  `apply`; re-targeting never nests; a `TranslatedCovariance` is refused by
  `BlockState`. In `tests/test_translation.py`, which runs everywhere, the
  seam raises `ImportError` when domainia is absent, tested by hiding the
  module.
- The seven strict xfails become passing tests. Six pass without domainia under
  section 4: the toy time blocks on frequency and WDM observations (the noise
  block publishes on the time grid they sample on), the GB probe seeded with
  its native frequency covariance on time and WDM observations, the noise-only
  frequency seed, and the block-state round trip, which now builds its WDM
  covariance with `like` on the WDM reference rather than through `transform`,
  since `transform` across grids returns a `TranslatedCovariance` that
  `BlockState` refuses. The WDM publication test
  adds a time-domain block that calls `quadratic_form` and skips without
  domainia.
- The isolation test in `tests/test_state.py` keeps passing: every handoff is a
  `copy()`.
- Coverage: the gate stays at its current threshold. `_domainia.py` and the
  `TranslatedCovariance` method bodies are excluded with `# pragma: no cover`
  because the per-PR lanes have no domainia; the domainia lane runs them as
  tests.

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
- `transform(cov, ..., reference_data=...)` drops the keyword (14 files).
- `covariance.form != "diagonal"` guards in `examples/toy_fit.py`,
  `examples/gb_model.py` and `examples/create_block.ipynb` become
  `not isinstance(covariance, Covariance)`.
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
- README: the attribute API, the shape rule (one form), the operator methods,
  the seam, the domainia install note, and the section 2.6 units beside the
  Fourier convention in place of the `from_psd` passage and the
  `psd(frequencies[, channel])` protocol; the Known limitations bullet "No
  covariance translation yet" is replaced by the domainia requirement.
- CHANGELOG 0.4.0: Added `Grid`, `L1Data.grid`, `TranslatedCovariance`,
  `NoiseModel`, the operator methods; Changed the storage layout, `like`'s
  grid capture, the Wheel's storage rule, `Block.sample`'s parameter type;
  Removed the full form, `form`, the derived-basis cache, `from_psd`,
  `from_variance`, `noise_psd`, `noise_variance`, `reference_data` for
  covariances, `NotImplementedError` from the seam.
- `pyproject.toml` version 0.4.0.
- `docs/wheel.svg` is the user's and is left alone.

## 8. Non-goals

- Moving the data transforms (`_transforms.py`, the `wdm` backend) to domainia.
- Band-limited or partial-grid covariances.
- A WDM window parameter on `WDMGrid`.
- Non-stationary or evolutionary noise models.
- Caching Cholesky factors between validation and `log_determinant`.

Each is a separate spec.
