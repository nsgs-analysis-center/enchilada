# A WDM home grid, covariance translation, and block-declared roles

**Date:** 2026-11-21
**Status:** approved design, awaiting a plan
**Builds on:** `docs/superpowers/specs/2026-11-19-noise-sum-design.md` (PR #12, roles and the noise sum)

## Summary

Blocks may live on whatever grid they need. The campaign stores all of its state
on one WDM grid, the *home grid*, chosen at `Wheel` construction. Observations,
residual, signal sum and noise sum live there. A block on another grid receives
the residual by an exact data transform and the noise covariance by a
translation rule that treats every covariance as diagonal in the grid index on
every grid. That diagonal assumption is a modelling approximation that the user
owns and assesses; Wheel applies it uniformly and documents what it smooths
away.

Each block also declares its own role, `"signal"` or `"noise"`, as an attribute
instead of the `role=` keyword at `Wheel.add`.

The one-grid rule from the noise-sum design (the first noise publication fixes
the covariance grid; no covariance leaves it) is gone, together with its
refusals.

## User decisions (already made)

- The covariance is assumed diagonal in the grid index on every grid. The user
  assesses the approximation; Wheel does not restrict grid choices on its
  account.
- The campaign stores everything on one grid in the WDM domain.
- The home grid is set by division counts at construction; observations already
  on a WDM grid use theirs; time or frequency observations without divisions
  are refused.
- Translation reads every covariance as a piecewise-constant spectral model
  S(f, t) ("spectral model as currency"), with domania's band average into WDM.
- A block declares its role; `add` and `check_block` lose the `role` keyword.

## 1. The home grid

```python
Wheel(observed_data, *, num_frequency_divisions=None, num_time_divisions=None,
      random_seed=None)
```

- Time or frequency observations: the divisions go to `resolve_wdm_grid`
  (which derives a missing count and checks even counts with
  `Nf * Nt == num_time_samples`); Wheel transforms the observations onto that
  grid once, exactly. Neither count given: `ValueError` naming
  `num_frequency_divisions` and `num_time_divisions`.
- WDM observations: their grid is the home grid. Divisions that disagree with
  it: `ValueError` naming both grids. Equal divisions are accepted.
- The home grid is fixed for the campaign's life and exposed as
  `wheel.home_grid` (a `Grid` with `data_domain == "wdm"`).

Wheel keeps two copies of the observations: the arrival-grid original (for
`observed_data`) and the home-grid copy (the pristine values every residual is
built from).

### What lives where

| State | Grid |
|---|---|
| working residual, signal sum, noise sum | home grid |
| ledger: a signal block's template | home grid |
| ledger: a noise block's covariance | the block's own grid (native) |
| `observed_data`, `working_residual`, `residual(...)`, `contribution(...)` | arrival grid, by exact inverse transform |
| `covariance` | home grid (deep copy), or `None` |

The signal side keeps today's mechanics with the home grid in place of the
observation grid: a template published on a block's grid is transformed onto
the home grid before it enters the signal sum and the ledger; a block's
`current_block_state` is transformed back to its grid at hand-over.

### Block grids

`add(block, *, data_domain=None, num_frequency_divisions=None,
num_time_divisions=None)`. `data_domain=None` means the arrival grid, so a
block written for the data as they came keeps receiving what it received
before. `data_domain="wdm"` with no divisions means the home grid, the fast
path: no data transform and no covariance translation. Any other grid,
including a WDM grid with other divisions, is served by exact data transforms
(domania regrids WDM to WDM through the rfft).

Cost (3 channels, 2^20 samples, measured): about 0.08 s per WDM transform
direction, 0.02 s for an rfft, round-trip error 3e-15. A block off the home
grid pays one transform each way per cycle.

## 2. Translating a covariance

`transform` gains an overload:

```python
transform(covariance: Covariance, target_domain: str, *,
          num_frequency_divisions: int | None = None,
          num_time_divisions: int | None = None) -> Covariance
```

The target grid is resolved as for data (`_target_grid`). The result is a new
`Covariance` through the validating constructor, with the same channel names in
the same order. Same grid in and out returns a copy.

### The rule

Every diagonal covariance is read as a piecewise-constant one-sided spectral
model S(f, t) in the units `Covariance` already uses (`x_n**2` on time,
`(dt * rfft(x))**2` on frequency, domania's cell coefficients on WDM):

| Source grid | Stored blocks | Reading |
|---|---|---|
| time | per-sample σ²(t_j) | S(f, t_j) = 2 dt σ²(t_j), flat in f |
| frequency | per-bin C_k | S(f_k, t) = 2 C_k / (N dt), flat in t |
| WDM | per-cell C_mn | S(f, t_n) = 2 C_mn for f in band m |

Each K × K channel block is carried elementwise, because every transform acts
per channel. The target grid's values are weighted means of the source values
with non-negative weights, so Hermitian symmetry and positive definiteness
survive and the constructor's validation passes for every valid input.

- **Into WDM:** `domania.wdm_covariance(S, Nf=..., Nt=..., dt=...)`, the
  φ̂²-weighted band mean of S at each column (WDM Paper II, Eq. 36), with the
  same Meyer window the data transforms use. Exact for a stationary PSD and for
  time-varying white noise; otherwise the band mean.
- **WDM to frequency:** bin k takes the mean over columns of S at f_k, with the
  window's band weights at f_k normalised over bands. A stationary input that
  went into WDM comes back band-smoothed.
- **WDM to time:** sample j takes the mean over bands of S at t_j, with the
  window's time weights at t_j normalised over columns. A time-varying input
  that went into WDM comes back column-smoothed.
- **Time ↔ frequency:** through the model directly (no WDM step): frequency
  from time is the record's mean power at every bin; time from frequency is
  the mean power at every sample. Both are the diagonal of the exact FFT
  conjugation.
- **WDM to another WDM grid:** through the model, sampled at the target's
  cells by `wdm_covariance`.

White noise round-trips exactly through every pair. The three smoothings
above (band, column, and flattening for time ↔ frequency) are the whole of the
approximation; the README states them in one table.

### Where the weights come from

The Meyer window and the cell layout come from domania (`domania.wilson`), so
the translation and the data transforms share one window. The plan verifies
the two out-of-WDM kernels numerically (section 6) rather than trusting the
derivation.

## 3. Roles declared by the block

`Block` (a runtime-checkable Protocol) gains a member `role: Role`, where
`Role = Literal["signal", "noise"]` and `ROLES = ("signal", "noise")` stay in
`wheel.py` and `Role` stays unexported. A block normally declares it as one
class-level line, `role = "noise"` or `role = "signal"`. `FixedNoiseBlock`
declares `role = "noise"`.

- The declaration is required. A block without a `role` attribute fails the
  existing Block check at `add`; the message says the block declares no role.
  A value outside `ROLES` is refused: `block 'x' declares role 'nosie'; a role
  is 'signal' or 'noise'`.
- Wheel reads the role once at `add` and freezes it in `_roles`. Changing the
  attribute afterwards has no effect.
- Every adoption is checked against the frozen role exactly as today; the
  messages say "declares role 'signal'" where they said "is registered as a
  signal block".

`add(block, *, data_domain=None, num_frequency_divisions=None,
num_time_divisions=None)` and
`check_block(block_under_test, observed_data, num_cycles=2, *,
noise_blocks=(), random_seed=0, num_frequency_divisions=None,
num_time_divisions=None)` lose `role`. `check_block` registers `noise_blocks`
first, each under its own declared role (which must be noise, or it is refused
with the block named), then the block under test, and passes the divisions to
its Wheel.

## 4. The noise sum on the home grid

- A noise block publishes on its own grid, as now; a covariance on another
  grid than the block's registration grid is refused at `add` or `run` with
  the existing message.
- At adoption Wheel validates the publication (`Covariance.copy`), translates
  it onto the home grid with `transform`, and replaces the block's leaf of the
  noise sum with the translated copy. The ledger keeps the native publication.
- Hand-over: a signal block receives the whole sum; a noise block receives the
  sum over the other noise blocks, or `None` if alone. On the home grid that
  is a deep copy. On any other grid it is `transform(sum, block grid)`, a fresh
  validated covariance. Cost off the home grid: one translation plus one
  validated construction, about one revalidation (0.12 s at 3 channels ×
  2^20).
- Atomicity is unchanged: all candidates (ledger, residual, both sums, roles,
  block grids) are computed first and committed together; a refusal leaves
  everything, the RNG included, untouched.
- `_covariance_grid`, "a covariance does not translate between grids" and "a
  covariance does not add across grids" are removed.

## 5. Error handling

| Where | Condition | Response |
|---|---|---|
| `Wheel(...)` | time/frequency observations, no divisions | `ValueError` naming both keywords |
| `Wheel(...)` | WDM observations, divisions disagree | `ValueError` naming both grids |
| `Wheel(...)` | counts `resolve_wdm_grid` rejects | its `ValueError` |
| `add` | no `role` attribute | the Block protocol refusal, saying the block declares no role |
| `add` | role outside `ROLES` | `ValueError` naming block and value |
| `add` / `run` | noise block publishes on another grid than its own | existing `ValueError`, both grids named |
| `add` / `run` | role violation in a publication | existing refusals, reworded to "declares role" |
| adoption | translated covariance fails validation (overflow) | `ValueError` naming the block; full rollback |

`transform(covariance, ...)` raises only for an invalid target (as for data) or
a non-`Covariance` input; a valid covariance always translates.

## 6. Testing

**Translation rule** (`tests/test_covariance_translation.py`):

- White round trips are exact to rounding: time → WDM → time, frequency → WDM
  → frequency, cell-constant WDM → time → WDM and → frequency → WDM, time ↔
  frequency.
- A stationary PSD into WDM equals `domania.wdm_covariance` called directly
  with that PSD (and so the forge's `analytic_covariance`).
- A coloured PSD into WDM, and a time-varying white variance into WDM, match
  the empirical per-cell variances of transformed noise realisations within
  Monte-Carlo tolerance; the same for WDM → frequency and WDM → time of those
  inputs against realisations transformed with domania.
- A 3-channel correlated covariance stays Hermitian and positive definite on
  every target, channel names and order preserved; the K × K structure is
  carried elementwise (a probe with distinct per-pair values).
- WDM to another WDM grid agrees with going through the model by hand.
- Same grid returns an equal, independent copy.

**Wheel** (`tests/test_wheel_home_grid.py`, plus the existing suites updated):

- A time campaign with divisions runs blocks on time, frequency, the home grid
  and a second WDM grid together; every public data view comes back on the
  arrival grid to rounding.
- A signal block off the home grid receives exactly `transform(sum, its grid)`;
  a noise block receives the others' sum translated; a noise block's own
  `current_block_state.covariance` is identical to what it published.
- `wheel.covariance` is on the home grid and is the home-grid sum of the
  translated publications.
- Every refusal in section 5, with its message and the rollback pinned
  (snapshot of ledger, residual, both sums by identity, roles, RNG).
- Roles: the attribute is read at `add`, a later change is ignored, a missing
  or invalid declaration is refused, `check_block` refuses a signal block in
  `noise_blocks`.
- Mutation probes in review: the hand-over translation replaced by the
  home-grid sum; the ledger storing the translated copy instead of the native;
  the role read at every adoption instead of at `add`.

**Examples and docs:** `noise_floor.py`'s posterior is unchanged (a white
floor translates exactly); the toy fit, the tutorials, the domain-translation
example, the GB and Mojito examples, the forge notebooks and the benchmark run
with divisions chosen in each file; `tests/test_docs.py` keeps the README
honest.

## 7. What ships

- `src/enchilada/translation.py`: the overload and the spectral-model helpers.
- `src/enchilada/wheel.py`: home grid, arrival-grid views, hand-over by
  translation, block-declared roles.
- `src/enchilada/block.py` (`role` member and docstring), `fixed_noise.py`,
  `testing.py`.
- Every example, notebook, the forge, the benchmark: divisions at
  construction, `role` on each block class, `role=` keywords removed (258
  sites).
- README: the home grid, the role attribute, the translation rule with its
  smoothing table and the cost numbers; CHANGELOG `[0.4.0]` describes the
  final 0.4.0 form once (the `role=` keyword never shipped); `docs/wheel.svg`
  labels.

## Out of scope / follow-ups

- A helper that reports the size of the approximation for a given covariance
  and grid pair (for example the relative change under a round trip). Useful
  for the assessment the user owns; not part of this change.
- The cost of revalidating the sum on every publication (carried over from
  PR #12).
- Caching a translated sum per grid within a cycle: each hand-over translates
  afresh; measure first.
