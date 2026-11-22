# Changelog

All notable changes to enchilada are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/) once tagged.

## [0.4.0] — Unreleased

### Added

- `Grid`: a frozen value (`data_domain`, `num_time_samples`, `sample_rate_hz`,
  `wdm_grid`) with `channel_shape`, `observation_duration_s` and
  `same_record`; `L1Data.grid` returns the data's grid and `enchilada.domains`
  owns `DOMAINS`.
- `enchilada.mojito.load_mojito_l1`: reads a DDPC Mojito L1 HDF5 file (TDI
  2.0 `X2`…`T2` stored in Hz) into a time-domain `L1Data` in fractional
  frequency on a chosen window, with the file's `tdis/sampling` clock in
  `start_time_gps` (TCB seconds after the LISA epoch) and its orbit table
  attached; a window with nonzero `tdi_flags` is refused.
- `enchilada.resample.decimate`: anti-aliased rate reduction of time-domain
  `L1Data` by an integer factor (Kaiser FIR, 120 dB, passband to 0.8 of the
  new Nyquist, zero-phase) keeping `start_time_gps`; `edge_samples` gives the
  transient length. Both functions are exported from `enchilada`.
- `examples/mojito_wdm_wheel.py`: fits a Mojito window to a WDM grid
  (`fit_window`), loads and decimates it, translates it to WDM and serves a
  diagnostic block on that grid; the Wheel side of the Mojito story.
- Optional extra `mojito` (h5py) for loading data without lisaorbits.
- `Block.initial_sample(conditional_residual, covariance, *, rng)`: every
  block draws or constructs its own start. `Wheel.add` calls it once with the
  observations minus the starts of the blocks registered so far, on the
  block's grid, the covariance its role receives and the campaign RNG, and
  adopts the result through the same validation as a `sample` return; a block
  without it is refused with `TypeError`. `EchoBlock` publishes an empty start.
- `Block.role`: every block declares `role = "signal"` or `role = "noise"` as
  an attribute, read once at `Wheel.add`; a block without one is refused with
  `TypeError`, another value with `ValueError`. Wheel sums the noise blocks'
  covariances the way it sums the signal blocks' templates: a signal block
  receives the sum over every noise block, a noise block the sum over the
  other noise blocks (`None` when it is alone) with its own last publication
  on `current_block_state.covariance`, and `wheel.covariance` is the whole
  sum. A signal block that returns a covariance, or a noise block that
  returns a template or omits its covariance, is refused with `ValueError`
  naming the block, the method and the role; the campaign is left unchanged.
- `FixedNoiseBlock(name, covariance)`: a noise block that publishes one
  configured covariance on every call; the way to give a campaign a known
  noise model.
- `Covariance.__add__`: `noise_a + noise_b` adds two covariances on one grid
  with one channel order, validated like any construction; mismatches raise
  `ValueError`.
- `check_block(block_under_test, observed_data, num_cycles=2, *,
  noise_blocks=(), random_seed=0, num_frequency_divisions=None,
  num_time_divisions=None)`: `noise_blocks` are registered first (each must
  declare `role = "noise"`); the division counts set the scratch Wheel's home
  grid.
- `Wheel(observed_data, *, num_frequency_divisions=None,
  num_time_divisions=None, random_seed=None)`: the campaign lives on a WDM
  *home grid* chosen at construction (`wheel.home_grid`); time or frequency
  observations are transformed onto it once, WDM observations bring their grid.
  Templates, the residual and both sums live there; `add(block, *,
  data_domain=None, ...)` serves each block on the grid it asks for by exact
  data transforms (`None` is the observations' own grid, `"wdm"` without counts
  the home grid), and a returned template is validated on the block's grid and
  transformed onto the home grid. `observed_data` is the untouched arrival copy;
  `working_residual`, `residual()` and `contribution()` come back on the
  observations' own grid through one exact inverse transform.
- `transform(covariance, target_domain, *, num_frequency_divisions=None,
  num_time_divisions=None)`: translates a diagonal covariance between time,
  frequency and WDM grids through a piecewise-constant spectral model (white
  noise round-trips exactly; a stationary PSD is band-averaged into WDM with
  domania's window, equal to `domania.wdm_covariance`). The result is
  `float64` on every grid, frequency included: a complex cross term keeps its
  real part. Wheel uses it for the noise sum: a noise block's publication is
  kept natively in the ledger, translated onto the home grid for the sum, and
  every block receives the sum translated onto its own grid.
- `examples/noise_floor.py`: a known floor (`FixedNoiseBlock`) plus a sampled
  excess variance as two noise blocks, with `tests/test_noise_floor_example.py`.
- `forge/`: a workbench for building blocks against simulated Mojito-layout
  L1 data. `forge/data.py` loads a decimated window onto the time, frequency
  or WDM grid (`load_window`) and builds a `Covariance` from an analytic PSD
  (`white_psd`, `analytic_covariance`); `forge/block.py` is a skeleton
  `ForgeBlock` (a frequency-grid sinusoid with a conjugate sampler) whose
  `# EDIT` methods the author replaces; `block.ipynb` and `wheel.ipynb`
  exercise it by hand, through `check_block` and in a short campaign;
  `tests/test_forge.py` runs all of it on a synthetic Mojito-layout file.

### Changed

- **Breaking:** the core depends on `numpy>=2.1`, `scipy>=1.14.1` and
  `domania>=3.0,<4`; every WDM conversion goes through domania. The WDM
  coefficient layout is domania's folded `(num_frequency_divisions,
  num_time_divisions)`, in which every cell carries data (row 0 interleaves the
  DC and Nyquist bands); `WDMGrid.active_mask` and the inactive-zero rules on
  coefficients and covariances are removed. The private `wdm` backend, its
  Python >=3.13 requirement and the `wdm.yml` workflow are gone. scipy moves
  from the `numeric-orbits` extra into the core, so `decimate` and
  `NumericOrbit`'s interpolation need no extra; the `numeric-orbits` and
  `mojito` extras both require `h5py>=3.12.1`. WDM arrays in the old
  `(Nf + 1, Nt)` layout are refused (`num_frequency_divisions must be a
  positive even integer, got <Nf + 1>`); regenerate them with `transform` from
  time or frequency data.
- `Covariance` stores one stacked `(K, K, *channel_shape)` Hermitian array,
  `blocks`, and carries its `grid`. `Covariance.like(data, pairs)` is the one
  classmethod and captures the data's grid and channel order; `pair`, the
  channel attributes and `in_basis` read it. Construction validates exact
  Hermitian symmetry, realness per domain, positive variances and positive
  definiteness with one batched Cholesky.
  `copy()` revalidates; `deepcopy` and pickle restore without validating.
- **Breaking:** the direct constructor is
  `Covariance(blocks, *, channel_names, grid)`; the `pairs` attribute and
  `form` are gone.
- **Breaking:** the keyword `reference_data` is renamed `data` on `like` and
  `check_compatible`.
- **Breaking:** `check_compatible` checks only channel order and grid equality;
  its shape, real/complex kind and positivity checks moved into the
  constructor.
- **Breaking:** `Wheel.add(initial_block_state=...)` and
  `check_block(initial_block_state=...)` are gone; a chosen start (an
  injection, a checkpoint) is block configuration rendered by
  `initial_sample`, and the example prior helpers (`draw_sine_prior`,
  `draw_white_noise_prior`, `draw_gb_prior`) became the blocks'
  `initial_sample`. Registration now consumes campaign RNG draws when the
  start draws from `rng`. Under `check_block`, construct the block with its
  start.
- **Breaking:** every block must declare `role` (`"signal"` or `"noise"`) as
  a class attribute; `Wheel.add` and `check_block` read it. The single
  covariance owner is gone: every noise block adds its covariance to the sum,
  published on its own grid and translated onto the home grid, and the sum
  reaches each block translated onto its grid; a covariance a noise block
  returns on another grid than its own is refused at `add` or `run`. A joint
  signal-and-noise model becomes two blocks, one per role.
- **Breaking:** `Wheel` stores the campaign on a WDM home grid. Time or
  frequency observations need `num_frequency_divisions` or
  `num_time_divisions` at construction (`ValueError` otherwise) and the record
  length must factor into two even counts; `wheel.covariance` is on the home
  grid; ledger templates are on the home grid and noise publications on their
  blocks' grids; the residual views equal the arrival-grid values to rounding,
  not bit for bit.
- The tutorial's `WhiteNoiseVarianceBlock` and `examples/toy_fit.py`'s
  `WhiteNoiseBlock` model the whole noise and refuse to run beside another
  noise block.
- `benchmarks/benchmark_wheel.py` runs one `FixedNoiseBlock` beside the fixed
  signals, reports `"noise_blocks": 1` and `"rows"`, and takes `--rows` for
  the home grid (default 128). Every noise publication, an unchanged one
  included, is revalidated each cycle, and its time-grid blocks pay three data
  transforms per call and one covariance translation per block per cycle, so
  its timings are not comparable with runs from before the noise sum or the
  home grid.
- The 0.3.0 covariance-translation seam (`NotImplementedError` from
  `transform` of a `Covariance`, and from a Wheel handing a covariance to a
  block on another grid) now translates through the diagonal spectral model;
  `reference_data` is accepted and ignored for a `Covariance`, and a
  `BlockState` carries its covariance unchanged. No `NotImplementedError`
  remains in the package.
- The examples and notebooks build covariances with `like` from the convention
  `C = (Tobs / 2) S(f)` under `X = dt * rfft(x)` and read them with `pair`.
- `L1Data` now rejects a bool or non-real `sample_rate_hz` with `TypeError` at
  construction, matching `Grid`.

### Removed

- The full form, `form`, the module constant `FORMS`, the derived-basis cache
  and `__getstate__`/`__setstate__` (replaced by `__reduce__`).
- `Covariance.from_psd`, `from_variance`, `noise_psd`, `noise_variance` and the
  `psd(frequencies[, channel])` model protocol: a noise block computes its
  variances on the grid of the data it received and passes them to `like`.
  Migration:
  - `from_variance(data, sigma**2)` → `like` with `sigma**2` per time sample,
    `N * dt**2 * sigma**2` per frequency bin or `dt * sigma**2` per WDM cell.
  - `from_psd(data, model)` → `like` with `data.Tobs / 2 * model.psd(f)` per
    channel at `f = np.fft.rfftfreq(data.N, data.dt)`.
  - `noise_psd(ch, data)` → `2 * noise.pair(ch, ch) / data.Tobs`.
  - `noise_variance(ch, data)` → `noise.pair(ch, ch)` on time data.
- `Wheel(initial_covariance=...)` and the positional covariance argument: a
  known noise model is a `FixedNoiseBlock` (a block with `role = "noise"`).
- `NoiseOverwrittenWarning` and the "owner must republish" rule: noise blocks
  add, and each republishes into its own slot of the sum.
- `L1Data.zero_block_state(covariance=...)`: a zero template plus a covariance
  is refused for either role; a noise block returns `BlockState(covariance=...)`.
- `check_block(initial_covariance=...)`: pass
  `noise_blocks=(FixedNoiseBlock(name, covariance),)` instead.

## [0.3.0] — Unreleased

### Added

- `Wheel.add(initial_block_state=...)` starts a block from an explicit injection
  or other complete physical state without consuming campaign RNG draws. It uses the
  usual validation, copied state, domain translation, and covariance ownership.
  `check_block` accepts the same required starting state, demonstrated in
  `examples/create_block.ipynb`.
- `transform` converts observations and block states between time, frequency,
  and WDM representations; covariance translation is a seam that raises
  `NotImplementedError` until a translation layer is added. `WDMGrid` validates
  dependent division counts; the optional local WDM backend is loaded lazily.
- Per-block representation selection in `Wheel.add` translates the residual and
  current signal together and adopts returned outputs on the canonical
  observation grid. A runnable translation example checks both grid selectors
  against the real optional backend.
- `Covariance` stores noise covariance as one array per channel pair and is
  read through channel attributes: `cov.X`, `cov.XY`, `cov.A`, `cov.AE`, with
  reversed pairs as conjugate transposes and absent cross terms as zero. With
  channels exactly X, Y, Z the A, E, T basis is derived on demand through the
  orthonormal LDC combination, and the reverse; `in_basis` returns the whole
  derived object. Each array has the data's channel shape S
  (`form="diagonal"`) or S + S (`form="full"`); full-form arrays are checked
  Hermitian per channel, with tolerance scaled to each pair of points'
  variances, and stored exactly Hermitian. `like`, `from_psd`, `from_variance`,
  `noise_psd`, `noise_variance`, and `check_compatible` take the `L1Data` the
  covariance accompanies; the object carries no grid.
- `Ledger` provides defensive snapshots of each block's complete `BlockState`:
  optional `template` and `covariance`, and physical `parameters`.
  `ledger.snapshot()` returns an independent dictionary for
  inspection; the Ledger itself compares by identity.
- `examples/requirements-gb.txt` installs the external galactic-binary example
  stack separately from core dependencies. Its preflight checks and documented
  Python/platform requirements make the optional example easier to run.

### Changed

- Rename the block container to `BlockState` in `enchilada.block_state`, its
  factories to `block_state` and `zero_block_state`, and its arguments to
  `initial_block_state` and `current_block_state`, with no compatibility aliases.
- `BlockState` publishes only `template`, `covariance`, and `parameters`, with
  `with_covariance` for a replacement covariance. Sampler continuation and
  diagnostics move to each block's own typed `State`; Wheel owns accepted
  physical estimates and its RNG, and cannot roll back mutations to block-local
  state. Independent campaigns require separate block instances.
- `L1Data.wdm_grid` is automatically derived from WDM coefficient shapes and is
  no longer a constructor argument. This read-only metadata is `None` for time
  and frequency data. Select WDM divisions through `transform` or `Wheel.add`.
- `L1Data` infers channel names from dictionary insertion order when omitted or
  None. Explicit ordering remains supported for covariance alignment. Only
  `channel_data` may be positional; all observation settings are keyword-only.
- Change the project license from MIT to Apache License 2.0.
- The core NumPy floor is 1.26.4, tested independently of the numeric-orbit
  dependencies on Python 3.12. CI and release gates also run optional WDM
  integration against a pinned backend revision.
- Blocks require only `name` and
  `sample(conditional_residual, covariance, current_block_state, *, rng)`.
  Both `Wheel.add` and `check_block` require a complete, non-None
  `initial_block_state`; initialization is explicit campaign setup, with no
  prior-drawing hook in the protocol. Instances hold configuration and their own
  sampler state; Wheel owns accepted physical estimates.
- Wheel maintains independent `observed_data` and `working_residual`, the current
  `covariance`, a ledger, and its RNG. Each block receives the observations
  minus all other blocks' accepted signals. Published state, covariance, residual,
  and RNG adoption is atomic; a failed call leaves that block's accepted estimate
  intact, but does not restore its local sampler state.
  A persistent balanced tree caches signal sums for logarithmic block replacement
  and exclusion.
- Every block state is a complete physical snapshot. A missing template means zero and
  removes any previous contribution. Noise-only blocks publish covariance directly.
  The current covariance owner must always return its full covariance; another
  block taking ownership emits `NoiseOverwrittenWarning`.
- Covariance is explicit: pass `initial_covariance` to Wheel or publish it from
  a block; `None` starts a campaign without one. `Covariance.from_psd(spectrum,
  model)` needs frequency-domain data and fills every bin including DC with
  `(Tobs / 2) S(f)`; `from_variance(data, variance)` needs time-domain data.
  `L1Data.block_state`/`zero_block_state` take `covariance=`.
- `L1Data` carries channel arrays and observation conventions, with descriptive
  field names and units. Scientific aliases `fs`, `dt`, `N`, `Tobs`, `df`, `fny`,
  and `t0` are supported notation. Orbit interfaces name GPS times, positions in
  metres, nominal arm length, and transfer frequency explicitly.
- `on_cycle_complete(cycle_index, wheel)` runs after a full cycle accepts, with a
  zero-based index local to each `run`. It does not run for a partially failed cycle.
- Examples prepare injected values or prior draws in setup, with block-owned
  continuation state and callback history collection. EchoBlock records calls in
  `state.num_sample_calls` and diagnostics in `state.metadata`; the GB example
  keeps Eryn continuation state on the block and refreshes conditional likelihoods.
- Release builds test the packaged source before publication and use a pinned
  PEP 517 backend. Dependency-floor tests preserve their resolved environment;
  notebook setup installs project extras before external model dependencies.

### Fixed

- `L1Data.to_time()` and `to_frequency()` share `transform()` validation and
  return independent arrays, including same-domain calls. Edited arrays with
  invalid lengths or Fourier endpoints cannot bypass validation.
- Fourier normalization widens complex64 input before scaling, preserving
  representable results that would otherwise underflow or overflow.
- `Block.name` is a read-only protocol property, so frozen dataclasses satisfy
  the same public typing contract as mutable configurations, including explicit
  protocol subclasses.
- Manual release dispatches cannot publish, even when run against a tag.
- Source archives include the examples and workflows linked from the README;
  packaged documentation links are checked after extraction.
- The optional GB example constrains NumPy below 2.4 because released Eryn 1.2.6
  calls the removed `numpy.in1d` function. This upper bound applies only to the
  example dependencies.
- NumericOrbit copies its input arrays and rejects nonfinite query times, keeping
  caller mutations and invalid times from changing its interpolation bounds.
- Observation and block-state spectra require real DC coefficients and real
  Nyquist coefficients for even time-series lengths.
- Residuals subtract aggregated signals from pristine observations, preserving
  the data when large block contributions cancel. Signal sums use at least
  float64 or complex128 precision.

### Removed

- Noise storage and weighting helpers on observation data, the `DataCovariance`
  stacked point × channel × channel matrix with its `active_mask` and inactive
  DC bin, the `TranslatedCovariance` operator, and the `apply`, `solve`,
  `project`, `quadratic_form`, `log_determinant`, and `degrees_of_freedom`
  methods. Covariance is one `Covariance` object managed by Wheel.
- Separate noise-block and stateful-block contracts; signal, noise, and joint
  models use the same `Block` protocol.
- Heuristic signal-withdrawal warnings. An omitted or explicit zero signal removes
  the block's previous contribution.

## [0.1.0] — 2026-07-29

First working release of the blocked-Gibbs orchestration layer.

### Added
- `Residuals`: the frozen cross-group data contract — TDI arrays plus run
  settings, with a required `observable` field (`domain` defaults to `"time"`,
  `epoch` to `0.0`), long/short name aliases (`Tobs`, `fs`, `dt`, ...), a typo
  catcher, and full self-validation on every construction (tdi/channels
  consistency, per-domain array lengths, orbit-must-span-data).
  `n_samples` is derived from the tdi arrays for time-domain data (where they
  carry it exactly) and required only for frequency-domain data, where the
  rfft grid loses the parity of n — 513 bins are consistent with n=1024 and
  n=1025, which imply different `Tobs`/`df`, so it is asked for rather than
  guessed. This moved `n_samples` after the required fields, so construct
  `Residuals` by keyword (positional construction changed shape).
  `noise_variance()` gives the per-sample variance a time-domain likelihood
  needs (PSD integrated over the grid, Nyquist half-weighted, so it does not
  depend on the parity of `n_samples`). `dtype` is part of the validated
  contract (floating or complex; integer and object arrays are refused at
  construction rather than failing inside the Wheel), `channels` is normalised
  to a tuple, and equality is identity-based so comparing two `Residuals` no
  longer raises a numpy error.
  `to_frequency()` / `to_time()` transform a dataset between representations,
  carrying `n_samples` (so the round trip is exact for either parity) and
  applying the campaign's `X(f) = dt * rfft(x)` convention in code rather than
  in prose -- the same convention `noise_psd` is normalized against, which a
  test now pins. Since data enters as a time series and transforms from there,
  `n_samples` never needs stating by hand in the normal workflow.
- Vocabulary, settled before the first release so none of it needs a
  deprecation period. Three nested scales get three words, and no word does
  double duty:
  a **cycle** is one pass over every block (`Wheel.run(n_cycles=...)`,
  `on_cycle=...`, `check_block(n_cycles=...)`);
  a **block update** is one block's turn within that pass
  (`Block.update(residual)`);
  a **step** is what a block's own sampler does, many times, inside a single
  `update()` call (`steps_per_cycle` in the GB example).
  The unit itself is a **block** (`Block`, `NoiseBlock`, `EchoBlock`,
  `check_block`, `enchilada.block`) -- the word blocked Gibbs already uses for
  a jointly-updated group of parameters. "Iteration" and "sweep" are not used
  as API names: the first could mean any of the three scales, and the second
  invites confusion with a block's own sampler steps. Note for anyone reading
  GLASS alongside this: GLASS's `cycle` means repeat updates of a single
  module, which is not what enchilada calls a cycle.
- `Block` protocol — two methods, `start(residual)` and `update(residual)`,
  each returning the updated residual. The residual handed to a block is
  the data minus every *other* block (its own model excluded), so the
  block fits it directly and subtracts its new model — there is no
  add-back to forget. Everything else — parameters, RNG, chains, checkpoints,
  and the block's own current model — is block-internal state the Wheel
  never sees. Noise is not special: a noise block returns the residual with
  an updated `noise` object (a zero ledger entry), consumed via
  `Residuals.noise_psd` (documented interface `psd(freqs[, channel])`).
- `NoiseOverwrittenWarning`: `Residuals.noise` is a single slot, so when a
  second block writes it the first block's model is simply gone -- the Wheel
  does not combine noise models. That was documented but undetected; it now
  warns, naming both blocks and the fix (publish one combined model from a
  single noise block, or make the foreground a signal block where the ledger
  *does* combine). It stays a warning rather than an error because handing
  ownership between blocks may be deliberate, and it deliberately stays quiet
  for the two legitimate cases: one noise block re-estimating every cycle, and
  a noise block taking over the model the dataset arrived with.
- `Wheel` boundary guards: `add` verifies `name`/`start`/`update` before
  registering anything; a returned residual may not drop the noise model
  (symmetric with the existing orbit check) and may not contain NaN/inf, which
  would otherwise be recorded as that block's model and handed to every
  block updated later; and because the ledger is derived, the Wheel warns
  when a previously non-zero model becomes exactly zero (a block that returns
  the residual unchanged silently withdraws itself from the fit).
- `Wheel`: the Gibbs ring, owning the pristine data and a per-block ledger
  (each block's current model). It hands each block the data minus every
  other model and derives that block's new ledger entry from what it
  returns, so the residual bookkeeping — and the add-back — lives in the
  framework, not the block (the split the GLASS global fit uses). Atomic
  registration, per-update validation that the returned residual kept the fixed
  run settings, `residual(exclude=...)` and `contribution(name)` accessors,
  and an `on_cycle` callback on `run`.
- `NumericOrbit`: tabulated ephemerides with cubic-spline interpolation,
  loaders for LDC/Mojito HDF5 files and lisaorbits objects
  (validated against lisaorbits 3.0.3), equatorial-to-ecliptic frame
  rotation, and a hard refusal to extrapolate outside the tabulated span.
- `enchilada.testing`: `EchoBlock` and the `check_block` conformance
  helper for third-party block implementations.
- Examples: `examples/demo.py` (plumbing walkthrough), `examples/demo.ipynb`
  (annotated notebook incl. orbits), `examples/toy_fit.py` (a converging
  two-source + sampled-noise Gibbs fit), and
  `examples/gb_block_eryn.ipynb` + `examples/gb_model.py` -- a real LISA
  source class (GBGPU waveforms, an Eryn sampler inside the block, a fixed
  LISA Analysis Tools PSD) recovering an injected galactic binary through the
  Wheel. That one needs an external stack (`gbgpu`, `eryn`,
  `lisaanalysistools`, `matplotlib`, `corner`) that is deliberately not a
  enchilada dependency, so it is not exercised by CI.
- Ergonomics: `enchilada.replace` re-exports `dataclasses.replace`, so updating
  a residual needs no second import; `Block` is `runtime_checkable`, so
  `isinstance(obj, Block)` is a usable registration check. (`NoiseBlock`
  deliberately is *not*: it declares no member beyond `Block`, so a
  runtime check against it would return True for every block.)
- `ModelWithdrawnWarning`: the withdrawal heuristic raises a named, filterable
  category rather than a bare `RuntimeWarning`, since a legitimate RJMCMC death
  move is indistinguishable from a forgotten re-subtraction from outside the
  block. Filter it if your sampler does death moves; `check_block`
  escalates it to an error, because a conformance check is exactly where the
  strict reading belongs.
- Packaging and release: installable via uv or pip (uv_build backend),
  numpy-only core, `numeric-orbits` and `examples` extras, MIT license, PEP 561
  `py.typed` marker (with `check_untyped_defs`, so unannotated bodies are
  checked too -- consumers' type checkers trust these annotations). CI runs
  ruff (lint + format), mypy, and the suite behind a 95% coverage gate on
  Python 3.12/3.13 across Linux and macOS, plus a numpy-only leg through 3.14,
  a dependency-floors leg, an installed-wheel smoke test, and an sdist leg that
  unpacks the archive and runs the packaged suite from it -- so "the sdist is
  self-testing" is verified rather than asserted. Tagging `v*` re-runs the
  whole gate, checks the tag against the project version as parsed versions,
  and publishes through PyPI Trusted Publishing (no token exists to leak).
