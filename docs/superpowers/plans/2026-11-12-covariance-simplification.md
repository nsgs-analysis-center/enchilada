# Covariance Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink `Covariance` to one stacked `(K, K, *S)` Hermitian array that carries its `Grid`, remove the PSD helpers and the covariance translation seam, and have the Wheel hold one native covariance on its owner's grid, refusing a block on another grid with `ValueError`.

**Architecture:** A frozen `Grid` (domain, sample count, sample rate, optional `WDMGrid`) joins `domains.py` and `L1Data.grid` returns it. `Covariance(blocks, *, channel_names, grid)` validates once in the constructor (finite, exactly Hermitian, real where the domain requires, positive variances, zero at inactive WDM pixels, batched Cholesky) and is built from a friendly mapping with `Covariance.like(data, pairs)`, the one classmethod. `transform` no longer takes a `Covariance`; a `BlockState` carries its covariance through unchanged. The Wheel stores the covariance on whatever grid its owner published it, hands a `deepcopy` to blocks registered on that grid, and refuses the rest. The work is staged so every task leaves the suite green: tests and examples migrate off the removed helpers under the old API first (Tasks 2 and 3), then the class is rewritten (Task 4), then the seam is removed and the seven strict xfails become passing tests (Task 5), then docs and version (Task 6).

**Tech Stack:** Python ≥3.12, NumPy only in the core; pytest, ruff, mypy. In this sandbox run tools as `.venv/bin/python -m pytest`, `.venv/bin/python -m ruff`, `.venv/bin/python -m mypy` (`uv run` and `uvx prek` fail). The optional `wdm` backend is **not** installed locally: tests marked `requires_wdm` or using `pytest.importorskip("wdm")` skip here and run only in `.github/workflows/wdm.yml`. Notebooks are linted by ruff (`ruff check`/`ruff format --check` include `*.ipynb`). Temporary files go in `$TMPDIR`. Stage named paths only (never `git add -A`, `.` or bare `-u`); end commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

**Spec:** `docs/superpowers/specs/2026-11-11-covariance-simplification-design.md` (HEAD c68ee1e; section numbers below refer to it).

**Baseline (recorded before Task 1):** `.venv/bin/python -m pytest -q` → `529 passed, 20 skipped, 3 xfailed`; `ruff check src tests examples benchmarks` → `All checks passed!`; `ruff format --check src tests examples benchmarks` → `45 files already formatted`; `mypy` → `Success: no issues found in 13 source files`. The 3 local xfails are `test_initial_block_state.py::test_seed_without_signal_can_sample_in_selected_domain[True]`, `test_toy_translation.py::...[frequency]` and `test_gb_example_contract.py::...[time]`; the four WDM-marked xfails skip locally.

**User decisions (already made):**
- "Covariance looks super complicated. Can we make something smaller and more maintainable without significant feature loss?" — the goal; a convenience one line of `like` away is not a capability.
- Keep every channel feature: cross terms, the positive-definiteness check, the XYZ ↔ AET derivation (no cache, no back-link).
- The object carries its grid (`Grid`); this reverses the 2026-11-10 grid-free decision.
- Storage is one stacked `(K, K, *S)` Hermitian array; the friendly dict stays the input; the full form is removed.
- "Why does enchilada need any information about the psd? This is the job of the noise blocks" → "Drop all four": `from_psd`, `from_variance`, `noise_psd`, `noise_variance` and the `psd(freqs[, channel])` protocol are removed; `Covariance.like` is the only classmethod; `pair(ch, ch)` is the read-back; the §2.6 units go in the README.
- "Core now, components later": no domainia dependency, no operator view, no `_domainia.py`, no new CI lane; spec §2.8, §3 and §3.5 withdrawn. The components layer gets its own spec after domainia 3.0.
- A block on another grid than the covariance is refused with `ValueError` naming both grids (at `run`); a seed or returned covariance on another grid than its block's is refused at `add`/`run`.
- `transform()` loses its `Covariance` overload (`copy()` covers the same-grid copy); `transform(block_state)` carries the covariance unchanged apart from the deep copy.
- The Wheel holds one native covariance on its owner's grid; nothing is translated back.
- Version 0.4.0; `NotImplementedError` disappears from the package and from the coverage `exclude_also`.
- All seven strict xfails become passing tests; the WDM ones keep `requires_wdm`.
- `docs/wheel.svg` is the user's and is left alone.

**Task ordering and cross-task couplings (read before starting any task):**
- Task 1 adds `Grid`; Tasks 2 and 3 use `data.grid` from it.
- Tasks 2 and 3 run against the **old** `Covariance` API (dict `pairs`, `form`): the swaps they make (`white_covariance`, `Covariance.like`, `pair`/`.A` reads) are valid under both APIs. Sites that only exist under one API are explicitly deferred: `.pairs` → `.blocks`, `match="shape"` → `match="grid"` and `_copy_validated` → `deepcopy` all happen in Task 4; `xfail` removals, seam-test deletions and `transform(cov, ...)` removals all happen in Task 5.
- The form guards in `examples/toy_fit.py`, `examples/gb_model.py` and `create_block.ipynb` are deleted in Task 3, so the two tests that assert on them (`test_examples.py::test_sine_rejects_full_form_covariance`, the `full` case of `test_gb_example_contract.py::test_gb_context_requires_independent_channels_and_one_psd_value_per_bin`) are removed in Task 3 too.
- Notebook edits go through the JSON script in Task 3 (edit `source` by cell `id`, never `metadata`, write back with `json.dumps(nb, indent=1, ensure_ascii=False) + "\n"`); the three notebooks round-trip byte for byte under that writer. `create_block.ipynb` cells tagged `definitions` are exec'd by `tests/test_block_tutorial.py::tutorial_definitions`.

---

## File structure

| File | Responsibility after this plan |
|---|---|
| `src/enchilada/domains.py` | `DOMAINS`, `WDMGrid`, `resolve_wdm_grid`, `check_wdm_coefficients`, **`Grid`** (new) |
| `src/enchilada/data.py` | `L1Data`; `DOMAINS` re-exported as a `ClassVar`; **`grid` property** (new) |
| `src/enchilada/covariance.py` | **rewritten**: `AET_FROM_XYZ`, `BASES`, `split_pair_name`, `_validated_channel_names`, `_restore`, `Covariance` (`blocks`, `channel_names`, `grid`; `like`, `pair`, `__getattr__`, `in_basis`, `check_compatible`, `copy`, `__reduce__`) |
| `src/enchilada/translation.py` | `transform` for `L1Data | BlockState` only; a `Covariance` argument raises `TypeError` |
| `src/enchilada/wheel.py` | one native covariance on its owner's grid; `_block_domains: dict[str, Grid]`; cross-grid refusals; `DomainSelection`, `_empty_channel` deleted |
| `src/enchilada/__init__.py` | exports `Grid` |
| `tests/conftest.py` | `white_covariance(data, variance)` (new) |
| `tests/test_grid.py` | `Grid` and `L1Data.grid` (new) |
| `tests/test_covariance.py` | rewritten around the stacked layout (~40 tests) |
| `tests/test_translation.py`, `tests/test_wheel_translation.py`, `tests/test_translation_review.py`, `tests/test_initial_block_state.py`, `tests/test_toy_translation.py`, `tests/test_gb_example_contract.py` | seam tests → refusal tests; xfails → passing tests |
| other `tests/*.py` | migrated off the removed helpers |
| `examples/toy_fit.py`, `examples/gb_model.py`, `benchmarks/benchmark_wheel.py`, the three notebooks | migrated to `like` + `pair`; form guards gone |
| `README.md`, `CHANGELOG.md`, `pyproject.toml`, `.github/workflows/wdm.yml` | docs, 0.4.0, coverage config, CI file list |

---

### Task 1: `Grid` and `L1Data.grid`

**Goal:** Add the frozen `Grid` value type to `enchilada.domains`, move `DOMAINS` there (re-exported by `L1Data.DOMAINS`), expose `L1Data.grid`, export `Grid` from the package, and pin it in the typing tests.

**Files:**
- Modify: `src/enchilada/domains.py` (add `numbers` import, `DOMAINS`, `Grid` after `WDMGrid`)
- Modify: `src/enchilada/data.py:8` (import), `:70` (`DOMAINS` ClassVar), new `grid` property after `nyquist_frequency_hz` (around line 362)
- Modify: `src/enchilada/__init__.py` (import and `__all__`)
- Modify: `tests/test_typing.py` (CONSUMER import + `assert_type(r.grid, Grid)`; `test_public_surface_is_pinned` list)
- Create: `tests/test_grid.py`

**Acceptance Criteria:**
- [ ] `from enchilada import Grid` works; `enchilada.__all__` has `"Grid"` right after `"Covariance"`.
- [ ] `L1Data.DOMAINS is enchilada.domains.DOMAINS` and equals `("time", "frequency", "wdm")`.
- [ ] `Grid("time", 64, 0.5).channel_shape == (64,)`, `Grid("frequency", 63, 1.0).channel_shape == (32,)`, `Grid("frequency", 64, 1.0).channel_shape == (33,)`, `Grid("wdm", 64, 1.0, WDMGrid(num_frequency_divisions=8, num_time_divisions=8)).channel_shape == (9, 8)`; `active_mask` is all-true except on WDM where it is `wdm_grid.active_mask`; `observation_duration_s == N / fs`; `same_record` compares `N` and `fs` only.
- [ ] `num_time_samples` is stored as `int`, `sample_rate_hz` as `float` (`type(...) is int`/`float`), and the validation table in `tests/test_grid.py` raises as listed (`ValueError` for a bad domain, a non-positive/bool/float sample count, a non-positive or non-finite rate, a WDM grid on a non-WDM domain, a WDM grid whose product differs from `N`; `TypeError` for a non-real rate and for a missing or non-`WDMGrid` `wdm_grid` on `"wdm"`).
- [ ] `L1Data.grid` returns the matching `Grid` for time, frequency and WDM data.
- [ ] Full suite green: `pytest -q` adds the new tests to the baseline with `0 failed`, still `3 xfailed`; ruff check/format and mypy clean.

**Verify:** `.venv/bin/python -m pytest -q tests/test_grid.py tests/test_typing.py tests/test_data.py` → all pass; `.venv/bin/python -m pytest -q` → `0 failed, 3 xfailed`; `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy` → clean.

**Steps:**

- [ ] **Step 1: Write the failing tests** — create `tests/test_grid.py`:

```python
"""Grid: the four numbers that fix a channel array's layout, and L1Data.grid."""

from dataclasses import replace

import numpy as np
import pytest

from conftest import make_observed
from enchilada import Grid, L1Data, WDMGrid
from enchilada.domains import DOMAINS


def test_domains_are_defined_once_and_re_exported():
    assert DOMAINS == ("time", "frequency", "wdm")
    assert L1Data.DOMAINS is DOMAINS


class TestMembers:
    def test_time_grid(self):
        grid = Grid("time", 64, 0.5)
        assert grid.channel_shape == (64,)
        assert grid.active_mask.shape == (64,) and grid.active_mask.all()
        assert grid.observation_duration_s == 128.0
        assert grid.wdm_grid is None

    @pytest.mark.parametrize(("n", "bins"), [(63, 32), (64, 33), (1, 1)])
    def test_frequency_grid(self, n, bins):
        grid = Grid("frequency", n, 1.0)
        assert grid.channel_shape == (bins,)
        assert grid.active_mask.shape == (bins,) and grid.active_mask.all()

    def test_wdm_grid(self):
        wdm = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
        grid = Grid("wdm", 64, 1.0, wdm)
        assert grid.channel_shape == (9, 8)
        np.testing.assert_array_equal(grid.active_mask, wdm.active_mask)
        assert not grid.active_mask[0, 1] and grid.active_mask[1, 1]

    def test_counts_and_rates_are_normalised(self):
        grid = Grid("time", np.int64(64), np.float32(0.5))
        assert type(grid.num_time_samples) is int
        assert type(grid.sample_rate_hz) is float
        assert grid == Grid("time", 64, 0.5)

    def test_equality_and_same_record(self):
        time = Grid("time", 64, 0.5)
        frequency = Grid("frequency", 64, 0.5)
        assert time != frequency
        assert time.same_record(frequency)
        assert not time.same_record(Grid("time", 32, 0.5))
        assert not time.same_record(Grid("time", 64, 1.0))
        assert hash(time) == hash(Grid("time", 64, 0.5))


class TestValidation:
    @pytest.mark.parametrize(
        ("args", "error", "match"),
        [
            (("wavelet", 64, 1.0), ValueError, "data_domain"),
            (("time", 0, 1.0), ValueError, "num_time_samples"),
            (("time", -4, 1.0), ValueError, "num_time_samples"),
            (("time", True, 1.0), ValueError, "num_time_samples"),
            (("time", 64.0, 1.0), ValueError, "num_time_samples"),
            (("time", 64, 0.0), ValueError, "sample_rate_hz"),
            (("time", 64, -1.0), ValueError, "sample_rate_hz"),
            (("time", 64, float("inf")), ValueError, "sample_rate_hz"),
            (("time", 64, "1.0"), TypeError, "sample_rate_hz"),
            (("time", 64, True), TypeError, "sample_rate_hz"),
            (("wdm", 64, 1.0), TypeError, "wdm_grid"),
            (("wdm", 64, 1.0, (8, 8)), TypeError, "wdm_grid"),
            (
                ("wdm", 32, 1.0, WDMGrid(num_frequency_divisions=8, num_time_divisions=8)),
                ValueError,
                "num_time_samples",
            ),
            (
                ("time", 64, 1.0, WDMGrid(num_frequency_divisions=8, num_time_divisions=8)),
                ValueError,
                "wdm_grid",
            ),
        ],
    )
    def test_rejects(self, args, error, match):
        with pytest.raises(error, match=match):
            Grid(*args)

    def test_is_frozen(self):
        grid = Grid("time", 64, 1.0)
        with pytest.raises(AttributeError):
            grid.num_time_samples = 32  # type: ignore[misc]


class TestL1DataGrid:
    def test_time_data(self, rng):
        data = make_observed(rng)
        assert data.grid == Grid("time", 64, 0.5)

    def test_frequency_data(self, rng):
        spectrum = make_observed(rng).to_frequency()
        assert spectrum.grid == Grid("frequency", 64, 0.5)
        assert spectrum.grid.channel_shape == spectrum.channel_data["A"].shape

    def test_wdm_data(self, rng):
        wdm = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
        data = replace(
            make_observed(rng),
            data_domain="wdm",
            channel_data={
                ch: np.zeros(wdm.array_shape) for ch in make_observed(rng).channel_names
            },
        )
        assert data.grid == Grid("wdm", 64, 0.5, wdm)
        assert data.grid.wdm_grid is data.wdm_grid
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `.venv/bin/python -m pytest -q tests/test_grid.py`
Expected: collection error `ImportError: cannot import name 'Grid' from 'enchilada'`.

- [ ] **Step 3: Add `DOMAINS` and `Grid` to `src/enchilada/domains.py`**

Replace the import block at the top with:

```python
"""Representation metadata independent of any optional transform backend."""

import numbers
from dataclasses import dataclass

import numpy as np

DOMAINS = ("time", "frequency", "wdm")
"""Valid values for a data domain."""
```

Append after the `WDMGrid` class (before `resolve_wdm_grid`):

```python
@dataclass(frozen=True)
class Grid:
    """The layout of one channel's array: domain, sample count, rate, WDM grid.

    Two grids of one record in different domains compare unequal; use
    :meth:`same_record` for "the same time series". The epoch is not part of a
    grid: a stationary noise model has no epoch and ``transform`` does not
    shift records.
    """

    data_domain: str
    num_time_samples: int
    sample_rate_hz: float
    wdm_grid: WDMGrid | None = None

    def __post_init__(self) -> None:
        if self.data_domain not in DOMAINS:
            raise ValueError(
                f"data_domain must be one of {DOMAINS}, got {self.data_domain!r}"
            )
        count = self.num_time_samples
        if (
            not isinstance(count, (int, np.integer))
            or isinstance(count, bool)
            or count <= 0
        ):
            raise ValueError(
                f"num_time_samples must be a positive integer, got {count!r}"
            )
        object.__setattr__(self, "num_time_samples", int(count))
        rate = self.sample_rate_hz
        if isinstance(rate, bool) or not isinstance(rate, numbers.Real):
            raise TypeError(f"sample_rate_hz must be a real number, got {rate!r}")
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError(
                f"sample_rate_hz must be positive and finite, got {rate!r}"
            )
        object.__setattr__(self, "sample_rate_hz", float(rate))
        if self.data_domain == "wdm":
            if not isinstance(self.wdm_grid, WDMGrid):
                raise TypeError("wdm_grid must be a WDMGrid when data_domain='wdm'")
            if self.wdm_grid.num_time_samples != self.num_time_samples:
                raise ValueError(
                    "wdm_grid division product must equal num_time_samples: "
                    f"{self.wdm_grid.num_time_samples} != {self.num_time_samples}"
                )
        elif self.wdm_grid is not None:
            raise ValueError(
                "wdm_grid applies only to data_domain='wdm', "
                f"got {self.data_domain!r}"
            )

    @property
    def channel_shape(self) -> tuple[int, ...]:
        """Shape of one channel's array on this grid."""
        if self.data_domain == "time":
            return (self.num_time_samples,)
        if self.data_domain == "frequency":
            return (self.num_time_samples // 2 + 1,)
        assert self.wdm_grid is not None
        return self.wdm_grid.array_shape

    @property
    def active_mask(self) -> np.ndarray:
        """Independent boolean mask of the points that carry data."""
        if self.wdm_grid is None:
            return np.ones(self.channel_shape, dtype=bool)
        return self.wdm_grid.active_mask

    @property
    def observation_duration_s(self) -> float:
        """``num_time_samples / sample_rate_hz``."""
        return self.num_time_samples / self.sample_rate_hz

    def same_record(self, other: "Grid") -> bool:
        """True when both grids describe one time series: same count and rate."""
        return (
            self.num_time_samples == other.num_time_samples
            and self.sample_rate_hz == other.sample_rate_hz
        )
```

- [ ] **Step 4: Re-export `DOMAINS` and add `L1Data.grid` in `src/enchilada/data.py`**

Change the import on line 8 to:

```python
from enchilada.domains import DOMAINS, Grid, WDMGrid, check_wdm_coefficients
```

Replace the `DOMAINS` ClassVar (line 70) and its docstring with:

```python
    DOMAINS: ClassVar[tuple[str, ...]] = DOMAINS
    """Valid values for `data_domain`; the same tuple as `enchilada.domains.DOMAINS`."""
```

(Inside a class body the right-hand `DOMAINS` resolves to the module-level name because the class attribute is not bound yet.)

Insert after the `nyquist_frequency_hz` property (after line 361, before the `# ---- conventional short-name aliases` comment):

```python
    @property
    def grid(self) -> Grid:
        """The channel array layout: domain, sample count, rate and WDM grid."""
        return Grid(
            self.data_domain, self.num_time_samples, self.sample_rate_hz, self.wdm_grid
        )
```

- [ ] **Step 5: Export `Grid` from `src/enchilada/__init__.py`**

```python
from enchilada.domains import Grid, WDMGrid
```

and in `__all__` insert `"Grid",` directly after `"Covariance",`.

- [ ] **Step 6: Pin the export in `tests/test_typing.py`**

In `CONSUMER`, change the import to

```python
from enchilada import (
    Block, L1Data, Covariance, BlockState, Wheel, WDMGrid, Grid,
    transform,
)
```

and add after `assert_type(explicit_none.channel_names, tuple[str, ...])`:

```python
assert_type(r.grid, Grid)
```

In `test_public_surface_is_pinned`, insert `"Grid",` after `"Covariance",`.

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m ruff format tests/test_grid.py` (some parametrize rows above exceed 88 columns; the formatter splits them), then `.venv/bin/python -m pytest -q tests/test_grid.py tests/test_typing.py tests/test_data.py`
Expected: all pass (the mypy-backed typing tests take ~20 s).

Run: `.venv/bin/python -m pytest -q`
Expected: `0 failed`, `3 xfailed`, passed count = baseline 529 + the new `test_grid.py` tests.

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks && .venv/bin/python -m mypy`
Expected: `All checks passed!`, `... already formatted`, `Success: no issues found`. If `ruff format --check` flags `tests/test_grid.py`, run `.venv/bin/python -m ruff format tests/test_grid.py` and re-check.

- [ ] **Step 8: Commit**

```bash
git add src/enchilada/domains.py src/enchilada/data.py src/enchilada/__init__.py tests/test_typing.py tests/test_grid.py
git commit -m "Add Grid and L1Data.grid; DOMAINS lives in enchilada.domains

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 2: Migrate the tests off `from_variance`, `from_psd`, `noise_variance`, `noise_psd` (old API still in place)

**Goal:** Add `tests/conftest.py::white_covariance` and replace every test use of the four removed helpers and the PSD test doubles with `white_covariance`, `Covariance.like` and `pair`/`.A` reads, so Task 4 can delete the helpers without touching these files again. The old `Covariance` stays; every swap below is valid under both APIs.

**Files:**
- Modify: `tests/conftest.py` (add `white_covariance`)
- Modify: `tests/test_wheel.py`, `tests/test_block_state.py`, `tests/test_data.py`, `tests/test_state.py`, `tests/test_initial_block_state.py`, `tests/test_translation.py`, `tests/test_wheel_translation.py`, `tests/test_gb_example_contract.py`, `tests/test_translation_review.py`, `tests/test_examples.py`, `tests/test_toy_translation.py`, `tests/test_testing.py`, `tests/test_block_tutorial.py`, `tests/test_typing.py`, `tests/test_block_state_contract.py`
- Not touched: `tests/test_covariance.py` (rewritten in Task 4)

**Acceptance Criteria:**
- [ ] `grep -ln "from_variance\|from_psd\|noise_variance\|noise_psd\|FlatPSD\|UnitPSD\|class White:" tests/*.py` prints only `tests/test_covariance.py`.
- [ ] `white_covariance(data, variance)` returns a `Covariance` on `data`'s grid holding `sigma**2` per time sample, `N * sigma**2 / fs**2` per frequency bin, `sigma**2 / fs` at active WDM pixels and zero elsewhere; `variance` may be a scalar or a channel → scalar mapping.
- [ ] A new `@requires_wdm` test in `tests/test_translation.py` compares the mean squared WDM coefficient of simulated white noise with `white_covariance(wdm, sigma**2).A` (skips locally, runs in `wdm.yml`).
- [ ] Full suite: `0 failed`, `3 xfailed`; the strict xfails still raise `NotImplementedError` (the seam is untouched in this task); ruff check/format clean.

**Verify:** `.venv/bin/python -m pytest -q` → `0 failed, 3 xfailed`; `grep -ln "from_variance\|from_psd\|noise_variance\|noise_psd\|FlatPSD\|UnitPSD\|class White:" tests/*.py` → `tests/test_covariance.py` only; `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks` → clean.

**Steps:**

- [ ] **Step 1: Add `white_covariance` to `tests/conftest.py`**

Replace the imports with

```python
from collections.abc import Mapping

import numpy as np
import pytest

from enchilada import Covariance, L1Data
```

and append at the end of the file:

```python
def white_covariance(data, variance):
    """White noise of `variance` per time sample, on `data`'s grid.

    `variance` is one number for every channel or a channel -> number mapping.
    With X = dt * rfft(x) a stationary one-sided PSD S gives bin variance
    C = (Tobs / 2) S at every rfft bin; white noise has S = 2 sigma^2 / fs, so
    C = N sigma^2 / fs^2. On a WDM grid an active pixel has variance
    sigma^2 / fs and inactive pixels are exactly zero.
    """
    grid = data.grid
    if grid.data_domain == "time":
        scale = 1.0
    elif grid.data_domain == "frequency":
        scale = grid.num_time_samples / grid.sample_rate_hz**2
    else:
        scale = 1.0 / grid.sample_rate_hz
    levels = (
        variance
        if isinstance(variance, Mapping)
        else {ch: variance for ch in data.channel_names}
    )
    return Covariance.like(
        data,
        {
            ch: np.where(grid.active_mask, scale * levels[ch], 0.0)
            for ch in data.channel_names
        },
    )
```

- [ ] **Step 2: `tests/test_wheel.py`**

Change the import `from conftest import const_block_state, make_observed` to `from conftest import const_block_state, make_observed, white_covariance` (keep the `Covariance` import: `isinstance(covariance, Covariance)` and `UncheckedCovariance(Covariance)` still use it). Delete the `class FlatPSD:` block (lines 60–66, including its blank lines). Then apply these exact replacements:

| Old | New |
|---|---|
| `covariance=Covariance.from_variance(observed, level),` | `covariance=white_covariance(observed, level),` |
| `Covariance.from_variance(`<br>`                conditional_residual, float(self.state.updates + 1)`<br>`            )` (in `LevelNoiseBlock.sample`) | `white_covariance(conditional_residual, float(self.state.updates + 1))` |
| `assert wheel.covariance.noise_variance("A", observed) == 1.0` (every occurrence: lines 488, 1055, 1137, 1172, 1219) | `np.testing.assert_array_equal(wheel.covariance.A, 1.0)` |
| `assert wheel.covariance.noise_variance("A", observed) == 3.0` | `np.testing.assert_array_equal(wheel.covariance.A, 3.0)` |
| `assert wheel.ledger["noise"].covariance.noise_variance("A", observed) == 3.0` | `np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 3.0)` |
| `initial = constant_initial_result(`<br>`            observed, covariance.noise_variance("A", observed)`<br>`        )` | `initial = constant_initial_result(observed, float(covariance.A[0]))` |
| `ConstBlock("rec", covariance.noise_variance("A", observed)),` | `ConstBlock("rec", float(covariance.A[0])),` |
| `assert covariance.noise_variance("A", observed) == 1.0` | `np.testing.assert_array_equal(covariance.A, 1.0)` |
| `levels.append(covariance.noise_variance("A", conditional_residual))` | `levels.append(float(covariance.A[0]))` |
| `seen.append(current_block_state.covariance.pairs[("A", "A")])` | `seen.append(current_block_state.covariance.A)` |
| `assert not any(`<br>`                arr.flags.writeable for arr in state.covariance.pairs.values()`<br>`            )` | `assert not state.covariance.A.flags.writeable` |
| `covariance = Covariance.from_variance(observed, 7.0)` | `covariance = white_covariance(observed, 7.0)` |
| `covariance = Covariance.from_psd(obs, FlatPSD(3.0))` | `covariance = white_covariance(obs, 3.0)` |
| `covariance = Covariance.from_variance(mismatched_grid, 1.0)` | `covariance = white_covariance(mismatched_grid, 1.0)` |
| `observed, initial_covariance=Covariance.from_variance(observed, 1.0)` | `observed, initial_covariance=white_covariance(observed, 1.0)` |
| `dict(Covariance.from_variance(observed, 2.0).pairs),` | `dict(white_covariance(observed, 2.0).pairs),` (Task 4 turns `.pairs` into `.blocks`) |
| `Covariance.from_variance(conditional_residual, level)` | `white_covariance(conditional_residual, level)` |
| `).with_covariance(Covariance.from_variance(conditional_residual, 2.0))` | `).with_covariance(white_covariance(conditional_residual, 2.0))` |
| `covariance = Covariance.from_variance(observed, 9.0)` | `covariance = white_covariance(observed, 9.0)` |

Replace the whole test `test_initial_psd_model_must_be_converted_explicitly` (lines 589–600) with:

```python
    def test_initial_covariance_must_be_a_covariance(self, observed):
        with pytest.raises(TypeError, match="initial_covariance.*Covariance"):
            Wheel(observed, initial_covariance=object())
        spectrum = observed.to_frequency()
        covariance = white_covariance(spectrum, 7.0)
        wheel = Wheel(spectrum, initial_covariance=covariance)
        wheel.add(
            EchoBlock("echo"),
            initial_block_state=BlockState(),
        )
        wheel.run(2)
        np.testing.assert_array_equal(wheel.covariance.A, covariance.A)
```

- [ ] **Step 3: `tests/test_block_state.py`**

Imports: `from conftest import make_observed, white_covariance` and `from enchilada import BlockState, Wheel` (`Covariance` becomes unused). Delete the `white_noise` helper (lines 111–123) and replace its two calls: `white_noise(observed, 1)` → `white_covariance(observed, 1)`, `white_noise(conditional_residual, variance)` → `white_covariance(conditional_residual, variance)`. Then (`observed` may be frequency-domain in these tests, so compare against the same builder):

| Old | New |
|---|---|
| `assert actual.covariance.noise_variance("A", observed) == 1` (two sites) | `np.testing.assert_array_equal(actual.covariance.A, white_covariance(observed, 1).A)` |
| `assert actual.covariance.noise_variance("A", observed) == 2` (two sites) | `np.testing.assert_array_equal(actual.covariance.A, white_covariance(observed, 2).A)` |
| `assert result.covariance.noise_variance("A", observed) == 3` | `np.testing.assert_array_equal(result.covariance.A, white_covariance(observed, 3).A)` |
| `assert wheel.covariance.noise_variance("A", observed) == 3` | `np.testing.assert_array_equal(wheel.covariance.A, white_covariance(observed, 3).A)` |
| `assert wheel.covariance.noise_variance("A", observed) == 2` | `np.testing.assert_array_equal(wheel.covariance.A, white_covariance(observed, 2).A)` |
| `covariance = Covariance.from_variance(wrong_grid, 2)` | `covariance = white_covariance(wrong_grid, 2)` |

Leave `error = "must return.*covariance" if failure == "omitted" else "shape"` alone (Task 4 changes `"shape"` to `"grid"`).

- [ ] **Step 4: `tests/test_data.py`**

Import: `from conftest import make_observed, white_covariance`. Replacements:

| Old | New |
|---|---|
| `covariance = Covariance.from_variance(data, {"A": 1.0, "E": 4.0})` (two sites) | `covariance = white_covariance(data, {"A": 1.0, "E": 4.0})` |
| `covariance = Covariance.from_variance(data, {"E": 4.0, "A": 1.0})` | `covariance = white_covariance(data, {"E": 4.0, "A": 1.0})` |
| the three-line `class White:` with its `psd` method (lines 613–615) | delete |
| `covariance = Covariance.from_psd(t.to_frequency(), White())` | `covariance = white_covariance(t.to_frequency(), sigma**2)` |
| `model = Covariance.from_variance(observed, 1.0)` | `model = white_covariance(observed, 1.0)` |

`Covariance` is then unused in this module: change the import to `from enchilada import BlockState, L1Data, Wheel, transform`.

- [ ] **Step 5: `tests/test_state.py`**

Imports: `from conftest import make_observed, white_covariance` and `from enchilada import Wheel`. Replacements: `return result.with_covariance(Covariance.from_variance(wrong_grid, 1))` → `return result.with_covariance(white_covariance(wrong_grid, 1))`; `covariance = Covariance.from_variance(observed, 1)` → `covariance = white_covariance(observed, 1)`; `assert wheel.covariance.noise_variance("A", observed) == 1` → `np.testing.assert_array_equal(wheel.covariance.A, 1)`. The `.pairs[("A", "A")]` lines stay until Task 4.

- [ ] **Step 6: `tests/test_initial_block_state.py`**

Add `from conftest import white_covariance` after the `import pytest` line (own import group, before `from enchilada import (`). Replacements:

| Old | New |
|---|---|
| `covariance=Covariance.from_variance(observed, 0.25),` (two sites) | `covariance=white_covariance(observed, 0.25),` |
| `assert wheel.covariance.noise_variance("A", observed) == 0.25` (two sites) | `np.testing.assert_array_equal(wheel.covariance.A, 0.25)` |
| `covariance=Covariance.from_variance(`<br>`                replace(`<br>`                    observed, channel_data={"A": np.full(32, 5.0)}, num_time_samples=32`<br>`                ),`<br>`                1.0,`<br>`            ),` | same block with `Covariance.from_variance(` → `white_covariance(` |
| `initial = BlockState(covariance=Covariance.from_variance(observed, 1.0))` | `initial = BlockState(covariance=white_covariance(observed, 1.0))` |
| `new_noise = BlockState(covariance=Covariance.from_variance(observed, 2.0))` | `new_noise = BlockState(covariance=white_covariance(observed, 2.0))` |
| `assert wheel.covariance.noise_variance("A", observed) == 1.0` | `np.testing.assert_array_equal(wheel.covariance.A, 1.0)` |
| `assert wheel.covariance.noise_variance("A", observed) == pytest.approx(0.25)` | `np.testing.assert_allclose(wheel.covariance.A, 100.0)` |

Also replace the comment and assertion at lines 285–290 (from `# White noise has PSD 2 sigma^2 / fs = 1.25 at every bin. noise_psd` through the closing `)` of the `assert_allclose`) with:

```python
                # White noise with variance 0.25 per sample on this frequency grid:
                # C = N sigma^2 / fs^2 = 64 * 0.25 / 0.4**2 = 100 at every bin.
                np.testing.assert_allclose(covariance.A, 100.0)
```

The `Covariance` import stays (`Covariance.like(spectrum, ...)` uses it). The `xfail` on `with_noise=True` stays until Task 5; the `.pairs[("A", "A")]` lines and `"covariance": (ValueError, "shape")` stay until Task 4.

- [ ] **Step 7: `tests/test_translation.py`**

Add `from conftest import white_covariance` (own group, before `from enchilada import`). Replacements: `covariance = Covariance.from_variance(data, 2.0)` → `covariance = white_covariance(data, 2.0)`; `result = BlockState(covariance=Covariance.from_variance(data, 1.0))` → `result = BlockState(covariance=white_covariance(data, 1.0))`; the argument `Covariance.from_variance(data, 1.0),` → `white_covariance(data, 1.0),`; `cov = Covariance.from_variance(data, 2.0)` (two sites) → `cov = white_covariance(data, 2.0)`; `unchecked(Covariance.from_variance(data, 2.0))` → `unchecked(white_covariance(data, 2.0))`; `unchecked(Covariance.from_variance(observations(32), 2.0))` → `unchecked(white_covariance(observations(32), 2.0))`. The seam tests themselves stay until Task 5.

Append this statistical check of the WDM constant at the end of the file:

```python
@requires_wdm
def test_white_noise_wdm_coefficient_variance_matches_white_covariance():
    """E[w^2] == sigma^2 / fs at every active pixel for white noise of variance
    sigma^2 per sample: the WDM constant used by white_covariance."""
    fs, n, sigma = 0.2, 1 << 12, 0.7
    rng = np.random.default_rng(3)
    acc = None
    trials = 40
    for _ in range(trials):
        t = L1Data(
            channel_data={"A": rng.normal(0.0, sigma, n)},
            sample_rate_hz=fs,
            channel_names=("A",),
            tdi_generation="2.0",
            physical_observable="fractional_frequency",
        )
        wdm = transform(t, "wdm", num_frequency_divisions=64)
        power = wdm.channel_data["A"] ** 2
        acc = power if acc is None else acc + power
    measured = acc / trials
    predicted = white_covariance(wdm, sigma**2).A
    active = wdm.wdm_grid.active_mask
    np.testing.assert_array_equal(predicted[~active], 0.0)
    interior = active.copy()
    interior[:2] = False  # the edge bands carry the window's half-band
    interior[-2:] = False
    ratio = float(np.mean(measured[interior] / predicted[interior]))
    assert ratio == pytest.approx(1.0, abs=0.05)
```

If this test fails in the `wdm.yml` lane, the WDM constant in spec §2.6 (and therefore `white_covariance` and the README paragraph written in Task 6) is wrong by that ratio: correct the constant, not the test.

- [ ] **Step 8: `tests/test_wheel_translation.py`**

Add `from conftest import white_covariance`. Replacements: `covariance=Covariance.from_variance(`<br>`transform(conditional_residual, "time"), 1.0`<br>`)` → the same with `white_covariance(`; `wheel = Wheel(data, Covariance.from_variance(data, 0.25), random_seed=3)` → `wheel = Wheel(data, white_covariance(data, 0.25), random_seed=3)`; `assert wheel.covariance.noise_variance("A", data) == 0.25` → `np.testing.assert_array_equal(wheel.covariance.A, 0.25)`. Docstring, xfail and seam tests stay until Task 5.

- [ ] **Step 9: `tests/test_gb_example_contract.py`**

Add `from conftest import white_covariance`. In `spectral_context`, delete the `class UnitPSD:` and return a unit-PSD covariance directly (`C = (Tobs / 2) * 1` at every bin):

```python
    return data, Covariance.like(data, {"A": np.full(33, data.Tobs / 2)})
```

`time_noise = Covariance.from_variance(transform(data, "time"), 1.0)` → `time_noise = white_covariance(transform(data, "time"), 1.0)`. In `test_gb_rejects_covariance_on_a_different_grid`, delete the inner `class UnitPSD:` and replace `noise = Covariance.from_psd(transform(shorter, "frequency"), UnitPSD())` with:

```python
    spectrum = transform(shorter, "frequency")
    noise = Covariance.like(
        spectrum, {"A": np.full(spectrum.channel_data["A"].shape, spectrum.Tobs / 2)}
    )
```

Leave `test_gb_context_requires_independent_channels_and_one_psd_value_per_bin` and the xfail for Tasks 3 and 5.

- [ ] **Step 10: `tests/test_translation_review.py`**

Add `from conftest import white_covariance`. Replacements: `Covariance.from_variance(observed, 1.0),` → `white_covariance(observed, 1.0),` (inside the xfailed test; it stays xfailed until Task 5); `wheel = Wheel(observed, Covariance.from_variance(observed, 2.0))` (two sites) → `wheel = Wheel(observed, white_covariance(observed, 2.0))`; `assert wheel.covariance.noise_variance("A", observed) == 2.0` → `np.testing.assert_array_equal(wheel.covariance.A, 2.0)`; `received.append(covariance.noise_variance("A", conditional_residual))` → `received.append(float(covariance.A[0]))`. `Covariance` stays imported (`isinstance(value, Covariance)`).

- [ ] **Step 11: `tests/test_examples.py`**

Add `from conftest import white_covariance` after `import pytest`. Function-local imports: `from enchilada import Covariance, L1Data` in `test_sine_conditional_includes_declared_amplitude_prior` (line 109) → `from enchilada import L1Data`, and `noise = Covariance.from_variance(data, 2.0)` → `noise = white_covariance(data, 2.0)`. In `gb_test_context` (line 184 onward) delete `class UnitPSD:` and make the last line `return data, Covariance.like(data, {"A": np.full(33, data.Tobs / 2)})`. In `test_sine_requires_time_white_noise_on_the_data_grid` drop the local `from enchilada import Covariance`, delete `class UnitPSD:` and write `noise = white_covariance(data.to_frequency(), 1.0)`. The `ConstantPSD` class passed to `inject_gb` is user code for the injector and stays. `test_sine_rejects_full_form_covariance` stays until Task 3.

- [ ] **Step 12: `tests/test_toy_translation.py`**

Replace lines 74–81 (the `restored_covariance` block and its comment) with:

```python
    # Both wheels hold the noise block's publication on the time grid it
    # samples on, whatever grid the observations are stored on.
    assert wheels[1].covariance.grid == wheels[0].covariance.grid
    np.testing.assert_allclose(
        wheels[1].covariance.A, wheels[0].covariance.A, rtol=1e-12
    )
```

The xfail stays until Task 5 (the test still raises `NotImplementedError` at the noise block's `add`).

- [ ] **Step 13: `tests/test_testing.py`**

Imports: `from conftest import make_observed, white_covariance`, `from enchilada import BlockState`. `initial_covariance=Covariance.from_variance(observed, 1.0),` → `initial_covariance=white_covariance(observed, 1.0),`. In `test_conforming_noise_block_passes` delete `class FlatPSD:` (lines 74–77) and the local `from enchilada import Covariance`, and return `conditional_residual.block_state().with_covariance(white_covariance(conditional_residual, 1.0))` (ruff-format it over three lines if it exceeds 88 columns).

- [ ] **Step 14: `tests/test_block_tutorial.py`**

In the two tests with a local `from enchilada import Covariance, L1Data`, use `from conftest import white_covariance` and `from enchilada import L1Data`; `noise = Covariance.from_variance(data, 0.2)` (two sites) → `noise = white_covariance(data, 0.2)`; `changed_noise = Covariance.from_variance(changed_data, 0.4)` → `changed_noise = white_covariance(changed_data, 0.4)`.

- [ ] **Step 15: `tests/test_typing.py` and `tests/test_block_state_contract.py`**

`tests/test_typing.py` CONSUMER: `noise = Covariance.from_variance(r, 1.0)` → `noise = Covariance.like(r, {"A": np.ones(8)})` (the `assert_type(transform(noise, ...))` line stays until Task 5).

`tests/test_block_state_contract.py`: add `from conftest import white_covariance`; import `from enchilada import BlockState, Wheel`; `covariance=Covariance.from_variance(conditional_residual, 2.0),` → `covariance=white_covariance(conditional_residual, 2.0),`; `assert accepted.covariance.noise_variance(name, observed) == 2.0` → `np.testing.assert_array_equal(accepted.covariance.pair(name, name), 2.0)`.

- [ ] **Step 16: Verify and commit**

Run: `grep -ln "from_variance\|from_psd\|noise_variance\|noise_psd\|FlatPSD\|UnitPSD\|class White:" tests/*.py`
Expected: `tests/test_covariance.py` only.

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks`
Expected: clean (fix unused imports ruff reports; `ruff format` any file it flags).

Run: `.venv/bin/python -m pytest -q`
Expected: `0 failed`, `3 xfailed`, skipped = baseline 20 + 1 (the new WDM test).

```bash
git add tests/conftest.py tests/test_wheel.py tests/test_block_state.py tests/test_data.py tests/test_state.py tests/test_initial_block_state.py tests/test_translation.py tests/test_wheel_translation.py tests/test_gb_example_contract.py tests/test_translation_review.py tests/test_examples.py tests/test_toy_translation.py tests/test_testing.py tests/test_block_tutorial.py tests/test_typing.py tests/test_block_state_contract.py
git commit -m "Tests build white noise with like through conftest.white_covariance; PSD doubles removed

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 3: Migrate the examples, benchmark and notebooks (old API still in place)

**Goal:** Replace the removed helpers and the `form` guards in `examples/toy_fit.py`, `examples/gb_model.py`, `benchmarks/benchmark_wheel.py` and the three notebooks with `Covariance.like` and `pair` reads in the §2.6 units, and remove the two tests that asserted on the form guards.

**Files:**
- Modify: `examples/toy_fit.py` (docstring lines 6–9; `SineBlock.sample` lines 56–62 and 82–87; `WhiteNoiseBlock._render` lines 105–110)
- Modify: `examples/gb_model.py` (`FixedLISANoise` docstring lines 19–25; `GBBlock` docstring lines 133–136; `_context` lines 200–203 and 214–216)
- Modify: `benchmarks/benchmark_wheel.py:35`
- Modify: `examples/create_block.ipynb` (cells `create-block-05`, `-06`, `-08`, `-19`, `-20`, `-21`, `-23`, `-24`, `-25`), `examples/domain_translation.ipynb` (cells `translation-03`, `-04`, `-12`), `examples/gb_block_eryn.ipynb` (cells `224c7cd3`, `d5f66fb4`, `ebeb17db`, `7261596e`) via the script in Step 4
- Modify: `tests/test_gb_example_contract.py` (`test_gb_context_requires_independent_channels_and_one_psd_value_per_bin`), `tests/test_examples.py` (delete `test_sine_rejects_full_form_covariance`)

**Acceptance Criteria:**
- [ ] `grep -rn "from_variance\|from_psd\|noise_variance\|noise_psd\|\.form\b\|form=" examples benchmarks` prints nothing (the notebooks included).
- [ ] `examples/toy_fit.py` `SineBlock.sample` weights each channel by the per-sample array `covariance.pair(ch, ch)` and `WhiteNoiseBlock._render` publishes `Covariance.like(conditional_residual, {ch: np.full(N, variance)})`; `examples/gb_model.py` `_context` returns `{ch: 2.0 * covariance.pair(ch, ch) / conditional_residual.Tobs}` (one-sided PSD units `S = 2 C / Tobs`).
- [ ] `git diff --stat` for each notebook touches only the listed cells (the JSON writer round-trips the rest byte for byte); no `metadata` changed.
- [ ] `MPLBACKEND=Agg .venv/bin/python -m pytest -q tests/test_examples.py tests/test_block_tutorial.py tests/test_gb_example_contract.py tests/test_toy_translation.py tests/test_translation_example.py` passes with the two deleted tests gone; full suite `0 failed`, `3 xfailed`; ruff check/format clean including the notebooks.

**Verify:** `grep -rn "from_variance\|from_psd\|noise_variance\|noise_psd\|\.form\b\|form=" examples benchmarks` → nothing; `MPLBACKEND=Agg .venv/bin/python -m pytest -q` → `0 failed, 3 xfailed`; `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks` → clean.

**Steps:**

- [ ] **Step 1: `examples/toy_fit.py`**

Module docstring lines 6–9 become:

```python
posterior chains outside the blocks. These blocks request time-domain inputs;
Wheel also runs them on Fourier or WDM observations with data_domain="time",
because the noise block publishes its covariance on that time grid. Their
likelihood requires time-domain noise with independent channels.
```

In `SineBlock.sample`, replace lines 56–62 (`if covariance is None: ...` through the `form` guard's closing parenthesis) with just:

```python
        if covariance is None:
            raise ValueError("SineBlock requires a noise covariance")
```

(keep the following `try: covariance.check_compatible(...)` block and the cross-term check unchanged; `_basis` already rejects non-time data). Replace the per-channel loop (lines 79–87, from the comment `# The toy uses a single channel ...`) with:

```python
        # One independent noise variance per time sample and channel.
        precision = 1.0 / self.prior_std**2
        weighted_data = 0.0
        for ch in conditional_residual.channel_names:
            weights = basis / covariance.pair(ch, ch)
            precision += float(basis @ weights)
            weighted_data += float(conditional_residual.channel_data[ch] @ weights)
```

In `WhiteNoiseBlock._render` replace the `BlockState(...)` with:

```python
        return BlockState(
            covariance=Covariance.like(
                conditional_residual,
                {
                    ch: np.full(conditional_residual.num_time_samples, variance)
                    for ch in conditional_residual.channel_names
                },
            ),
            parameters={"sigma": float(np.sqrt(variance))},
        )
```

- [ ] **Step 2: `examples/gb_model.py`**

`FixedLISANoise` docstring (lines 19–25) becomes:

```python
    """One-sided LISA PSD from lisatools ``get_sensitivity`` (A == E).

    User code: enchilada defines no PSD protocol. The notebook evaluates
    ``psd`` at every rfft bin including ``f = 0`` and hands
    ``(Tobs / 2) * psd(f)`` per channel to ``Covariance.like``, so
    frequencies are floored at ``minimum_frequency_hz``.
    """
```

In the `GBBlock` docstring replace the last sentence of the registration paragraph (lines 134–136) with: `This narrowband likelihood requires a frequency-domain covariance on the data grid with independent channels; it reads each channel's bin variances and converts them to PSD units with ``2 C / Tobs``.`

In `_context`, delete the four lines

```python
        if covariance.form != "diagonal":
            raise ValueError(
                f"{self.name}: this likelihood needs one PSD value per bin"
            )
```

and replace the final `return _new_gb(), {...}` (lines 214–216) with:

```python
        # The likelihood is written in one-sided PSD units: with X = dt * rfft(x)
        # a bin variance C is S = 2 C / Tobs.
        return _new_gb(), {
            ch: 2.0 * covariance.pair(ch, ch) / conditional_residual.Tobs
            for ch in channels
        }
```

- [ ] **Step 3: `benchmarks/benchmark_wheel.py`**

Line 35: `covariance = Covariance.from_variance(observed, 1.0)` →

```python
    covariance = Covariance.like(
        observed, {name: np.ones(num_samples) for name in observed.channel_names}
    )
```

- [ ] **Step 4: Edit the notebooks by cell id**

Save this as `$TMPDIR/edit_notebooks.py` and run `.venv/bin/python "$TMPDIR/edit_notebooks.py"` from the repository root. It asserts each `old` text occurs exactly once, edits `source` only, and writes back with the byte-identical serializer.

```python
import json
from pathlib import Path


def load(path):
    text = Path(path).read_text()
    nb = json.loads(text)
    assert json.dumps(nb, indent=1, ensure_ascii=False) + "\n" == text, path
    return nb


def save(path, nb):
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def cell(nb, cell_id):
    (found,) = [c for c in nb["cells"] if c["id"] == cell_id]
    return found


def replace(nb, cell_id, old, new):
    c = cell(nb, cell_id)
    text = "".join(c["source"])
    assert text.count(old) == 1, (cell_id, old)
    c["source"] = text.replace(old, new).splitlines(keepends=True)


# ---------------------------------------------------------------- create_block
nb = load("examples/create_block.ipynb")
replace(
    nb,
    "create-block-05",
    "covariance = Covariance.from_variance(\n"
    "    observed, time_sample_variance=noise_standard_deviation**2\n"
    ")\n",
    "covariance = Covariance.like(\n"
    '    observed, {"A": np.full(num_time_samples, noise_standard_deviation**2)}\n'
    ")\n",
)
replace(
    nb,
    "create-block-06",
    "`C⁻¹` is the per-sample precision `1 / covariance.A`: this block requires one\n"
    'independent noise variance per time sample (`form="diagonal"`). The proper prior\n',
    "`C⁻¹` is the per-sample precision `1 / covariance.A`: this block requires one\n"
    "independent noise variance per time sample, which is what a `Covariance` on a\n"
    "time grid holds. The proper prior\n",
)
replace(
    nb,
    "create-block-08",
    '        if covariance.form != "diagonal":\n'
    "            raise ValueError(\n"
    '                "This sampler needs one noise variance per time sample, "\n'
    '                f"got form={covariance.form!r}"\n'
    "            )\n",
    "",
)
replace(
    nb,
    "create-block-19",
    "## 10. The same noise as a PSD on the Fourier grid (covariance translation pending)\n",
    "## 10. The same noise on the Fourier grid\n",
)
replace(
    nb,
    "create-block-19",
    "`data_domain=\"time\"` on a campaign that stores Fourier observations receives the\n"
    "conditional residual, covariance, and its previous signal in time coordinates for\n"
    "each call, and Wheel returns the accepted signal to the original grid for storage.\n",
    "`data_domain=\"time\"` on a campaign that stores Fourier observations receives the\n"
    "conditional residual and its previous signal in time coordinates for each call,\n"
    "together with the covariance when the noise owner published it on that time\n"
    "grid, and Wheel returns the accepted signal to the original grid for storage.\n",
)
replace(
    nb,
    "create-block-19",
    "A covariance cannot cross domains until the covariance translation layer is\n"
    "available, so this tutorial keeps its campaigns in the time domain. The following\n"
    "cell needs no optional backend: it expresses the same white noise as a flat\n"
    "one-sided PSD on the Fourier grid and checks that the covariance reads back that\n"
    "PSD and the time variance.\n",
    "A covariance is read on the grid it was built on and does not cross domains, so\n"
    "this tutorial keeps its campaigns in the time domain. The following cell needs no\n"
    "optional backend: it builds the same white noise on the Fourier grid from the\n"
    "convention `C = (Tobs / 2) S(f)` under `X = dt * rfft(x)` and checks that the two\n"
    "descriptions agree.\n",
)
cell(nb, "create-block-20")["source"] = (
    'spectrum = transform(observed, "frequency")\n'
    "time_variance = noise_standard_deviation**2\n"
    "\n"
    "# White noise of variance sigma^2 per sample has one-sided PSD 2 sigma^2 / fs.\n"
    "# With X = dt * rfft(x), a stationary PSD S gives bin variance C = (Tobs / 2) S\n"
    "# at every rfft bin, f = 0 included.\n"
    "flat_psd = 2 * time_variance / sample_rate_hz\n"
    "spectral_noise = Covariance.like(\n"
    "    spectrum,\n"
    '    {"A": np.full(spectrum.channel_data["A"].shape, spectrum.Tobs / 2 * flat_psd)},\n'
    ")\n"
    "np.testing.assert_allclose(\n"
    "    spectral_noise.A, num_time_samples * time_variance / sample_rate_hz**2\n"
    ")\n"
    "np.testing.assert_allclose(2 * spectral_noise.A / spectrum.Tobs, flat_psd)\n"
    'assert spectral_noise.A.shape == spectrum.channel_data["A"].shape\n'
    'print("PSD construction checks passed")'
).splitlines(keepends=True)
replace(
    nb,
    "create-block-21",
    "block with `data_domain=\"time\"` once covariance translation is available. A block\n",
    "block with `data_domain=\"time\"`, with the noise block publishing on that time\n"
    "grid. A block\n",
)
replace(
    nb,
    "create-block-21",
    "setup and the two division selectors. A translated covariance need not be\n"
    "independent per pixel, so check its `form` and reject models your likelihood\n"
    "cannot use.\n",
    "setup and the two division selectors. A block reads the covariance on the grid\n"
    "it was published on; register it on the grid of the noise it needs.\n",
)
replace(
    nb,
    "create-block-23",
    "            covariance=Covariance.from_variance(conditional_residual, variance),\n",
    "            covariance=Covariance.like(\n"
    "                conditional_residual,\n"
    "                {\n"
    "                    name: np.full(values.shape, variance)\n"
    "                    for name, values in conditional_residual.channel_data.items()\n"
    "                },\n"
    "            ),\n",
)
replace(
    nb,
    "create-block-24",
    '    joint_campaign.covariance.noise_variance("A", observed),\n',
    "    joint_campaign.covariance.A,\n",
)
replace(
    nb,
    "create-block-25",
    "- Read the covariance through its channel attributes (`covariance.A`,\n"
    "  `covariance.AE`) and reject a `form` your likelihood cannot use. Include the\n",
    "- Read the covariance through its channel attributes (`covariance.A`,\n"
    "  `covariance.AE`) on the grid your block registered for, and reject cross\n"
    "  terms your likelihood cannot use. Include the\n",
)
save("examples/create_block.ipynb", nb)

# ---------------------------------------------------------- domain_translation
nb = load("examples/domain_translation.ipynb")
replace(
    nb,
    "translation-03",
    "time sample; `Covariance.from_variance` describes it on the time grid.\n",
    "time sample; `Covariance.like` describes it on the time grid, one variance per\n"
    "sample.\n",
)
replace(
    nb,
    "translation-04",
    "covariance = Covariance.from_variance(\n"
    "    reference_data=observed,\n"
    "    time_sample_variance=noise_standard_deviation**2,\n"
    ")\n",
    "covariance = Covariance.like(\n"
    "    observed,\n"
    "    {\n"
    "        name: np.full(num_time_samples, noise_standard_deviation**2)\n"
    "        for name in channel_names\n"
    "    },\n"
    ")\n",
)
replace(
    nb,
    "translation-12",
    "np.testing.assert_allclose(\n"
    '    covariance.noise_variance("A", observed), noise_standard_deviation**2\n'
    ")\n",
    "np.testing.assert_allclose(covariance.A, noise_standard_deviation**2)\n",
)
save("examples/domain_translation.ipynb", nb)

# ------------------------------------------------------------- gb_block_eryn
nb = load("examples/gb_block_eryn.ipynb")
replace(
    nb,
    "224c7cd3",
    "  to the block as the explicit `covariance` argument and read with\n"
    "  `covariance.noise_psd(channel, data)`.\n",
    "  to the block as the explicit `covariance` argument and read with\n"
    "  `covariance.pair(channel, channel)`, converted back to PSD units with\n"
    "  `2 * C / Tobs`.\n",
)
replace(
    nb,
    "d5f66fb4",
    "`Covariance.from_psd` converts the fixed one-sided PSD into coefficient covariance on that grid.",
    "`Covariance.like` holds `(Tobs / 2) * psd(f)` per channel on that grid: the bin "
    "variance of `dt * rfft(x)` coefficients under the fixed one-sided PSD.",
)
replace(
    nb,
    "ebeb17db",
    "The narrowband likelihood requires a diagonal-form frequency covariance on the data "
    "grid with independent channels, as produced by `Covariance.from_psd`; a covariance "
    "translated from another domain generally has correlations between bins and cannot "
    "be used by this PSD likelihood.",
    "The narrowband likelihood requires a frequency covariance on the data grid with "
    "independent channels; the block converts each channel's bin variances back to PSD "
    "units with `2 * C / Tobs`.",
)
replace(
    nb,
    "7261596e",
    "covariance = Covariance.from_psd(reference_data=observed, noise_psd_model=noise)\n",
    "covariance = Covariance.like(\n"
    "    observed,\n"
    "    {ch: observed.Tobs / 2 * noise.psd(freqs) for ch in observed.channel_names},\n"
    ")\n",
)
save("examples/gb_block_eryn.ipynb", nb)
print("notebooks edited")
```

(`freqs` is `np.fft.rfftfreq(N_time, dt)` from cell `ff30cc08`, the full rfft grid of `observed`; `noise` and `observed` come from cell `9f8578b1`.) If a `replace` assertion fails, the cell text differs from what is quoted here: print `"".join(cell(nb, id)["source"])` and adjust `old` to the exact text, keeping `new` as written.

Check: `git diff --stat examples/*.ipynb` lists the three notebooks with small hunks; `git diff examples/create_block.ipynb | grep '"metadata"'` prints nothing.

- [ ] **Step 5: Dependent test edits**

`tests/test_gb_example_contract.py`: rename `test_gb_context_requires_independent_channels_and_one_psd_value_per_bin` to `test_gb_context_requires_independent_channels_and_returns_psd_units` and delete its three lines

```python
    full = Covariance.like(data, {"A": np.eye(33), "E": np.eye(33)})
    with pytest.raises(ValueError, match="one PSD value per bin"):
        block._context(data, full)
```

(the final `np.testing.assert_allclose(psds["A"], 2.0 * ones / data.Tobs)` stays: `C = 1` reads back as `S = 2 / Tobs`).

`tests/test_examples.py`: delete `test_sine_rejects_full_form_covariance` (the whole function, lines 345–363).

- [ ] **Step 6: Verify and commit**

Run: `grep -rn "from_variance\|from_psd\|noise_variance\|noise_psd\|\.form\b\|form=" examples benchmarks`
Expected: no output.

Run: `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks`
Expected: clean. If `ruff format --check` names a notebook, fix that cell's source in the script and rerun it rather than running `ruff format` on the notebook (that would rewrite the whole JSON).

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest -q`
Expected: `0 failed`, `3 xfailed` (two tests fewer than after Task 2). `tests/test_block_tutorial.py` must still print `PSD construction checks passed` and `Noise-only block checks passed` from the notebook code.

```bash
git add examples/toy_fit.py examples/gb_model.py benchmarks/benchmark_wheel.py examples/create_block.ipynb examples/domain_translation.ipynb examples/gb_block_eryn.ipynb tests/test_gb_example_contract.py tests/test_examples.py
git commit -m "Examples and notebooks build covariances with like and read pair; form guards removed

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 4: Rewrite `Covariance` around one stacked Hermitian array with a `Grid`

**Goal:** Replace `src/enchilada/covariance.py` with the spec §2 class (`blocks`, `channel_names`, `grid`; `like`, `pair`, attribute sugar, `in_basis` by `einsum`, `check_compatible`, `copy`, `__reduce__`), rewrite `tests/test_covariance.py`, and update the few call sites that only exist under the new layout (`.pairs` → `.blocks`, `"shape"` → `"grid"`, `_copy_validated` → `deepcopy`). `transform` and the Wheel keep their seam in this task; it goes in Task 5.

**Files:**
- Rewrite: `src/enchilada/covariance.py`
- Rewrite: `tests/test_covariance.py`
- Modify: `src/enchilada/wheel.py:149` (`Covariance._copy_validated(self._covariance)` → `deepcopy(self._covariance)`)
- Modify: `tests/test_wheel.py` (`UncheckedCovariance(...)` constructions, two `match="shape"`), `tests/test_state.py` (`.pairs[...]` edits), `tests/test_initial_block_state.py` (`.pairs[...]` edits, `"covariance": (ValueError, "shape")`), `tests/test_block_state.py` (`else "shape"`), `tests/test_translation.py` (`UncheckedCovariance(dict(cov.pairs), ...)`, four `match="shape"`)

**Acceptance Criteria:**
- [ ] `Covariance(blocks, channel_names=..., grid=...)` validates in the §2.2 order and messages; `blocks` is stored as a fresh read-only `float64`/`complex128` copy; `cov.A`, `cov.AE`, `cov.pair("A", "E")` are read-only views into `blocks`; `cov.EA` equals `np.conj(cov.AE)`.
- [ ] `Covariance.like(data, pairs)` is the only classmethod; `from_psd`, `from_variance`, `noise_psd`, `noise_variance`, `form`, `FORMS`, `_derived`, `_copy_validated`, `__getstate__`, `__setstate__` no longer exist (`grep -n "from_psd\|from_variance\|noise_psd\|noise_variance\|FORMS\|_derived\|_copy_validated\|__getstate__" src/enchilada/covariance.py` prints nothing).
- [ ] `in_basis` matches the explicit `M C Mᵀ` product per point, round-trips, returns a fresh object each call (no cache), keeps WDM structural zeros, and wraps a constructor `ValueError` as `cannot derive the ... basis`.
- [ ] `copy()` revalidates and `Covariance.copy(subclass_instance)` returns a plain `Covariance`; `deepcopy` and `pickle` restore a read-only fresh array without validating.
- [ ] `check_compatible(data)` requires equal channel order and `data.grid == self.grid` (message contains `grid`).
- [ ] Full suite `0 failed`, `3 xfailed`; mypy clean; ruff check/format clean; coverage of `covariance.py` ≥ 95 % lines.

**Verify:** `.venv/bin/python -m pytest -q tests/test_covariance.py` → all pass; `.venv/bin/python -m pytest -q` → `0 failed, 3 xfailed`; `.venv/bin/python -m mypy` → `Success`; `.venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks` → clean; `.venv/bin/python -m pytest -q --cov=enchilada --cov-report=term-missing tests/test_covariance.py | grep covariance.py` → ≥ 95 %.

**Steps:**

- [ ] **Step 1: Write the new `tests/test_covariance.py`** (replace the file):

```python
"""Covariance: one stacked Hermitian array per grid, read through channel names."""

import pickle
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import Covariance, Grid, L1Data, WDMGrid, transform
from enchilada.covariance import AET_FROM_XYZ

XYZ = ("X", "Y", "Z")
AET = ("A", "E", "T")


def time_data(num_time_samples=8, channel_names=("A", "E"), sample_rate_hz=2.0):
    rng = np.random.default_rng(0)
    return L1Data(
        {ch: rng.normal(size=num_time_samples) for ch in channel_names},
        sample_rate_hz=sample_rate_hz,
        channel_names=channel_names,
        tdi_generation="2.0",
        physical_observable="strain",
    )


def frequency_data(num_time_samples=8, channel_names=("A", "E")):
    return transform(time_data(num_time_samples, channel_names), "frequency")


def wdm_data(channel_names=("A", "E")):
    # Built by hand: the layout needs no WDM backend.
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=4)
    return L1Data(
        {ch: np.zeros(grid.array_shape) for ch in channel_names},
        sample_rate_hz=2.0,
        channel_names=channel_names,
        tdi_generation="2.0",
        physical_observable="strain",
        data_domain="wdm",
    )


def stacked(shape, variances, cross=0.0):
    """(K, K, *shape) blocks with constant variances and one constant cross term."""
    count = len(variances)
    blocks = np.full((count, count, *shape), cross, dtype=float)
    for index, variance in enumerate(variances):
        blocks[index, index] = variance
    return blocks


class TestConstruction:
    def test_stores_a_fresh_read_only_float_copy(self):
        data = time_data()
        blocks = stacked((8,), (1.0, 2.0), 0.5).astype(np.float32)
        cov = Covariance(blocks, channel_names=data.channel_names, grid=data.grid)
        assert cov.blocks.dtype == np.float64
        assert not cov.blocks.flags.writeable
        assert not np.shares_memory(cov.blocks, blocks)
        blocks[0, 0] = 100.0
        np.testing.assert_array_equal(cov.A, 1.0)
        assert cov.channel_names == ("A", "E") and cov.grid == data.grid

    def test_channel_names_may_be_a_list_and_are_stored_as_a_tuple(self):
        data = time_data()
        cov = Covariance(stacked((8,), (1.0, 2.0)), channel_names=["A", "E"], grid=data.grid)
        assert cov.channel_names == ("A", "E")

    @pytest.mark.parametrize(
        ("names", "error", "match"),
        [
            ("AE", TypeError, "tuple"),
            (("A", 1), TypeError, "strings"),
            ((), ValueError, "at least one"),
            (("A", "A"), ValueError, "unique"),
            (("A", ""), ValueError, "non-empty"),
        ],
    )
    def test_channel_names_validation(self, names, error, match):
        grid = Grid("time", 8, 2.0)
        with pytest.raises(error, match=match):
            Covariance(stacked((8,), (1.0, 2.0)), channel_names=names, grid=grid)

    def test_grid_must_be_a_grid(self):
        with pytest.raises(TypeError, match="grid must be a Grid"):
            Covariance(
                stacked((8,), (1.0, 2.0)), channel_names=("A", "E"), grid=("time", 8, 2.0)
            )

    def test_blocks_must_be_a_floating_array(self):
        grid = Grid("time", 8, 2.0)
        with pytest.raises(TypeError, match="numpy array"):
            Covariance(stacked((8,), (1.0, 2.0)).tolist(), channel_names=("A", "E"), grid=grid)
        with pytest.raises(TypeError, match="dtype"):
            Covariance(np.ones((2, 2, 8), dtype=int), channel_names=("A", "E"), grid=grid)

    @pytest.mark.parametrize("shape", [(2, 2, 7), (3, 3, 8), (2, 8), (2, 2, 8, 1)])
    def test_blocks_shape_must_match_channels_and_grid(self, shape):
        grid = Grid("time", 8, 2.0)
        with pytest.raises(ValueError, match=r"shape \(2, 2, 8\)"):
            Covariance(np.ones(shape), channel_names=("A", "E"), grid=grid)

    def test_blocks_must_be_finite(self):
        blocks = stacked((8,), (1.0, 2.0))
        blocks[1, 1, 3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    def test_blocks_must_be_exactly_hermitian(self):
        blocks = stacked((8,), (1.0, 2.0), 0.5)
        blocks[1, 0, 2] = 0.5 + 1e-15
        with pytest.raises(ValueError, match="Hermitian"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    @pytest.mark.parametrize("data", [time_data(), wdm_data()])
    def test_time_and_wdm_blocks_must_be_real(self, data):
        blocks = np.zeros((2, 2, *data.grid.channel_shape), dtype=complex)
        blocks[0, 0] = blocks[1, 1] = np.where(data.grid.active_mask, 1.0, 0.0)
        with pytest.raises(ValueError, match=f"real for data_domain={data.data_domain!r}"):
            Covariance(blocks, channel_names=data.channel_names, grid=data.grid)

    @pytest.mark.parametrize("n", [8, 9])
    def test_frequency_blocks_must_be_real_on_the_diagonal_and_at_real_bins(self, n):
        data = frequency_data(n)
        bins = n // 2 + 1
        blocks = stacked((bins,), (1.0, 2.0)).astype(complex)
        bad_diagonal = blocks.copy()
        bad_diagonal[0, 0, 2] = 1.0 + 1e-3j  # fails exact Hermitian symmetry
        with pytest.raises(ValueError, match="Hermitian"):
            Covariance(bad_diagonal, channel_names=("A", "E"), grid=data.grid)
        bad_dc = blocks.copy()
        bad_dc[0, 1, 0], bad_dc[1, 0, 0] = 0.1j, -0.1j
        with pytest.raises(ValueError, match="DC and even-length Nyquist"):
            Covariance(bad_dc, channel_names=("A", "E"), grid=data.grid)
        nyquist = blocks.copy()
        nyquist[0, 1, -1], nyquist[1, 0, -1] = 0.1j, -0.1j
        if n % 2 == 0:
            with pytest.raises(ValueError, match="DC and even-length Nyquist"):
                Covariance(nyquist, channel_names=("A", "E"), grid=data.grid)
        else:
            Covariance(nyquist, channel_names=("A", "E"), grid=data.grid)

    def test_frequency_blocks_may_have_complex_hermitian_cross_terms(self):
        data = frequency_data(8)
        blocks = stacked((5,), (1.0, 2.0)).astype(complex)
        blocks[0, 1, 1:-1] = 0.3 + 0.1j
        blocks[1, 0, 1:-1] = 0.3 - 0.1j
        cov = Covariance(blocks, channel_names=("A", "E"), grid=data.grid)
        assert cov.blocks.dtype == np.complex128
        np.testing.assert_array_equal(cov.EA, np.conj(cov.AE))
        assert cov.A.dtype == np.complex128 and np.all(cov.A.imag == 0)

    def test_variances_must_be_positive_at_every_active_point(self):
        blocks = stacked((8,), (1.0, 2.0))
        blocks[1, 1, 5] = 0.0
        with pytest.raises(ValueError, match="variance must be positive"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    def test_inactive_wdm_pixels_must_be_entirely_zero(self):
        data = wdm_data()
        active = data.grid.active_mask
        blocks = np.zeros((2, 2, *active.shape))
        blocks[0, 0] = blocks[1, 1] = np.where(active, 1.0, 0.0)
        cov = Covariance(blocks, channel_names=data.channel_names, grid=data.grid)
        np.testing.assert_array_equal(cov.A[~active], 0.0)
        leaking_variance = blocks.copy()
        leaking_variance[0, 0][~active] = 1.0
        with pytest.raises(ValueError, match="inactive WDM pixels"):
            Covariance(leaking_variance, channel_names=data.channel_names, grid=data.grid)
        leaking_cross = blocks.copy()
        leaking_cross[0, 1][~active] = leaking_cross[1, 0][~active] = 0.1
        with pytest.raises(ValueError, match="inactive WDM pixels"):
            Covariance(leaking_cross, channel_names=data.channel_names, grid=data.grid)

    def test_channel_matrices_must_be_positive_definite(self):
        blocks = stacked((8,), (1.0, 1.0), 0.5)
        blocks[0, 1, 4] = blocks[1, 0, 4] = 1.5
        with pytest.raises(ValueError, match="positive definite at every active point"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    def test_equality_is_identity(self):
        data = time_data()
        a = Covariance(stacked((8,), (1.0, 2.0)), channel_names=("A", "E"), grid=data.grid)
        b = Covariance(stacked((8,), (1.0, 2.0)), channel_names=("A", "E"), grid=data.grid)
        assert a == a and a != b


class TestLike:
    def test_adopts_the_data_grid_and_channel_order(self):
        data = L1Data(
            {"E": np.ones(8), "A": np.ones(8)},
            sample_rate_hz=2.0,
            tdi_generation="2.0",
            physical_observable="strain",
        )
        cov = Covariance.like(data, {"A": np.full(8, 1.0), "E": np.full(8, 4.0)})
        assert cov.channel_names == ("E", "A") and cov.grid == data.grid
        np.testing.assert_array_equal(cov.blocks[0, 0], 4.0)
        np.testing.assert_array_equal(cov.A, 1.0)
        np.testing.assert_array_equal(cov.AE, 0.0)

    def test_string_and_tuple_keys_in_either_orientation(self):
        data = frequency_data(8)
        cross = np.array([0.0, 0.1j, 0.2j, 0.3j, 0.0])
        by_string = Covariance.like(
            data, {"A": np.ones(5), "E": np.ones(5), "EA": cross}
        )
        by_tuple = Covariance.like(
            data, {"A": np.ones(5), "E": np.ones(5), ("E", "A"): cross}
        )
        for cov in (by_string, by_tuple):
            np.testing.assert_array_equal(cov.pair("E", "A"), cross)
            np.testing.assert_array_equal(cov.pair("A", "E"), np.conj(cross))

    def test_unknown_and_unresolvable_keys(self):
        data = time_data()
        ones = np.ones(8)
        with pytest.raises(ValueError, match="neither a channel nor a pair"):
            Covariance.like(data, {"A": ones, "E": ones, "Q": ones})
        with pytest.raises(ValueError, match="unknown channel 'Q'"):
            Covariance.like(data, {"A": ones, "E": ones, ("A", "Q"): ones})
        with pytest.raises(TypeError, match="pair keys"):
            Covariance.like(data, {"A": ones, "E": ones, 1: ones})
        with pytest.raises(TypeError, match="mapping"):
            Covariance.like(data, [("A", ones)])

    def test_two_keys_naming_one_pair_are_rejected(self):
        data = time_data()
        ones, cross = np.ones(8), np.full(8, 0.1)
        with pytest.raises(ValueError, match="both name pair"):
            Covariance.like(data, {"A": ones, "E": ones, "AE": cross, "EA": cross})
        with pytest.raises(ValueError, match="both name pair"):
            Covariance.like(data, {"A": ones, "E": ones, "AE": cross, ("A", "E"): cross})

    def test_every_diagonal_is_required(self):
        with pytest.raises(ValueError, match=r"missing the diagonal entry for channels \['E'\]"):
            Covariance.like(time_data(), {"A": np.ones(8)})

    def test_values_must_be_floating_arrays_with_the_channel_shape(self):
        data = time_data()
        with pytest.raises(TypeError, match="numpy array"):
            Covariance.like(data, {"A": [1.0] * 8, "E": np.ones(8)})
        with pytest.raises(TypeError, match="dtype"):
            Covariance.like(data, {"A": np.ones(8, dtype=int), "E": np.ones(8)})
        with pytest.raises(ValueError, match=r"channel shape \(8,\)"):
            Covariance.like(data, {"A": np.ones(7), "E": np.ones(8)})
        with pytest.raises(ValueError, match=r"channel shape \(8,\)"):
            Covariance.like(data, {"A": np.eye(8), "E": np.eye(8)})

    def test_ambiguous_string_keys_are_errors(self):
        data = time_data(channel_names=("A", "B", "AB", "BA"))
        pairs = {name: np.ones(8) for name in data.channel_names}
        with pytest.raises(ValueError, match="ambiguous"):
            Covariance.like(data, {**pairs, "ABA": np.full(8, 0.1)})
        cov = Covariance.like(data, {**pairs, ("AB", "A"): np.full(8, 0.1)})
        np.testing.assert_array_equal(cov.pair("AB", "A"), 0.1)

    def test_result_is_validated(self):
        data = time_data()
        with pytest.raises(ValueError, match="positive definite"):
            Covariance.like(data, {"A": np.ones(8), "E": np.ones(8), "AE": np.full(8, 2.0)})

    def test_wdm_structural_zeros(self):
        data = wdm_data()
        variances = np.where(data.grid.active_mask, 3.0, 0.0)
        cov = Covariance.like(data, {"A": variances, "E": variances})
        np.testing.assert_array_equal(cov.A, variances)
        assert cov.grid.data_domain == "wdm"


class TestAccess:
    def test_pair_returns_a_read_only_view_into_blocks(self):
        cov = white_covariance(time_data(), {"A": 1.0, "E": 4.0})
        view = cov.pair("A", "E")
        assert np.shares_memory(view, cov.blocks)
        assert not view.flags.writeable
        np.testing.assert_array_equal(cov.pair("E", "E"), 4.0)
        with pytest.raises(ValueError, match="unknown channel 'Q'"):
            cov.pair("A", "Q")

    def test_attribute_sugar(self):
        data = frequency_data(8)
        cross = np.array([0.0, 0.1j, 0.2j, 0.3j, 0.0])
        cov = Covariance.like(data, {"A": np.ones(5), "E": np.ones(5), "AE": cross})
        np.testing.assert_array_equal(cov.A, 1.0)
        np.testing.assert_array_equal(cov.AE, cross)
        np.testing.assert_array_equal(cov.EA, np.conj(cross))
        assert np.shares_memory(cov.AE, cov.blocks)

    def test_ambiguous_attribute_names_need_pair(self):
        data = time_data(channel_names=("A", "B", "AB", "BA"))
        cov = white_covariance(data, 1.0)
        with pytest.raises(AttributeError, match="ambiguous"):
            _ = cov.ABA
        np.testing.assert_array_equal(cov.pair("AB", "A"), 0.0)

    def test_unknown_attribute_lists_valid_names(self):
        cov = white_covariance(time_data(), 1.0)
        with pytest.raises(AttributeError, match="channel attributes are .*'AE'"):
            _ = cov.Q
        assert not hasattr(cov, "form")

    def test_uninitialised_instance_has_no_channel_attributes(self):
        empty = object.__new__(Covariance)
        assert not hasattr(empty, "blocks")
        assert not hasattr(empty, "A")
        assert not hasattr(empty, "__deepcopy__")


def xyz_covariance(data, cross=0.2):
    shape = data.grid.channel_shape
    pairs = {
        "X": np.full(shape, 1.0),
        "Y": np.full(shape, 2.0),
        "Z": np.full(shape, 3.0),
        "XY": np.full(shape, cross),
        "YZ": np.full(shape, -cross),
    }
    return Covariance.like(data, pairs)


class TestInBasis:
    def test_aet_matches_the_explicit_matrix_product(self):
        cov = xyz_covariance(time_data(channel_names=XYZ))
        aet = cov.in_basis("AET")
        assert aet.channel_names == AET and aet.grid == cov.grid
        expected = np.einsum("ai,bj,ijn->abn", AET_FROM_XYZ, AET_FROM_XYZ, cov.blocks)
        np.testing.assert_allclose(aet.blocks, expected, atol=1e-14)
        assert not aet.blocks.flags.writeable

    def test_round_trip_without_a_cache(self):
        cov = xyz_covariance(time_data(channel_names=XYZ))
        aet = cov.in_basis("AET")
        np.testing.assert_allclose(aet.in_basis("XYZ").blocks, cov.blocks, atol=1e-14)
        assert cov.in_basis("XYZ") is cov
        assert cov.in_basis("AET") is not aet

    def test_white_equal_noise_stays_white(self):
        cov = white_covariance(time_data(channel_names=XYZ), 2.0)
        aet = cov.in_basis("AET")
        for name in AET:
            np.testing.assert_allclose(aet.pair(name, name), 2.0)
        np.testing.assert_allclose(aet.AE, 0.0, atol=1e-14)

    def test_derived_attributes_read_through_the_other_basis(self):
        cov = xyz_covariance(time_data(channel_names=XYZ))
        np.testing.assert_array_equal(cov.A, cov.in_basis("AET").A)
        np.testing.assert_array_equal(cov.AE, cov.in_basis("AET").AE)
        aet = cov.in_basis("AET")
        np.testing.assert_allclose(aet.X, cov.X, atol=1e-14)
        np.testing.assert_allclose(aet.pair("Y", "Z"), cov.YZ, atol=1e-14)

    def test_any_channel_order_works(self):
        ordered = xyz_covariance(time_data(channel_names=XYZ))
        shuffled = Covariance.like(
            time_data(channel_names=("Z", "X", "Y")),
            {
                "X": ordered.X, "Y": ordered.Y, "Z": ordered.Z,
                "XY": ordered.XY, "YZ": ordered.YZ,
            },
        )
        np.testing.assert_allclose(
            shuffled.in_basis("AET").blocks, ordered.in_basis("AET").blocks, atol=1e-14
        )

    def test_basis_needs_the_full_triple_and_a_known_name(self):
        cov = white_covariance(time_data(), 1.0)
        with pytest.raises(ValueError, match="exactly X, Y, Z or A, E, T"):
            cov.in_basis("AET")
        with pytest.raises(ValueError, match="basis must be one of"):
            xyz_covariance(time_data(channel_names=XYZ)).in_basis("ABC")
        with pytest.raises(AttributeError):
            _ = cov.X

    def test_derivation_below_float64_precision_explains_itself(self):
        data = time_data(num_time_samples=64, channel_names=AET)
        cov = Covariance.like(
            data,
            {
                "A": np.linspace(1, 3, 64),
                "E": np.linspace(2, 5, 64),
                "T": np.full(64, 1e-20),
            },
        )
        with pytest.raises(ValueError, match="cannot derive the XYZ basis") as info:
            cov.in_basis("XYZ")
        assert isinstance(info.value.__cause__, ValueError)
        with pytest.raises(ValueError, match="cannot derive the XYZ basis"):
            _ = cov.X

    def test_frequency_complex_blocks_rotate_hermitian(self):
        data = frequency_data(8, channel_names=XYZ)
        pairs = {
            "X": np.ones(5), "Y": np.full(5, 2.0), "Z": np.full(5, 3.0),
            "XY": np.array([0.0, 0.1j, 0.2j, 0.1j, 0.0]),
        }
        aet = Covariance.like(data, pairs).in_basis("AET")
        assert aet.blocks.dtype == np.complex128
        np.testing.assert_array_equal(aet.blocks, np.conj(np.swapaxes(aet.blocks, 0, 1)))
        assert np.all(aet.A.imag == 0)

    def test_wdm_derivation_keeps_structural_zeros(self):
        data = wdm_data(channel_names=XYZ)
        active = data.grid.active_mask
        aet = Covariance.like(
            data,
            {name: np.where(active, level, 0.0) for name, level in zip(XYZ, (1.0, 2.0, 3.0))},
        ).in_basis("AET")
        np.testing.assert_array_equal(aet.blocks[:, :, ~active], 0.0)
        assert aet.grid == data.grid


class TestCopies:
    def test_copy_is_independent_and_revalidated(self):
        cov = white_covariance(time_data(), 1.0)
        copied = cov.copy()
        assert type(copied) is Covariance and copied is not cov
        np.testing.assert_array_equal(copied.blocks, cov.blocks)
        assert not np.shares_memory(copied.blocks, cov.blocks)
        assert not copied.blocks.flags.writeable
        cov.blocks.setflags(write=True)
        cov.A[:] = -1.0
        with pytest.raises(ValueError, match="variance must be positive"):
            cov.copy()

    def test_unbound_copy_returns_a_plain_covariance(self):
        class Unchecked(Covariance):
            def copy(self):
                return self

        cov = white_covariance(time_data(), 1.0)
        unchecked = Unchecked(cov.blocks, channel_names=cov.channel_names, grid=cov.grid)
        assert Covariance.copy(unchecked) is not unchecked
        assert type(Covariance.copy(unchecked)) is Covariance

    def test_deepcopy_and_pickle_restore_read_only_without_validation(self):
        cov = white_covariance(frequency_data(8), 1.0)
        for restored in (deepcopy(cov), pickle.loads(pickle.dumps(cov))):
            assert type(restored) is Covariance
            np.testing.assert_array_equal(restored.blocks, cov.blocks)
            assert not np.shares_memory(restored.blocks, cov.blocks)
            assert not restored.blocks.flags.writeable
            assert restored.channel_names == cov.channel_names
            assert restored.grid == cov.grid
            np.testing.assert_array_equal(restored.A, cov.A)
        cov.blocks.setflags(write=True)
        cov.A[:] = -1.0  # invalid, yet the restore paths do not validate
        np.testing.assert_array_equal(deepcopy(cov).A, -1.0)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(cov)).A, -1.0)


class TestCheckCompatible:
    def test_accepts_the_data_it_was_built_from(self):
        data = time_data()
        white_covariance(data, 1.0).check_compatible(data)
        white_covariance(data, 1.0).check_compatible(replace(data, start_time_gps=9.0))

    def test_rejects_other_channel_order_and_other_grids(self):
        data = time_data()
        cov = white_covariance(data, 1.0)
        reordered = L1Data(
            {"E": data.channel_data["E"], "A": data.channel_data["A"]},
            sample_rate_hz=2.0,
            tdi_generation="2.0",
            physical_observable="strain",
        )
        with pytest.raises(ValueError, match="channel_names"):
            cov.check_compatible(reordered)
        with pytest.raises(ValueError, match="grid"):
            cov.check_compatible(transform(data, "frequency"))
        with pytest.raises(ValueError, match="grid"):
            cov.check_compatible(time_data(num_time_samples=16))
        with pytest.raises(ValueError, match="grid"):
            cov.check_compatible(time_data(sample_rate_hz=1.0))
```

Run: `.venv/bin/python -m ruff format tests/test_covariance.py` (several lines above exceed 88 columns on purpose for readability; the formatter wraps them), then `.venv/bin/python -m pytest -q tests/test_covariance.py`.
Expected: failures everywhere (`Covariance.__init__() got an unexpected keyword argument 'grid'`, `AttributeError: blocks`, ...).

- [ ] **Step 2: Replace `src/enchilada/covariance.py` with the new class**

```python
"""Noise covariance: one stacked Hermitian array on a grid, read by channel name."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from enchilada.domains import Grid

if TYPE_CHECKING:
    from enchilada.data import L1Data

AET_FROM_XYZ = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ]
)
"""Rows A, E, T from columns X, Y, Z: the orthonormal LDC combination."""

BASES = {"XYZ": ("X", "Y", "Z"), "AET": ("A", "E", "T")}


def split_pair_name(
    name: str, channel_names: tuple[str, ...]
) -> tuple[str, str] | None:
    """Resolve ``"X"`` to ``("X", "X")`` and ``"XY"`` to ``("X", "Y")``.

    Returns None when ``name`` is neither a channel nor a concatenation of two
    channels. Raises ValueError when more than one split works.
    """
    if name in channel_names:
        return (name, name)
    candidates = [
        (name[:cut], name[cut:])
        for cut in range(1, len(name))
        if name[:cut] in channel_names and name[cut:] in channel_names
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"{name!r} is ambiguous between channel pairs {candidates}; "
            "use a (first, second) tuple or pair(first, second)"
        )
    return candidates[0] if candidates else None


def _validated_channel_names(names: object) -> tuple[str, ...]:
    if not isinstance(names, (tuple, list)):
        raise TypeError("channel_names must be a tuple of channel names")
    channels = tuple(names)
    for name in channels:
        if not isinstance(name, str):
            raise TypeError(f"channel names must be strings, got {name!r}")
        if not name:
            raise ValueError("channel names must be non-empty strings")
    if not channels:
        raise ValueError("channel_names must name at least one channel")
    if len(set(channels)) != len(channels):
        raise ValueError(f"channel names must be unique, got {channels}")
    return channels


def _restore(
    blocks: np.ndarray, channel_names: tuple[str, ...], grid: Grid
) -> "Covariance":
    """Rebuild a Covariance from its fields without validating (deepcopy, pickle).

    Both callers hand over a fresh, writeable array copy; only the write flag
    has to be restored. Only objects that already passed the constructor reach
    this path, so the batched Cholesky is not repeated.
    """
    restored = object.__new__(Covariance)
    blocks.setflags(write=False)
    object.__setattr__(restored, "blocks", blocks)
    object.__setattr__(restored, "channel_names", channel_names)
    object.__setattr__(restored, "grid", grid)
    return restored


@dataclass(frozen=True, eq=False)
class Covariance:
    """Noise covariance on a grid, as one array of channel-by-channel blocks.

    ``blocks[i, j]`` is ``E[x_i conj(x_j)]`` at every point of ``grid``, so
    ``blocks`` has shape ``(K, K) + grid.channel_shape`` for ``K`` channels and
    ``blocks[j, i] == conj(blocks[i, j])``. There is one value per point; a
    covariance between different points is out of scope.

    Build one from a mapping with :meth:`like`. Read it through channel
    attributes: ``cov.X`` is ``pair("X", "X")``, ``cov.XY`` is
    ``pair("X", "Y")`` and ``cov.YX`` its conjugate. With channels exactly
    X, Y, Z the names A, E, T and their pairs are derived on demand, and the
    reverse; see :meth:`in_basis`. A derivation that is not numerically
    positive definite raises ``ValueError``, so ``hasattr(cov, "X")`` can raise
    ``ValueError`` for such an object; this is deliberate so the numerical
    failure is not hidden.

    Units: ``blocks`` holds variances in the squared units of the data's
    coordinates on its grid (``x_n**2`` in time, ``(dt * rfft(x))**2`` in
    frequency, squared WDM coefficients). Stationary noise with one-sided PSD
    ``S`` has bin variance ``(Tobs / 2) * S(f_k)`` at every rfft bin.

    Construction validates channel names, the grid, dtype, shape, finiteness,
    exact Hermitian symmetry, realness where the domain requires it, positive
    variances at every active point, zero blocks at inactive WDM pixels and
    positive definiteness at every active point. The array is stored as a
    fresh read-only copy. Equality is identity; compare arrays explicitly.
    """

    blocks: np.ndarray
    channel_names: tuple[str, ...] = field(kw_only=True)
    grid: Grid = field(kw_only=True)

    # ---- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        channels = _validated_channel_names(self.channel_names)
        object.__setattr__(self, "channel_names", channels)
        if not isinstance(self.grid, Grid):
            raise TypeError("grid must be a Grid")
        blocks = self._validated_blocks(self.blocks, channels, self.grid)
        object.__setattr__(self, "blocks", blocks)

    @staticmethod
    def _validated_blocks(
        value: object, channels: tuple[str, ...], grid: Grid
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError("blocks must be a numpy array")
        if value.dtype.kind not in "fc":
            raise TypeError(
                f"blocks must have a floating or complex dtype, got {value.dtype}"
            )
        count = len(channels)
        expected = (count, count, *grid.channel_shape)
        if value.shape != expected:
            raise ValueError(
                f"blocks must have shape {expected} for {count} channels on this "
                f"{grid.data_domain} grid, got {value.shape}"
            )
        dtype = np.complex128 if value.dtype.kind == "c" else np.float64
        blocks = np.array(value, dtype=dtype, copy=True)
        if not np.all(np.isfinite(blocks)):
            raise ValueError("blocks must be finite")
        # Exact symmetry also forces every blocks[i, i] to be real.
        if np.any(blocks != np.conj(np.swapaxes(blocks, 0, 1))):
            raise ValueError(
                "blocks must be Hermitian at every point: "
                "blocks[j, i] == conj(blocks[i, j])"
            )
        if blocks.dtype.kind == "c":
            if grid.data_domain != "frequency":
                raise ValueError(
                    f"blocks must be real for data_domain={grid.data_domain!r}"
                )
            real_bins = [0] + ([-1] if grid.num_time_samples % 2 == 0 else [])
            if np.any(blocks[:, :, real_bins].imag != 0):
                raise ValueError(
                    "blocks must be real where both coefficients are real "
                    "(DC and even-length Nyquist)"
                )
        variances = np.real(np.diagonal(blocks, axis1=0, axis2=1))  # (*S, K)
        active = grid.active_mask
        if np.any(variances[active] <= 0):
            raise ValueError(
                "every channel's variance must be positive at every active point"
            )
        if np.any(blocks[:, :, ~active] != 0):
            raise ValueError(
                "blocks must be zero at structurally inactive WDM pixels"
            )
        matrices = np.moveaxis(blocks[:, :, active], (0, 1), (-2, -1))  # (P, K, K)
        try:
            np.linalg.cholesky(matrices)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "covariance must be positive definite at every active point"
            ) from exc
        blocks.setflags(write=False)
        return blocks

    @classmethod
    def like(cls, data: "L1Data", pairs: Mapping[Any, np.ndarray]) -> "Covariance":
        """Build a covariance on ``data``'s grid from per-pair arrays.

        Keys are a channel name (``"A"``), a two-channel concatenation
        (``"AE"``, resolved with :func:`split_pair_name`) or a
        ``(first, second)`` tuple. A key ``(a, b)`` sets ``blocks[a, b]`` to the
        array and ``blocks[b, a]`` to its conjugate; two keys naming one pair in
        either orientation raise ``ValueError``. Every channel's own entry is
        required; an absent cross term is zero. Each value is a floating or
        complex array with ``data.grid.channel_shape``. The result is validated
        by the constructor.
        """
        channels = tuple(data.channel_names)
        grid = data.grid
        if not isinstance(pairs, Mapping):
            raise TypeError(
                "pairs must be a mapping from channel or pair names to arrays"
            )
        origins: dict[tuple[str, str], object] = {}
        entries: dict[tuple[str, str], np.ndarray] = {}
        for key, value in pairs.items():
            first, second = cls._resolved_key(key, channels)
            i, j = channels.index(first), channels.index(second)
            canonical = (channels[min(i, j)], channels[max(i, j)])
            if canonical in entries:
                raise ValueError(
                    f"keys {origins[canonical]!r} and {key!r} both name pair "
                    f"{canonical!r}"
                )
            if not isinstance(value, np.ndarray):
                raise TypeError(f"pairs[{key!r}] must be a numpy array")
            if value.dtype.kind not in "fc":
                raise TypeError(
                    f"pairs[{key!r}] must have a floating or complex dtype, "
                    f"got {value.dtype}"
                )
            if value.shape != grid.channel_shape:
                raise ValueError(
                    f"pairs[{key!r}] must have the channel shape "
                    f"{grid.channel_shape} of this {grid.data_domain} grid, "
                    f"got {value.shape}"
                )
            entries[canonical] = value if i <= j else np.conj(value)
            origins[canonical] = key
        missing = [ch for ch in channels if (ch, ch) not in entries]
        if missing:
            raise ValueError(
                f"pairs is missing the diagonal entry for channels {missing}"
            )
        dtype = (
            np.complex128
            if any(np.iscomplexobj(array) for array in entries.values())
            else np.float64
        )
        blocks = np.zeros((len(channels), len(channels), *grid.channel_shape), dtype)
        for (first, second), array in entries.items():
            i, j = channels.index(first), channels.index(second)
            blocks[i, j] = array
            blocks[j, i] = np.conj(array)
        return cls(blocks, channel_names=channels, grid=grid)

    @staticmethod
    def _resolved_key(key: object, channels: tuple[str, ...]) -> tuple[str, str]:
        if isinstance(key, str):
            resolved = split_pair_name(key, channels)
            if resolved is None:
                raise ValueError(
                    f"key {key!r} is neither a channel nor a pair of channels "
                    f"from {channels}"
                )
            return resolved
        if isinstance(key, tuple) and len(key) == 2:
            first, second = key
            if isinstance(first, str) and isinstance(second, str):
                for name in (first, second):
                    if name not in channels:
                        raise ValueError(
                            f"unknown channel {name!r} in key {key!r}; "
                            f"channels are {channels}"
                        )
                return first, second
        raise TypeError(
            "pair keys must be a channel name, a two-channel name or a "
            f"(first, second) tuple, got {key!r}"
        )

    # ---- access -----------------------------------------------------------

    def pair(self, first: str, second: str) -> np.ndarray:
        """Covariance between two channels at every point: a read-only view."""
        for name in (first, second):
            if name not in self.channel_names:
                raise ValueError(
                    f"unknown channel {name!r}; channels are {self.channel_names}"
                )
        return self.blocks[
            self.channel_names.index(first), self.channel_names.index(second)
        ]

    def __getattr__(self, name: str) -> np.ndarray:
        # Only reached when normal lookup fails, so fields and methods win.
        if name.startswith("_") or name in _FIELD_NAMES:
            raise AttributeError(name)
        channels = self.channel_names
        try:
            resolved = split_pair_name(name, channels)
        except ValueError as exc:
            raise AttributeError(str(exc)) from None
        if resolved is not None:
            return self.pair(*resolved)
        derived_basis = self._other_basis()
        if derived_basis is not None:
            if split_pair_name(name, BASES[derived_basis]) is not None:
                return getattr(self.in_basis(derived_basis), name)
        valid = [*channels, *(a + b for a in channels for b in channels if a != b)]
        if derived_basis is not None:
            derived_channels = BASES[derived_basis]
            valid += [
                *derived_channels,
                *(a + b for a in derived_channels for b in derived_channels if a != b),
            ]
        raise AttributeError(
            f"Covariance has no attribute {name!r}; channel attributes are {valid}"
        )

    # ---- derived basis ----------------------------------------------------

    def _basis_name(self) -> str | None:
        current = frozenset(self.channel_names)
        for basis, names in BASES.items():
            if current == frozenset(names):
                return basis
        return None

    def _other_basis(self) -> str | None:
        basis = self._basis_name()
        if basis is None:
            return None
        return "AET" if basis == "XYZ" else "XYZ"

    def in_basis(self, basis: str) -> "Covariance":
        """Return this covariance in the ``"XYZ"`` or ``"AET"`` basis.

        Available only when the channels are exactly X, Y, Z or exactly A, E, T.
        The current basis returns ``self``; otherwise a new object with channel
        order ``("A", "E", "T")`` or ``("X", "Y", "Z")`` on the same grid is
        computed on every call (no cache). The rotation is applied per point and
        symmetrised exactly.

        Derivation needs the rotated covariance to be numerically positive
        definite; when one channel's variance is below float64 roundoff of the
        others (T << A at the lowest LISA frequencies) this raises
        ``ValueError`` with that explanation, chaining the constructor's error.
        """
        if basis not in BASES:
            raise ValueError(f"basis must be one of {tuple(BASES)}, got {basis!r}")
        current = self._basis_name()
        if current is None:
            raise ValueError(
                "deriving a basis needs channels exactly X, Y, Z or A, E, T; "
                f"got {self.channel_names}"
            )
        if basis == current:
            return self
        source, target = BASES[current], BASES[basis]
        order = [self.channel_names.index(name) for name in source]
        ordered = self.blocks[np.ix_(order, order)]
        matrix = AET_FROM_XYZ if basis == "AET" else AET_FROM_XYZ.T
        rotated = np.einsum("ai,bj,ij...->ab...", matrix, matrix, ordered)
        rotated = 0.5 * (rotated + np.conj(np.swapaxes(rotated, 0, 1)))
        for index in range(len(target)):
            rotated[index, index] = rotated[index, index].real
        try:
            return Covariance(rotated, channel_names=target, grid=self.grid)
        except ValueError as exc:
            raise ValueError(
                f"cannot derive the {basis} basis from {current}: the rotated "
                "covariance is not numerically positive definite (one channel's "
                "variance is below the roundoff of the others, e.g. T << A at "
                f"low frequency): {exc}"
            ) from exc

    # ---- the accompanying data --------------------------------------------

    def check_compatible(self, data: "L1Data") -> None:
        """Require this covariance to describe ``data``: same channels, same grid."""
        if tuple(data.channel_names) != self.channel_names:
            raise ValueError(
                f"covariance channel_names {self.channel_names} do not match the "
                f"data's {tuple(data.channel_names)}"
            )
        if data.grid != self.grid:
            raise ValueError(
                f"covariance grid {self.grid} does not match the data's grid "
                f"{data.grid}"
            )

    # ---- copies -----------------------------------------------------------

    def copy(self) -> "Covariance":
        """An independent, revalidated plain ``Covariance``.

        The trust boundary: the Wheel calls ``Covariance.copy(x)`` unbound on
        every covariance it receives, so a subclass cannot skip validation and
        an array whose write flag a caller flipped is caught.
        """
        return Covariance(
            self.blocks, channel_names=self.channel_names, grid=self.grid
        )

    def __reduce__(
        self,
    ) -> tuple[Callable[..., "Covariance"], tuple[np.ndarray, tuple[str, ...], Grid]]:
        # deepcopy and pickle both copy the array and restore without validating.
        return _restore, (self.blocks, self.channel_names, self.grid)


_FIELD_NAMES = frozenset(f.name for f in fields(Covariance))
```

Run `.venv/bin/python -m ruff format src/enchilada/covariance.py` and `.venv/bin/python -m pytest -q tests/test_covariance.py`.
Expected: all pass. If `test_derivation_below_float64_precision_explains_itself` does not raise, the 64 rotated 3×3 matrices all came out with a positive third pivot; widen the test's spread (`np.linspace(1, 30, 64)`) rather than weakening the check.

- [ ] **Step 3: Update the call sites that only exist under the new layout**

`src/enchilada/wheel.py` (the `covariance` property, line 149): `return Covariance._copy_validated(self._covariance)` → `return deepcopy(self._covariance)` (the comment line above it goes too; `deepcopy` is already imported).

`tests/test_wheel.py`, in `test_covariance_subclass_cannot_skip_grid_validation` (around line 1028):

```python
        unchecked = UncheckedCovariance(
            covariance.blocks, channel_names=covariance.channel_names, grid=covariance.grid
        )
```

and

```python
        compatible = UncheckedCovariance(
            white_covariance(observed, 2.0).blocks,
            channel_names=observed.channel_names,
            grid=observed.grid,
        )
```

and both `with pytest.raises(ValueError, match="shape"):` in that test → `match="grid"`.

`tests/test_state.py` (`test_covariance_input_and_output_snapshots_are_isolated`): every pair of lines

```python
    covariance.pairs[("A", "A")].setflags(write=True)
    covariance.pairs[("A", "A")][:] *= 2
```

becomes

```python
    covariance.blocks.setflags(write=True)
    covariance.A[:] *= 2
```

(same for `inspected` with `*= 3` and the block-local `covariance` with `*= 4`).

`tests/test_initial_block_state.py`: `initial.covariance.pairs[("A", "A")].setflags(write=True)` / `initial.covariance.pairs[("A", "A")][:] = 100` → `initial.covariance.blocks.setflags(write=True)` / `initial.covariance.A[:] = 100`; `"covariance": (ValueError, "shape"),` → `"covariance": (ValueError, "grid"),`.

`tests/test_block_state.py`: `error = "must return.*covariance" if failure == "omitted" else "shape"` → `... else "grid"`.

`tests/test_translation.py`: `return UncheckedCovariance(dict(cov.pairs), channel_names=cov.channel_names)` → `return UncheckedCovariance(cov.blocks, channel_names=cov.channel_names, grid=cov.grid)`; the four `match="shape"` on covariance calls (in `test_conversion_rejects_missing_or_mismatched_reference`, twice in `test_covariance_cross_domain_transform_is_the_seam`, once in `test_covariance_seam_copies_through_covariance_methods`) → `match="grid"`. (These seam tests are deleted in Task 5; keeping them green here keeps the suite green.)

- [ ] **Step 4: Verify and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: `0 failed`, `3 xfailed`.

Run: `.venv/bin/python -m mypy && .venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks`
Expected: clean.

Run: `.venv/bin/python -m pytest -q --cov=enchilada --cov-report=term-missing tests/test_covariance.py | grep covariance.py`
Expected: ≥ 95 % for `src/enchilada/covariance.py`; add a test for any missed branch.

```bash
git add src/enchilada/covariance.py src/enchilada/wheel.py tests/test_covariance.py tests/test_wheel.py tests/test_state.py tests/test_initial_block_state.py tests/test_block_state.py tests/test_translation.py
git commit -m "Covariance is one stacked Hermitian array on a Grid; like is the one constructor

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 5: Remove the seam: `transform` refuses a `Covariance`, the Wheel holds one native covariance on its owner's grid, the seven xfails pass

**Goal:** Drop the `Covariance` overload of `transform` (a `Covariance` argument raises `TypeError`; a `BlockState` carries its covariance through deep copy), make the Wheel keep the covariance on whatever grid its owner published it and refuse a block on another grid with `ValueError` naming both grids, replace the seam tests with refusal tests, and turn the seven strict xfails into passing tests.

**Files:**
- Rewrite: `src/enchilada/translation.py`
- Rewrite: `src/enchilada/wheel.py`
- Modify: `tests/test_translation.py`, `tests/test_wheel_translation.py`, `tests/test_translation_review.py`, `tests/test_initial_block_state.py`, `tests/test_gb_example_contract.py`, `tests/test_toy_translation.py`, `tests/test_typing.py`

**Acceptance Criteria:**
- [ ] `transform(covariance, ...)` raises `TypeError` matching `does not take a Covariance.*copy`; `transform(block_state, ...)` translates the template (with `reference_data`) and returns the covariance as an unaliased, read-only deep copy on its own grid; `transform` of anything else raises `TypeError("transform expects L1Data or BlockState")`.
- [ ] `Wheel(observed, cov)` accepts a covariance in any domain of the same record (`same_record`) with the observations' channel names, and refuses others with `ValueError` mentioning `grid`; `Wheel.covariance` is a `deepcopy`.
- [ ] A block whose grid differs from the current covariance's grid is refused at `run` with `ValueError` naming the block, both grids (`data_domain=...` of each) and the fix, before the residual is transformed (a WDM block on a time covariance never loads the backend) and with the ledger, RNG and covariance unchanged.
- [ ] A seed covariance (at `add`) or a returned covariance (at `run`) on another grid than its block's is refused with `ValueError` `<block>.<method> covariance is on grid ... but the block is registered on grid ...`, committing nothing; one on the block's grid is adopted and held there (`wheel.covariance.grid == block grid`).
- [ ] `grep -rn "NotImplementedError\|section 6\|DomainSelection\|_empty_channel" src tests` prints nothing; `pytest -q` reports `0 failed`, `0 xfailed`, `0 xpassed`, and the former xfails pass locally where they need no backend (`test_initial_block_state.py::...[True]`, `test_toy_translation.py::...[frequency]`, `test_gb_example_contract.py::...[time]`, the rewritten `test_translation_review.py` round trip).
- [ ] mypy, ruff check and ruff format clean; `tests/test_typing.py` passes with `assert_type(noise.copy(), Covariance)`.

**Verify:** `.venv/bin/python -m pytest -q` → `0 failed, 0 xfailed`; `grep -rn "NotImplementedError\|section 6\|DomainSelection\|_empty_channel" src tests` → nothing; `.venv/bin/python -m mypy && .venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks` → clean.

**Steps:**

- [ ] **Step 1: Replace `src/enchilada/translation.py`**

```python
"""One object-level interface for time, Fourier and WDM representations."""

from copy import deepcopy
from dataclasses import replace
from typing import overload

from enchilada._transforms import transform_channels
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import WDMGrid, resolve_wdm_grid


def _target_grid(
    target_domain: str,
    num_time_samples: int,
    source_wdm_grid: WDMGrid | None,
    num_frequency_divisions: int | None,
    num_time_divisions: int | None,
) -> WDMGrid | None:
    if target_domain not in L1Data.DOMAINS:
        raise ValueError(f"target_domain must be one of {L1Data.DOMAINS}")
    if target_domain != "wdm":
        if num_frequency_divisions is not None or num_time_divisions is not None:
            raise ValueError("division counts apply only to target_domain='wdm'")
        return None
    if num_frequency_divisions is None and num_time_divisions is None:
        if source_wdm_grid is not None:
            return source_wdm_grid
    return resolve_wdm_grid(
        num_time_samples,
        num_frequency_divisions=num_frequency_divisions,
        num_time_divisions=num_time_divisions,
    )


@overload
def transform(
    value: L1Data,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> L1Data: ...


@overload
def transform(
    value: BlockState,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> BlockState: ...


def transform(
    value: L1Data | BlockState,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> L1Data | BlockState:
    """Return an independent representation of data or of a block state.

    Args:
        value: Object to translate. A BlockState with a template needs
            reference_data because its grid is intentionally external; its
            covariance carries its own grid and is deep-copied unchanged.
        target_domain: "time", "frequency", or "wdm".
        num_frequency_divisions: WDM frequency divisions (Nf, yielding Nf+1 rows).
            Specify this or num_time_divisions on first conversion into WDM.
        num_time_divisions: WDM time divisions. The other count is derived from
            num_time_samples; both counts must be positive even integers. Both
            may be supplied if consistent. WDM-to-WDM defaults to its current grid.
        reference_data: Required for a BlockState with a template, as the
            L1Data on the template's source grid.

    Preserves observation metadata, orbit, published physical parameters and a
    block state's covariance. Signal arrays follow dt*rfft and the WDM backend's
    normalized-edge convention. A Covariance does not translate between domains:
    passing one raises TypeError; ``Covariance.copy`` makes an independent copy.
    WDM coefficient transforms require the separately installed local backend
    and Python >=3.13. Same-grid copies do not.
    """
    if target_domain not in L1Data.DOMAINS:
        raise ValueError(f"target_domain must be one of {L1Data.DOMAINS}")
    if target_domain != "wdm" and (
        num_frequency_divisions is not None or num_time_divisions is not None
    ):
        raise ValueError("division counts apply only to target_domain='wdm'")
    if reference_data is not None and not isinstance(reference_data, L1Data):
        raise TypeError("reference_data must be L1Data")
    if isinstance(value, L1Data):
        if value.data_domain == "wdm":
            # Validate against the stored grid before construction derives it anew.
            value._validate_channel_data_lengths()
        source = replace(value)  # Revalidate potentially edited input buffers.
        grid = _target_grid(
            target_domain,
            source.num_time_samples,
            source.wdm_grid,
            num_frequency_divisions,
            num_time_divisions,
        )
        converted = transform_channels(
            source.channel_data,
            source_domain=source.data_domain,
            target_domain=target_domain,
            num_time_samples=source.num_time_samples,
            sample_rate_hz=source.sample_rate_hz,
            source_wdm_grid=source.wdm_grid,
            target_wdm_grid=grid,
        )
        return replace(source, channel_data=converted, data_domain=target_domain)
    if isinstance(value, BlockState):
        # deepcopy restores the covariance read-only and unaliased on its own grid.
        source_state = deepcopy(value)
        source_state.__post_init__()
        signal = source_state.template
        if signal is not None:
            if reference_data is None:
                raise ValueError(
                    "a BlockState signal requires its source reference_data"
                )
            if reference_data.data_domain == "wdm":
                reference_data._validate_channel_data_lengths()
            source_reference = replace(reference_data)
            check_on_grid(
                signal,
                channel_names=source_reference.channel_names,
                num_time_samples=source_reference.num_time_samples,
                data_domain=source_reference.data_domain,
                wdm_grid=source_reference.wdm_grid,
                context_label="BlockState.template",
            )
            grid = _target_grid(
                target_domain,
                source_reference.num_time_samples,
                source_reference.wdm_grid,
                num_frequency_divisions,
                num_time_divisions,
            )
            signal = transform_channels(
                signal,
                source_domain=source_reference.data_domain,
                target_domain=target_domain,
                num_time_samples=source_reference.num_time_samples,
                sample_rate_hz=source_reference.sample_rate_hz,
                source_wdm_grid=source_reference.wdm_grid,
                target_wdm_grid=grid,
            )
        return replace(source_state, template=signal)
    if isinstance(value, Covariance):
        raise TypeError(
            "transform does not take a Covariance: it carries its grid and does "
            "not translate between domains; use copy() for an independent copy"
        )
    raise TypeError("transform expects L1Data or BlockState")
```

- [ ] **Step 2: Replace `src/enchilada/wheel.py`**

```python
"""Orchestrator-owned data, accepted physical estimates, and signal accounting."""

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace

import numpy as np

from enchilada._signal_sum import Signal, SignalSum
from enchilada.block import Block
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid
from enchilada.ledger import Ledger
from enchilada.translation import _target_grid, transform


class NoiseOverwrittenWarning(RuntimeWarning):
    """A second block replaces the current noise owner's covariance.

    Wheel keeps one covariance, not a sum of separate noise publications.
    Publish a combined covariance from a single noise block when modelling
    several components.
    """


class Wheel:
    """Run blocks while owning their accepted physical estimates and shared data.

    The pristine observation is copied at construction. A second L1Data stores
    the full working residual, rebuilt after every accepted update. For block i,
    the input is d - sum(j != i, T_j); the block's own signal remains in its input.
    An absent TDI signal contributes zero, including when a block previously
    returned a signal. Ledger holds complete BlockState snapshots. Public data and
    covariance access returns copies. The ledger property returns a live handle;
    indexing and snapshot() copy its contents. Treat orbit resources as immutable.

    Registration requires and adopts an explicit initial_block_state. It never
    calls model initialization or consumes random draws.
    Each cycle calls sample with those inputs and the current_block_state from
    the ledger. Sampling receives the supplied RNG and independent input snapshots.
    Failed calls/validation/copying leave the ledger and Wheel RNG unchanged.
    Blocks own their sampler state and diagnostics; Wheel does not roll back
    changes to those internals. Use separate block instances for separate campaigns.
    Earlier successful updates in a failed cycle remain committed. Exceptions in
    on_cycle_complete occur after that entire cycle has committed.

    `initial_covariance` supplies a Covariance on any grid of the same record.
    If omitted, covariance stays unset until a block supplies it. Blocks receive
    it separately from L1Data. Signals are stored on the observation grid; the
    covariance stays on the grid its owner published it on. A block registered
    with data_domain has its inputs and returned signal translated; it receives
    the covariance only when registered on the covariance's grid, and a block on
    another grid is refused at run with ValueError naming both grids. The block
    that owns the current covariance must return it on every call, on the grid
    of the data it received; other blocks can omit covariance without changing
    it. Delivering a noise model to a block on another grid is a separate layer.
    """

    def __init__(
        self,
        observed_data: L1Data,
        initial_covariance: Covariance | None = None,
        *,
        random_seed: int | None = None,
    ) -> None:
        """Initialize a campaign from observations and an optional covariance.

        Args:
            observed_data: L1 observations and shared run settings. The Wheel
                makes independent pristine and working copies.
            initial_covariance: Starting Covariance on any grid of the same
                record (sample count and rate) with the observations' channels.
                None leaves it unset until a block supplies it.
            random_seed: Seed for the campaign's NumPy generator. None requests
                nondeterministic initialization.
        """
        if not isinstance(observed_data, L1Data):
            raise TypeError("observed_data must be L1Data")
        for ch in observed_data.channel_names:
            if not np.isfinite(observed_data.channel_data[ch]).all():
                raise ValueError(
                    f"observed_data.channel_data[{ch!r}] has non-finite samples; "
                    "enchilada has no data-quality mask for gaps; "
                    "fill or trim them first"
                )
        if initial_covariance is not None and not isinstance(
            initial_covariance, Covariance
        ):
            raise TypeError("initial_covariance must be a Covariance or None")
        if initial_covariance is not None:
            # Unbound call: copy without trusting a subclass's overridden methods.
            initial_covariance = Covariance.copy(initial_covariance)
            if initial_covariance.channel_names != tuple(observed_data.channel_names):
                raise ValueError(
                    f"covariance channel_names {initial_covariance.channel_names} "
                    f"do not match the data's {tuple(observed_data.channel_names)}"
                )
            if not initial_covariance.grid.same_record(observed_data.grid):
                raise ValueError(
                    f"initial_covariance is on grid {initial_covariance.grid} but "
                    f"the observations are on grid {observed_data.grid}; a "
                    "covariance must describe the same record (sample count and "
                    "rate)"
                )
        self._observed_data = self._copy_data(observed_data)
        self._working_residual = self._copy_data(observed_data)
        self._covariance = initial_covariance
        self._noise_owner_block_name: str | None = None
        self._ledger = Ledger()
        self._blocks: list[tuple[str, Block]] = []
        self._block_indices: dict[str, int] = {}
        self._block_domains: dict[str, Grid] = {}
        self._signal_sum = SignalSum()
        self._rng = np.random.default_rng(random_seed)

    @staticmethod
    def _copy_data(data_to_copy: L1Data) -> L1Data:
        """Copy channel arrays, preserving immutable orbit data."""
        if data_to_copy.data_domain == "wdm":
            data_to_copy._validate_channel_data_lengths()
        return replace(
            data_to_copy,
            channel_data={
                ch: arr.copy() for ch, arr in data_to_copy.channel_data.items()
            },
        )

    @property
    def observed_data(self) -> L1Data:
        """A defensive copy of the pristine observation."""
        return self._copy_data(self._observed_data)

    @property
    def working_residual(self) -> L1Data:
        """A defensive copy of the current full residual."""
        return self._copy_data(self._working_residual)

    @property
    def covariance(self) -> Covariance | None:
        """The current covariance on its owner's grid, as an independent copy."""
        if self._covariance is None:
            return None
        # Stored objects passed Covariance.copy at adoption: restore unvalidated.
        return deepcopy(self._covariance)

    @property
    def ledger(self) -> Ledger:
        """Live state store; indexing copies entries, snapshot() copies all entries."""
        return self._ledger

    def add(
        self,
        block_to_register: Block,
        *,
        initial_block_state: BlockState,
        data_domain: str | None = None,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
    ) -> None:
        """Register a block from a complete, explicitly prepared starting state.

        block_to_register must have a unique name and implement sample.
        Failed registration leaves the campaign unchanged.

        initial_block_state supplies a starting template, covariance, and physical
        parameters, for example at an injected signal. Sampler continuation state
        is managed inside the block.
        Wheel copies and validates it without calling an initialization method or
        consuming random draws. The state is required and cannot be None.
        Its signal uses the selected block domain/grid, or the observation grid when no
        data_domain is selected. A seed covariance must be on the block's grid,
        like its template; it is adopted and held there.
        The caller must keep published estimates and block-local state consistent.
        This sets the starting state only; sample still runs normally.

        data_domain selects time, frequency or wdm for this block's residual,
        current state and published covariance. None keeps the observation grid.
        For WDM, specify either division count; the other follows from the
        observation length. Both counts must be even. An existing observation
        WDM grid is reused if neither count is supplied.
        """
        if not isinstance(initial_block_state, BlockState):
            raise TypeError("initial_block_state must be a BlockState")
        observed = self._observed_data
        if data_domain is None:
            if num_frequency_divisions is not None or num_time_divisions is not None:
                raise ValueError("WDM division counts require data_domain='wdm'")
            block_grid = observed.grid
        else:
            block_grid = Grid(
                data_domain,
                observed.num_time_samples,
                observed.sample_rate_hz,
                _target_grid(
                    data_domain,
                    observed.num_time_samples,
                    observed.wdm_grid,
                    num_frequency_divisions,
                    num_time_divisions,
                ),
            )
        block_name = getattr(block_to_register, "name", None)
        if not isinstance(block_name, str) or not block_name:
            raise ValueError(
                f"block name must be a non-empty string, got {block_name!r}"
            )
        if block_name in self._ledger:
            raise ValueError(f"block name {block_name!r} already registered")
        if not callable(getattr(block_to_register, "sample", None)):
            raise TypeError(
                f"block {block_name!r} does not implement sample; "
                "a Block needs name and sample(conditional_residual, "
                "covariance, current_block_state, *, rng)."
            )
        new_block_state: object = initial_block_state
        if block_grid != observed.grid and initial_block_state.template is not None:
            # Only the seed's grid is needed. Transforming observation
            # values here wastes work and can overflow independently of it.
            source_reference = replace(
                observed,
                channel_data=initial_block_state.template,
                data_domain=block_grid.data_domain,
            )
            check_on_grid(
                initial_block_state.template,
                channel_names=observed.channel_names,
                num_time_samples=observed.num_time_samples,
                data_domain=block_grid.data_domain,
                wdm_grid=block_grid.wdm_grid,
                context_label="initial_block_state.template",
            )
            new_block_state = self._canonical_state(new_block_state, source_reference)
        self._adopt(
            block_name,
            new_block_state,
            "initial_block_state",
            self._rng,
            block_to_register=block_to_register,
            block_grid=block_grid,
        )

    def _block_inputs(self, block_name: str) -> tuple[L1Data, Covariance | None]:
        """Conditional residual on the block's grid, and the covariance when on it."""
        block_grid = self._block_domains[block_name]
        if self._covariance is not None and self._covariance.grid != block_grid:
            raise ValueError(
                f"block {block_name!r} is registered on grid {block_grid} but the "
                f"current covariance is on grid {self._covariance.grid}; a "
                "covariance does not translate between grids. Register the block "
                "on the covariance's grid, or have the noise block publish on the "
                "block's grid."
            )
        conditional_residual = self.residual(exclude_block_name=block_name)
        if block_grid != self._observed_data.grid:
            conditional_residual = transform(
                conditional_residual,
                block_grid.data_domain,
                num_frequency_divisions=self._frequency_divisions(block_grid),
            )
        covariance = None if self._covariance is None else deepcopy(self._covariance)
        return conditional_residual, covariance

    @staticmethod
    def _frequency_divisions(grid: Grid) -> int | None:
        if grid.wdm_grid is None:
            return None
        return grid.wdm_grid.num_frequency_divisions

    def _canonical_state(self, state: object, source_reference: L1Data) -> object:
        if not isinstance(state, BlockState):
            return state  # _adopt supplies the block/method-specific error.
        return transform(
            state,
            self._observed_data.data_domain,
            reference_data=source_reference,
            num_frequency_divisions=self._frequency_divisions(self._observed_data.grid),
        )

    def run(
        self,
        num_cycles: int,
        on_cycle_complete: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Visit blocks in registration order; notify after each complete cycle.

        Cycle indices are local to this invocation. Multiple run calls continue
        the accepted ledger and RNG; run(0) does nothing.

        Args:
            num_cycles: Non-negative number of complete Gibbs cycles to attempt.
            on_cycle_complete: Optional callback receiving (cycle_index, wheel)
                after every block in a cycle has returned an accepted state.
                Indices start at zero for each run call. Partial cycles do not
                invoke the callback.
        """
        if (
            not isinstance(num_cycles, (int, np.integer))
            or isinstance(num_cycles, bool)
            or num_cycles < 0
        ):
            raise ValueError(
                f"num_cycles must be a non-negative integer, got {num_cycles!r}"
            )
        for cycle_index in range(num_cycles):
            for block_name, block in self._blocks:
                candidate_rng = deepcopy(self._rng)
                block_grid = self._block_domains[block_name]
                conditional_residual, covariance = self._block_inputs(block_name)
                current_block_state = self._ledger[block_name]
                translated = block_grid != self._observed_data.grid
                if translated:
                    current_block_state = transform(
                        current_block_state,
                        block_grid.data_domain,
                        reference_data=self._observed_data,
                        num_frequency_divisions=self._frequency_divisions(block_grid),
                    )
                new_block_state: object = block.sample(
                    conditional_residual=conditional_residual,
                    covariance=covariance,
                    current_block_state=current_block_state,
                    rng=candidate_rng,
                )
                if translated:
                    new_block_state = self._canonical_state(
                        new_block_state, conditional_residual
                    )
                self._adopt(
                    block_name,
                    new_block_state,
                    "sample",
                    candidate_rng,
                    block_grid=block_grid,
                )
            if on_cycle_complete is not None:
                on_cycle_complete(cycle_index, self)

    def _form_residual(self, signal_sum: Signal) -> L1Data:
        """Subtract an aggregate signal from pristine observations."""
        channel_data = {}
        for ch, observed_channel_values in self._observed_data.channel_data.items():
            if signal_sum is None:
                residual_channel_values = observed_channel_values.copy()
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_channel_values = observed_channel_values - signal_sum[ch]
            if not np.isfinite(residual_channel_values).all():
                raise ValueError(f"residual in channel {ch!r} became non-finite")
            channel_data[ch] = residual_channel_values
        return replace(self._observed_data, channel_data=channel_data)

    def residual(self, exclude_block_name: str | None = None) -> L1Data:
        """Return a copied full residual or a block's conditional residual.

        exclude_block_name identifies a registered block whose signal is left
        in the observations. None subtracts every block's signal.
        """
        if exclude_block_name is None:
            return self.working_residual
        self._require_block_name(exclude_block_name)
        return self._form_residual(
            self._signal_sum.excluding(self._block_indices[exclude_block_name])
        )

    def contribution(self, block_name: str) -> dict[str, np.ndarray]:
        """Copy the registered block's summed TDI signal.

        block_name must identify a registered block. An absent contribution
        returns fresh zero arrays on the observation grid.
        """
        self._require_block_name(block_name)
        signal = self._ledger._states_by_block_name[block_name].template
        if signal is None:
            return {
                ch: np.zeros_like(arr)
                for ch, arr in self._observed_data.channel_data.items()
            }
        return {ch: arr.copy() for ch, arr in signal.items()}

    def _require_block_name(self, block_name: str) -> None:
        if block_name not in self._ledger:
            raise ValueError(
                f"unknown block {block_name!r}; registered: {sorted(self._ledger)}"
            )

    def _adopt(
        self,
        block_name: str,
        new_block_state: object,
        source_method_name: str,
        candidate_rng: np.random.Generator,
        *,
        block_to_register: Block | None = None,
        block_grid: Grid,
    ) -> None:
        """Validate a candidate state and RNG before committing campaign state.

        source_method_name identifies the block call or initial state in diagnostics.
        block_to_register is supplied only when adopting a new block's initial state.
        block_grid is the grid the block is registered on; a published covariance
        must be on it.
        """
        if not isinstance(new_block_state, BlockState):
            raise TypeError(
                f"{block_name}.{source_method_name} must return a BlockState, got "
                f"{type(new_block_state).__name__}"
            )
        # Snapshot BEFORE validation so subsequent use only touches owned state.
        candidate_block_state = deepcopy(new_block_state)
        candidate_block_state.__post_init__()
        signal = candidate_block_state.template
        if signal is not None:
            check_on_grid(
                signal,
                channel_names=self._observed_data.channel_names,
                num_time_samples=self._observed_data.num_time_samples,
                data_domain=self._observed_data.data_domain,
                wdm_grid=self._observed_data.wdm_grid,
                context_label=f"{block_name}.{source_method_name} template",
            )
            for ch, arr in signal.items():
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"{block_name}.{source_method_name} returned non-finite "
                        f"samples in channel {ch!r}"
                    )
        candidate_covariance = self._covariance
        candidate_noise_owner_block_name = self._noise_owner_block_name
        if (
            block_name == candidate_noise_owner_block_name
            and candidate_block_state.covariance is None
        ):
            raise ValueError(
                f"{block_name}.{source_method_name} must return its complete "
                "covariance because it owns the current covariance"
            )
        if candidate_block_state.covariance is not None:
            # Revalidate: callers can edit arrays inside a frozen dataclass.
            # Unbound call: copy without trusting a subclass's overridden methods.
            candidate_covariance = Covariance.copy(candidate_block_state.covariance)
            if candidate_covariance.channel_names != tuple(
                self._observed_data.channel_names
            ):
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance channel_names "
                    f"{candidate_covariance.channel_names} do not match the data's "
                    f"{tuple(self._observed_data.channel_names)}"
                )
            if candidate_covariance.grid != block_grid:
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance is on grid "
                    f"{candidate_covariance.grid} but the block is registered on "
                    f"grid {block_grid}; publish the covariance on the grid of the "
                    "data the block received"
                )
            candidate_block_state = replace(
                candidate_block_state, covariance=candidate_covariance
            )
            candidate_noise_owner_block_name = block_name
        candidate_block_states = {
            **self._ledger._states_by_block_name,
            block_name: candidate_block_state,
        }
        block_index = (
            len(self._blocks)
            if block_to_register is not None
            else self._block_indices[block_name]
        )
        candidate_signal_sum = self._signal_sum.with_signal(block_index, signal)
        candidate_residual = self._form_residual(candidate_signal_sum.total)
        candidate_block_indices = (
            {**self._block_indices, block_name: block_index}
            if block_to_register is not None
            else self._block_indices
        )
        next_blocks = (
            self._blocks
            if block_to_register is None
            else [*self._blocks, (block_name, block_to_register)]
        )
        candidate_block_domains = (
            self._block_domains
            if block_to_register is None
            else {**self._block_domains, block_name: block_grid}
        )
        next_rng = deepcopy(candidate_rng)
        if (
            candidate_noise_owner_block_name != self._noise_owner_block_name
            and self._noise_owner_block_name is not None
        ):
            warnings.warn(
                f"{block_name}.{source_method_name} replaced the noise model that "
                f"{self._noise_owner_block_name!r} owns; "
                "Wheel does not combine noise models. "
                "Publish one combined model from a single noise block.",
                NoiseOverwrittenWarning,
                stacklevel=3,
            )
        # All potentially failing work, including warnings-as-errors, is done.
        self._ledger._states_by_block_name = candidate_block_states
        self._working_residual = candidate_residual
        self._signal_sum = candidate_signal_sum
        self._block_indices = candidate_block_indices
        self._block_domains = candidate_block_domains
        self._covariance, self._noise_owner_block_name = (
            candidate_covariance,
            candidate_noise_owner_block_name,
        )
        self._rng = next_rng
        self._blocks = next_blocks
```

Run `.venv/bin/python -m ruff format src/enchilada/translation.py src/enchilada/wheel.py` and `.venv/bin/python -m pytest -q`.
Expected now: the seam tests fail (they expect `NotImplementedError`), the seven xfails report `XPASS(strict)` failures where they need no backend, everything else passes. Step 3 fixes the tests.

- [ ] **Step 3: `tests/test_translation.py`**

1. In `test_same_domain_transform_is_an_independent_snapshot`, delete the last three lines (`covariance = white_covariance(data, 2.0)`, `copied_covariance = transform(...)`, `assert not np.shares_memory(...)`).
2. Replace `test_noise_only_state_needs_reference_data_and_parameters_only_does_not` with:

```python
def test_noise_only_state_translation_carries_its_covariance_without_a_reference():
    data = observations()
    covariance = white_covariance(data, 1.0)
    result = BlockState(covariance=covariance, parameters={"variance": 1.0})
    converted = transform(result, "frequency")
    assert converted.template is None
    assert converted.parameters == {"variance": 1.0}
    assert converted.covariance is not covariance
    assert converted.covariance.grid == data.grid  # a covariance keeps its grid
    np.testing.assert_array_equal(converted.covariance.A, 1.0)
    assert not np.shares_memory(converted.covariance.blocks, covariance.blocks)
    assert not converted.covariance.blocks.flags.writeable
    state = BlockState(parameters={"amplitude": 1.0})
    copied = transform(state, "frequency")
    assert copied is not state
    assert copied.parameters == state.parameters


def test_block_state_with_template_and_covariance_translates_only_the_template():
    data = observations()
    covariance = white_covariance(data, 2.0)
    original = data.block_state(data.channel_data, covariance=covariance)
    converted = transform(original, "frequency", reference_data=data)
    spectrum = transform(data, "frequency")
    for name in data.channel_names:
        np.testing.assert_allclose(
            converted.template[name], spectrum.channel_data[name]
        )
    assert converted.covariance.grid == covariance.grid
    np.testing.assert_array_equal(converted.covariance.blocks, covariance.blocks)
    assert not np.shares_memory(converted.covariance.blocks, covariance.blocks)


def test_transform_rejects_a_covariance():
    data = observations()
    covariance = white_covariance(data, 2.0)
    with pytest.raises(TypeError, match="does not take a Covariance.*copy"):
        transform(covariance, "time")
    with pytest.raises(TypeError, match="does not take a Covariance"):
        transform(covariance, "frequency", reference_data=data)
    with pytest.raises(TypeError, match="expects L1Data or BlockState"):
        transform({"A": np.zeros(4)}, "time")
```

3. In `test_conversion_rejects_missing_or_mismatched_reference`, delete the final `with pytest.raises(ValueError, match="grid"):` block that passes `white_covariance(data, 1.0)` to `transform` (six lines).
4. Delete the five seam tests `test_covariance_same_domain_transform_is_an_independent_copy`, `test_covariance_cross_domain_transform_is_the_seam`, `test_covariance_seam_copies_through_covariance_methods`, `test_covariance_wdm_same_grid_copies_and_regrid_is_the_seam_without_backend` and `test_covariance_wdm_regrid_is_the_seam_but_same_grid_copies`. Keep `test_white_noise_wdm_coefficient_variance_matches_white_covariance` (Task 2).
5. Run `grep -n "expects L1Data" tests/*.py` and change any remaining `"expects L1Data, Covariance, or BlockState"` match to `"expects L1Data or BlockState"`. Drop imports ruff reports unused (`Covariance` is likely one).

- [ ] **Step 4: `tests/test_wheel_translation.py`**

Module docstring:

```python
"""Wheel keeps accepted signals on its observation grid.

A covariance stays on the grid its owner published it on. A block registered
on that grid receives it; a block on another grid is refused at run.
"""
```

Add `from conftest import white_covariance` if Task 2 did not (it did), and this helper after `observations()`:

```python
def block_reference(data, domain):
    """An L1Data on the block's grid and the matching Wheel.add options.

    Built by hand, so no test here needs the WDM backend to build a covariance.
    """
    if domain == "frequency":
        return data.to_frequency(), {}
    grid = WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
    zeros = {name: np.zeros(grid.array_shape) for name in data.channel_names}
    reference = replace(data, channel_data=zeros, data_domain="wdm")
    return reference, {"num_frequency_divisions": 8}
```

On `test_wdm_noise_publication_is_usable_in_every_domain`: delete the `@pytest.mark.xfail(...)` decorator (keep `@requires_wdm`), rename it `test_wdm_noise_publication_is_held_on_the_wdm_grid`, and append after `wheel.run(2)`:

```python
    assert wheel.covariance.grid == wdm_data.grid
    np.testing.assert_array_equal(wheel.covariance.A, variances)
```

Replace the three seam tests (`test_block_in_another_domain_reaches_the_covariance_seam_at_run`, `test_covariance_returned_in_another_domain_reaches_the_seam_at_run`, `test_noise_only_seed_in_another_domain_reaches_the_seam_at_registration`, lines 269–357) with:

```python
def test_block_on_another_grid_than_the_covariance_is_refused_at_run():
    """The Wheel does not translate a covariance: a block registered on another
    grid than the current covariance is refused before sample is called."""

    class NeverSampled:
        name = "frequency_block"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            raise AssertionError("the Wheel must refuse before sample is called")

    data = observations()
    wheel = Wheel(data, white_covariance(data, 0.25), random_seed=3)
    wheel.add(NeverSampled(), initial_block_state=BlockState(), data_domain="frequency")
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(ValueError, match="does not translate between grids") as info:
        wheel.run(1)
    message = str(info.value)
    assert "'frequency_block'" in message
    assert "data_domain='frequency'" in message and "data_domain='time'" in message
    assert wheel._rng.bit_generator.state == rng_before
    assert wheel.covariance.grid == data.grid
    np.testing.assert_array_equal(wheel.covariance.A, 0.25)


def test_wdm_block_on_a_time_covariance_is_refused_before_the_backend(monkeypatch):
    def unexpected_backend():
        raise AssertionError("the grid check must come before any WDM transform")

    monkeypatch.setattr("enchilada._transforms._load_wdm", unexpected_backend)
    data = observations()
    wheel = Wheel(data, white_covariance(data, 1.0))
    wheel.add(
        EchoBlock("wdm_block"),
        initial_block_state=BlockState(),
        data_domain="wdm",
        num_frequency_divisions=8,
    )
    with pytest.raises(ValueError, match="does not translate between grids"):
        wheel.run(1)
    assert wheel.covariance.grid == data.grid


def test_covariance_returned_on_another_grid_than_the_block_is_refused():
    """A block must publish on the grid of the data it received."""

    class TimeNoiseFromFrequency:
        name = "frequency_noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            return BlockState(
                covariance=white_covariance(transform(conditional_residual, "time"), 1.0),
                parameters={"level": 1.0},
            )

    data = observations()
    wheel = Wheel(data, random_seed=3)
    wheel.add(
        TimeNoiseFromFrequency(),
        initial_block_state=BlockState(parameters={"level": 0.0}),
        data_domain="frequency",
    )
    rng_before = wheel._rng.bit_generator.state
    with pytest.raises(ValueError, match="frequency_noise.sample covariance is on grid"):
        wheel.run(1)
    assert wheel.covariance is None
    assert wheel._rng.bit_generator.state == rng_before
    entry = wheel.ledger["frequency_noise"]
    assert entry.parameters == {"level": 0.0}
    assert entry.template is None and entry.covariance is None
    np.testing.assert_array_equal(
        wheel.residual().channel_data["A"], data.channel_data["A"]
    )


def test_covariance_published_on_the_block_grid_is_held_there():
    class FrequencyNoise:
        name = "frequency_noise"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            assert conditional_residual.data_domain == "frequency"
            level = 1.0 + abs(rng.normal())
            return BlockState(
                covariance=Covariance.like(
                    conditional_residual,
                    {
                        name: np.full(values.shape, level)
                        for name, values in conditional_residual.channel_data.items()
                    },
                ),
                parameters={"level": level},
            )

    data = observations()
    wheel = Wheel(data, random_seed=3)
    wheel.add(
        FrequencyNoise(),
        initial_block_state=BlockState(parameters={"level": 0.0}),
        data_domain="frequency",
    )
    wheel.run(2)
    assert wheel.covariance.grid == data.to_frequency().grid
    level = wheel.ledger["frequency_noise"].parameters["level"]
    np.testing.assert_array_equal(wheel.covariance.A, level)
    assert wheel.working_residual.data_domain == "time"


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_seed_on_the_block_grid_is_adopted_there(domain):
    """A seed covariance is on the block's grid; the Wheel keeps it there."""
    data = observations()
    reference, options = block_reference(data, domain)
    variances = np.where(reference.grid.active_mask, 1.0, 0.0)
    seed = BlockState(
        covariance=Covariance.like(
            reference, {name: variances for name in data.channel_names}
        )
    )
    wheel = Wheel(data)
    wheel.add(EchoBlock("noise"), initial_block_state=seed, data_domain=domain, **options)
    assert wheel.covariance.grid == reference.grid
    np.testing.assert_array_equal(wheel.covariance.A, variances)
    assert wheel.ledger["noise"].covariance.grid == reference.grid


@pytest.mark.parametrize("domain", ["frequency", "wdm"])
def test_noise_only_seed_on_another_grid_than_the_block_is_refused_at_registration(
    domain,
):
    data = observations()
    _, options = block_reference(data, domain)
    seed = BlockState(covariance=white_covariance(data, 1.0))  # on the time grid
    wheel = Wheel(data)
    with pytest.raises(
        ValueError, match="noise.initial_block_state covariance is on grid"
    ):
        wheel.add(EchoBlock("noise"), initial_block_state=seed, data_domain=domain, **options)
    assert len(wheel.ledger) == 0
    assert wheel.covariance is None
```

Run `.venv/bin/python -m ruff format tests/test_wheel_translation.py` afterwards.

- [ ] **Step 5: `tests/test_translation_review.py`**

Replace the first test (its `skipif`, `xfail` and body, lines 12–33) with a round trip that needs no backend:

```python
def test_noise_only_wdm_state_translation_carries_its_covariance(observed):
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    wdm_reference = replace(
        observed,
        channel_data={
            name: np.zeros(grid.array_shape) for name in observed.channel_names
        },
        data_domain="wdm",
    )
    variances = np.where(grid.active_mask, 1.0, 0.0)
    covariance = Covariance.like(
        wdm_reference, {name: variances for name in observed.channel_names}
    )
    original = BlockState(covariance=covariance, parameters={"variance": 1.0})
    converted = transform(original, "wdm", reference_data=wdm_reference)
    assert converted.parameters == original.parameters
    assert converted.template is None
    assert converted.covariance is not covariance
    assert converted.covariance.grid == wdm_reference.grid
    np.testing.assert_array_equal(converted.covariance.A, covariance.A)
    assert not np.shares_memory(converted.covariance.blocks, covariance.blocks)
```

Imports become `from dataclasses import replace` and `from enchilada import BlockState, Covariance, WDMGrid, Wheel` (`importlib.util` stays: the last test's `skipif` uses it). Rename `test_same_grid_block_receives_the_covariance_without_the_seam` to `test_same_grid_block_receives_the_covariance_without_transform` with docstring `"""An explicit domain equal to the observations' hands the covariance over as a deep copy; nothing goes through transform()."""`.

- [ ] **Step 6: The remaining xfails**

`tests/test_initial_block_state.py`: replace the `with_noise` parametrize (lines 264–275) with `@pytest.mark.parametrize("with_noise", [False, True])`; after `np.testing.assert_allclose(wheel.covariance.A, 100.0)` add `assert wheel.covariance.grid == spectrum.grid`; replace the comment `# Each kind has its own error, so the seam's NotImplementedError (a` / `# RuntimeError) cannot pass for an invalid seed being rejected.` with `# Each kind has its own error, so one failure cannot pass for another.`

`tests/test_gb_example_contract.py`: delete the `@pytest.mark.xfail(...)` decorator (lines 138–144), rename the test `test_gb_frequency_registration_reads_its_native_frequency_covariance`, replace

```python
        initial_covariance=transform(
            noise, storage_domain, reference_data=data, **options
        ),
```

with `initial_covariance=noise,`, and add after `wheel.run(1)`:

```python
    assert wheel.covariance.grid == data.grid
```

`tests/test_toy_translation.py`: delete the `@pytest.mark.xfail(...)` decorator (lines 22–27); keep the `importorskip` for WDM.

`tests/test_typing.py` CONSUMER: `assert_type(transform(noise, "time", reference_data=r), Covariance)` → `assert_type(noise.copy(), Covariance)`.

- [ ] **Step 7: Verify and commit**

Run: `grep -rn "NotImplementedError\|section 6\|DomainSelection\|_empty_channel" src tests`
Expected: no output.

Run: `.venv/bin/python -m pytest -q`
Expected: `0 failed`, no `xfailed`/`xpassed` in the summary; skipped = the WDM-marked tests only.

Run: `.venv/bin/python -m mypy && .venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks`
Expected: clean.

```bash
git add src/enchilada/translation.py src/enchilada/wheel.py tests/test_translation.py tests/test_wheel_translation.py tests/test_translation_review.py tests/test_initial_block_state.py tests/test_gb_example_contract.py tests/test_toy_translation.py tests/test_typing.py
git commit -m "Wheel holds one native covariance on its owner's grid; transform refuses a Covariance; xfails pass

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

### Task 6: README, CHANGELOG 0.4.0, version, coverage config, CI file list and the remaining seam prose

**Goal:** Document the new `Covariance` (one form, `Grid`, `like`, the §2.6 units beside the Fourier convention), state that a covariance does not translate and that a block reads it on its own grid, record 0.4.0 in the CHANGELOG and `pyproject.toml`, drop the `NotImplementedError` coverage exclusion and the deleted test file from `wdm.yml`, and remove the last "translation layer / section 6" prose from the example script and the translation notebook.

**Files:**
- Modify: `README.md` (lines 244–249, 260–337, 466–474, 535, 648–651, 665–667, 674)
- Modify: `CHANGELOG.md` (new `## [0.4.0] — Unreleased` section above `## [0.3.0] — Unreleased`)
- Modify: `pyproject.toml:3` (version) and `:112` (`exclude_also`)
- Modify: `.github/workflows/wdm.yml` (drop `tests/test_covariance_translation.py`)
- Modify: `examples/domain_translation.py:37-38`, `examples/domain_translation.ipynb` cells `translation-11`, `translation-14`, `translation-17`, `translation-19`

**Acceptance Criteria:**
- [ ] `grep -rn "NotImplementedError\|section 6\|translation layer\|from_psd\|from_variance\|noise_psd\|noise_variance\|form=\"\|psd(frequencies" README.md src tests examples benchmarks pyproject.toml .github/workflows/wdm.yml` prints nothing (CHANGELOG history is exempt).
- [ ] README's covariance section shows `Covariance.like`, states the units `C_k = (Tobs / 2) S(f_k)` under `X = dt * rfft(x)`, white noise `N dt² σ²` per bin and `dt σ²` per active WDM pixel, names `Grid`/`noise.grid`, `pair`, `blocks`, `in_basis`, and says `check_compatible` requires the same grid and channel order; the translation section says a covariance does not translate and a block reads it on its own grid; Known limitations has "No covariance delivery across grids yet"; Status reads 0.4.0.
- [ ] `tests/test_docs.py` passes (every README python block parses, keyword names are real, exports are real).
- [ ] `pyproject.toml` version is `0.4.0` and `[tool.coverage.report]` has no `exclude_also`; `wdm.yml` no longer lists `tests/test_covariance_translation.py`.
- [ ] Final verification: `pytest -q --cov=enchilada` → `0 failed`, `0 xfailed`, total coverage ≥ 95 %; ruff check/format and mypy clean; `MPLBACKEND=Agg` notebook tests pass.

**Verify:** `MPLBACKEND=Agg .venv/bin/python -m pytest -q --cov=enchilada --cov-report=term` → `0 failed`, coverage ≥ 95 %; the grep above → nothing; `.venv/bin/python -m mypy && .venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks` → clean.

**Steps:**

- [ ] **Step 1: README block-registration paragraph (lines 244–249)**

Replace `An optional covariance follows the normal ownership rules and, until covariance translation lands, must be on the observation grid; a block registered with another domain or WDM grid cannot carry one (see [Time, frequency, and WDM translation](#time-frequency-and-wdm-translation)).` with:

```markdown
An optional covariance follows the normal ownership rules and must be on the
block's own grid, like its template; Wheel holds it on that grid (see
[Time, frequency, and WDM translation](#time-frequency-and-wdm-translation)).
```

- [ ] **Step 2: README covariance section (lines 260–337)**

Replace everything from `` `L1Data` holds channel arrays and observation conventions. Supply noise as an `` through the paragraph ending `[Time, frequency, and WDM translation](#time-frequency-and-wdm-translation).` (just before `Wheel uses descriptive arguments at each entry point:`) with:

````markdown
`L1Data` holds channel arrays and observation conventions. Supply noise as an
explicit `Covariance` when constructing Wheel:

```python
import numpy as np
from enchilada import Covariance, Wheel

noise = Covariance.like(observed, {"A": np.full(observed.N, 0.25)})
wheel = Wheel(
    observed_data=observed,
    initial_covariance=noise,
    random_seed=42,
)
```

With `initial_covariance=None`, Wheel starts without a covariance; a block can
publish one. `Covariance.like(data, pairs)` is the one constructor: it takes the
`L1Data` whose grid and channel order the covariance adopts and a mapping from
channel names (`"A"`), concatenated pairs (`"AE"`) or `(first, second)` tuples
to arrays of the data's channel shape. Every channel's own entry is required; an
absent cross term is zero; two keys naming one pair raise `ValueError`.

A covariance holds variances in the squared units of the data's coordinates on
its grid: `x_n**2` in time, `(dt * rfft(x))**2` in frequency, squared WDM
coefficients in WDM. The package converts neither to nor from a one-sided PSD;
that is a noise block's business. The one convention a block needs follows from
`X = dt * rfft(x)`: stationary noise with one-sided PSD `S` has bin variance
`C_k = (Tobs / 2) S(f_k)` at every rfft bin, `f = 0` and Nyquist included, with
`Tobs = N / fs`. White noise of variance `sigma**2` per sample therefore has
`C_k = N * dt**2 * sigma**2` at every bin and `dt * sigma**2` at every active
WDM pixel (zero at the structurally inactive edge pixels). A block that
evaluates a PSD model builds its covariance with
`Covariance.like(spectrum, {ch: spectrum.Tobs / 2 * model.psd(f) for ch in spectrum.channel_names})`
and reads a channel's bin variances back with `noise.pair("A", "A")`.

Signal-only, noise-only, and joint signal/noise models all implement `Block`.
A noise-only block publishes covariance directly:

```python
from enchilada import BlockState

new_block_state = BlockState(
    covariance=covariance,
    parameters={"sigma": sigma},
)
```

Publishing covariance makes that block the current covariance owner. The owner
must return its full covariance on every subsequent call, even when unchanged;
omitting it raises `ValueError`. Other blocks may omit `covariance` to keep the
Wheel's current covariance. Joint signal/noise states publish a template and a
covariance in one `BlockState`; validation and adoption are atomic.
`state.with_covariance(covariance)` builds a new state with that covariance.

A `Covariance` is read through its channels. `noise.A`, `noise.E`, `noise.T`
are each channel's own variances; `noise.AE`, `noise.ET`, `noise.AT` are the
cross terms (or `noise.pair("A", "E")` for names that do not concatenate
cleanly), with `noise.EA` the complex conjugate of `noise.AE`; an absent
cross term reads as zero. When the channels are exactly X, Y, Z the names
A, E, T and their pairs are derived on demand with the orthonormal LDC
combination A = (Z − X)/√2, E = (X − 2Y + Z)/√6, T = (X + Y + Z)/√3, and
X, Y, Z are derived the same way from A, E, T channels.
`noise.in_basis("AET")` returns the whole derived object, computed on each
call. Derivation raises `ValueError` when the rotated covariance is not
numerically positive definite, as happens when one channel's variance is
below float64 roundoff of the others (T ≪ A at the lowest LISA frequencies).

Internally the object is one array, `noise.blocks`, of shape
`(K, K) + channel_shape` for `K` channels: `blocks[i, j]` is the covariance
between channels `i` and `j` at every point and `blocks[j, i]` its conjugate,
so there is one value per point (`(N,)` time, `(N//2 + 1,)` frequency,
`(num_frequency_divisions + 1, num_time_divisions)` WDM). Construction checks
finite entries, exact Hermitian symmetry, real values where the domain requires
them, positive variances at every active point, zero blocks at inactive WDM
pixels and positive definiteness at every active point; the array is stored
read-only.

The object carries its grid: `noise.grid` is a `Grid` (domain, sample count,
sample rate and WDM grid), the same value as `observed.grid`.
`noise.check_compatible(observed)` requires equal channel order and
`observed.grid == noise.grid`. Wheel validates the initial covariance and every
covariance a block publishes, and keeps it on the grid it was published on; see
[Time, frequency, and WDM translation](#time-frequency-and-wdm-translation).
````

- [ ] **Step 3: README translation section (lines 399 and 466–474)**

Line 401 `Use one wrapper for observations, covariance, and block states:` → `Use one wrapper for observations and block states:`.

Replace the paragraph starting `Covariance translation between representations is not built in.` (lines 466–474) with:

```markdown
A covariance does not translate between representations: it carries its grid,
`transform` refuses one with `TypeError`, and `noise.copy()` makes an
independent copy. A block state's covariance passes through `transform`
unchanged. Wheel keeps the current covariance on the grid its owner published
it on and hands it to blocks registered on that grid; a block registered on
another grid is refused at `run` with `ValueError` naming both grids, and a
seed or returned covariance on another grid than its block's is refused at
`add` or `run`. In a campaign that carries covariance, register the noise owner
and every block that reads the covariance on one grid.
```

- [ ] **Step 4: README convention bullet, Known limitations, Scaling, Status**

Line 535: `the campaign's Fourier convention, `X(f) = dt * rfft(x)`, also used by` / `` `Covariance.from_psd`. They return independent channel arrays even when `` → `the campaign's Fourier convention, `X(f) = dt * rfft(x)`, under which a` / `covariance bin holds `(Tobs / 2) S(f)`. They return independent channel arrays even when`.

Known limitations bullet (lines 648–651) becomes:

```markdown
- **No covariance delivery across grids yet.** A `Covariance` lives on one
  grid and is not translated, so a campaign that carries covariance keeps the
  noise owner and every block that reads it on one grid; a block on another
  grid is refused at `run`. Delivering a noise model to blocks on several grids
  is the noise-components layer that follows domainia 3.0.
```

Scaling (lines 665–667): `Defensive state and covariance copies still cost memory and work. Internal` / `copies of validated covariance skip revalidation; constructors and new` / `covariance publications are still validated.` → `Defensive state and covariance copies still cost memory and work. Handoffs of a` / `validated covariance are deep copies (one array copy, no revalidation);` / `constructors, `copy()` and new covariance publications are validated.`

Status (line 674): `0.3.0 — in development, with block-owned samplers and` → `0.4.0 — in development, with block-owned samplers and`.

- [ ] **Step 5: CHANGELOG**

Insert above `## [0.3.0] — Unreleased`:

```markdown
## [0.4.0] — Unreleased

### Added

- `Grid`: a frozen value (`data_domain`, `num_time_samples`, `sample_rate_hz`,
  `wdm_grid`) with `channel_shape`, `active_mask`, `observation_duration_s` and
  `same_record`; `L1Data.grid` returns the data's grid and `enchilada.domains`
  owns `DOMAINS`.

### Changed

- `Covariance` stores one stacked `(K, K, *channel_shape)` Hermitian array,
  `blocks`, and carries its `grid`. `Covariance.like(data, pairs)` is the one
  constructor and captures the data's grid and channel order; `pair`, the
  channel attributes and `in_basis` read it. Construction validates exact
  Hermitian symmetry, realness per domain, positive variances, zero blocks at
  inactive WDM pixels and positive definiteness with one batched Cholesky.
  `copy()` revalidates; `deepcopy` and pickle restore without validating.
- Wheel holds one covariance on the grid its owner published it on and hands a
  deep copy to blocks registered on that grid. A block on another grid is
  refused at `run` with `ValueError` naming both grids; a seed or returned
  covariance on another grid than its block's is refused at `add` or `run`.
- `transform` takes `L1Data` or `BlockState` only; a `BlockState` carries its
  covariance through unchanged.
- The examples and notebooks build covariances with `like` from the convention
  `C = (Tobs / 2) S(f)` under `X = dt * rfft(x)` and read them with `pair`.

### Removed

- The full form, `form`, the derived-basis cache and the pickle hooks.
- `Covariance.from_psd`, `from_variance`, `noise_psd`, `noise_variance` and the
  `psd(frequencies[, channel])` model protocol: a noise block computes its
  variances on the grid of the data it received and passes them to `like`.
- The covariance overload of `transform` with its `reference_data`, the
  covariance translation seam and every `NotImplementedError` in the package.
```

- [ ] **Step 6: `pyproject.toml`, `wdm.yml`, example prose**

`pyproject.toml`: `version = "0.3.0"` → `version = "0.4.0"`; delete the line `exclude_also = ["raise NotImplementedError"]` and the two comment lines above it (`# coverage's own defaults ...` / `# only add what they miss.`).

`.github/workflows/wdm.yml`: delete the line `            tests/test_covariance_translation.py \`.

`examples/domain_translation.py` lines 37–38: `# Covariance translation across domains arrives with the translation` / `# layer (design spec section 6); this walkthrough runs without one.` → `# This walkthrough carries no covariance: it shows data and block-state` / `# translation only.`

Notebook prose, with the same script shape as Task 3 (`load`/`replace`/`save`, run from the repository root):

```python
nb = load("examples/domain_translation.ipynb")
replace(
    nb,
    "translation-11",
    "## 4. Covariance translation\n"
    "\n"
    "Covariance translation between representations is the job of a separate\n"
    "translation layer; `transform` on a `Covariance` raises `NotImplementedError`\n"
    "across domains until it is installed, so this walkthrough translates data and\n"
    "block states only.\n",
    "## 4. A covariance stays on its grid\n"
    "\n"
    "A `Covariance` carries its `grid` and is not translated: `transform` refuses\n"
    "one, and a block reads the covariance on the grid it was published on. This\n"
    "walkthrough translates data and block states only.\n",
)
replace(
    nb,
    "translation-14",
    "`parameters` (a state carrying a `covariance` cannot cross domains until\n"
    "covariance translation is available, so this example omits it). Translation\n",
    "`parameters` (a covariance passes through unchanged on its own grid; this\n"
    "example omits it). Translation\n",
)
replace(
    nb,
    "translation-17",
    "        # Covariance translation across domains arrives with the translation\n"
    "        # layer (design spec section 6); this walkthrough runs without one.\n",
    "        # This campaign carries no covariance.\n",
)
replace(
    nb,
    "translation-19",
    "This campaign has no covariance, because translating one across domains needs\n"
    "the covariance translation layer. A block that owns the noise model must return\n"
    "its complete covariance on every call. Omitting `data_domain` in `Wheel.add`\n",
    "This campaign has no covariance; one would stay on the grid its owner published\n"
    "it on, and only blocks registered on that grid would receive it. A block that\n"
    "owns the noise model must return its complete covariance on every call.\n"
    "Omitting `data_domain` in `Wheel.add`\n",
)
replace(
    nb,
    "translation-19",
    "and the status of covariance translation, or [the Python script](domain_translation.py) for a\n",
    "and the covariance rule, or [the Python script](domain_translation.py) for a\n",
)
save("examples/domain_translation.ipynb", nb)
```

- [ ] **Step 7: Final verification and commit**

Run: `grep -rn "NotImplementedError\|section 6\|translation layer\|from_psd\|from_variance\|noise_psd\|noise_variance\|form=\"\|psd(frequencies" README.md src tests examples benchmarks pyproject.toml .github/workflows/wdm.yml`
Expected: no output.

Run: `MPLBACKEND=Agg .venv/bin/python -m pytest -q --cov=enchilada --cov-report=term`
Expected: `0 failed`, no xfail/xpass, `TOTAL` ≥ 95 %.

Run: `.venv/bin/python -m mypy && .venv/bin/python -m ruff check src tests examples benchmarks && .venv/bin/python -m ruff format --check src tests examples benchmarks`
Expected: clean.

```bash
git add README.md CHANGELOG.md pyproject.toml .github/workflows/wdm.yml examples/domain_translation.py examples/domain_translation.ipynb
git commit -m "Docs and 0.4.0: one covariance form on a Grid, units beside the Fourier convention, no covariance translation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes

- Spec coverage: §1 → Task 1; §2.1–2.5, 2.7 → Task 4; §2.6 → Tasks 2 (`white_covariance`, WDM statistics test), 3 (examples), 6 (README); §2.8 (no likelihood methods) → nothing to build; §3.1–3.3 → Task 5 (`transform`), Task 6 (`wdm.yml`); §4 → Task 5; §5 → Tasks 4–6 (`NotImplementedError` gone, coverage exclusion dropped); §6 → Tasks 1, 2, 4, 5; §7 → Tasks 2, 3, 5, 6; §8 non-goals untouched.
- Ordering: every task leaves the suite green; the three local strict xfails survive Tasks 1–4 and clear in Task 5.
- Known refinement over the spec's wording: §2.2 item 5's "every diagonal entry has zero imaginary part" is implied by exact Hermitian symmetry (item 4), so the constructor has no separate check for it and the test expects the Hermitian message.
