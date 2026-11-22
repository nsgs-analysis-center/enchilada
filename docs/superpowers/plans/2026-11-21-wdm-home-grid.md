# WDM Home Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `Wheel` stores a campaign on one WDM grid chosen at construction (the *home grid*), serves every block on the grid it asks for by exact data transforms, and delivers the noise sum to any grid through a covariance translation rule (`transform(covariance, target_domain, ...)`) that reads every covariance as a diagonal spectral model; each block declares its own `role` attribute instead of the `role=` keyword, and the one-covariance-grid rule from PR #12 is gone, together with its two refusals.

**Architecture:** A new private module `src/enchilada/_spectral.py` holds the data-independent weights of a WDM grid (domania's Meyer window sampled in frequency and in time, the folded cell layout) and the six O(N) kernels that move a diagonal covariance between time, frequency and WDM grids; `transform` gains a `Covariance` overload that resolves the target grid like data does and builds the result through the validating constructor. `Wheel` keeps the arrival-grid observations for its public views and a home-grid copy for everything else: the residual and both sums are formed on the home grid, a block's inputs are transformed (data) or translated (covariance) to its grid, a returned template is transformed onto the home grid at adoption and a returned covariance is validated, kept native in the ledger and translated into the noise sum. Roles come from `block.role`, read once at `add`. The rule and its overload land first with the suite green (Task 1); the core lands second with its own tests and the inverted core test files (Task 2; the rest of the suite is red until the migrations finish); five migrations with disjoint files run in parallel (Tasks 3–7); the documentation task closes with full gates (Task 8).

**Tech Stack:** Python 3.12+, NumPy, domania 3.0 (`domania.wilson.windows.sample_window`, `MEYER`, `domania.wilson.grid.cells`, `endpoint_weights`; `domania.testing`/`domania.wilson.matrix` as test oracles), the existing `Wheel`/`BlockState`/`Covariance`/`transform` machinery; pytest, ruff, mypy; notebooks edited as JSON by cell id (`demo.ipynb` by text substitution).

**User decisions (already made, 2026-11-21, see the spec):**
- The covariance is assumed diagonal in the grid index on every grid; the user assesses that approximation and Wheel does not restrict grid choices on its account ("Your worry is noted and I am dismissing it").
- The campaign stores everything on one grid in the WDM domain ("I want to store everything on a grid in the WDM domain").
- The home grid is set by division counts at construction; WDM observations use their own grid; time or frequency observations without divisions are refused.
- Translation reads every covariance as a piecewise-constant spectral model S(f, t) ("spectral model as currency"), band-averaged into WDM with domania's window.
- A block declares its role as an attribute ("Can we make the roles a part of the block definition?"); `add` and `check_block` lose the `role` keyword.

**Refinements of the spec recorded here (not open questions; reviewers compare against these):**
- **The kernels are enchilada's, domania supplies the window.** `domania.wdm_covariance` evaluates a time-varying model on a (bands × columns × window bins) grid, O(N · Nt): 285 s measured for 3 channels at 2^20 samples. The φ̂²-weighted band mean it computes is reproduced in `_spectral.py` in O(N) (0.08 s at that size) and the tests pin the two equal to 2e-16 on a stationary PSD, which is what the spec's "equals `domania.wdm_covariance`" and the forge's `analytic_covariance` need. `wdm_covariance` is the test oracle, not the production path.
- **Time weights are the window's squared time profile, in both time directions.** The spec names "the window's time weights normalised over columns" for WDM → time; the same weights give the column value of a time covariance going into WDM (the window-weighted mean of σ²(t_j) around each column centre), which is what makes "time-varying white noise into WDM" exact: it reproduces the exact cell variances `diag(W D Wᵀ)` to 9e-16, where a hard column mean is off by 5 %. The time profile is the inverse FFT of the sampled frequency window placed on the record's FFT grid; it equals the DC atom's envelope.
- **Reflection at DC and Nyquist, one way only.** Reading bins into bands (`bands_from_bins`) folds the window's negative-frequency samples back onto positive bins (a one-sided PSD mirrored), which is what `domania.wdm_covariance` does; spreading bands onto bins (`bins_from_bands`) and the band shares of the time mean do not reflect, because the DC and Nyquist atoms put their power on the positive bins once. Both directions are verified against dense oracles (2e-15); a reviewer must not "fix" the asymmetry.
- **DC and Nyquist bands have cells at every other column.** The padded band × column model repeats the previous column into the missing one; the folded layout reads them back (`domania.wilson.grid.cells`).
- **WDM → another WDM grid:** target bands take the window band mean of the source model (a band-to-band matrix cached per grid pair, `functools.lru_cache`); target columns read the source column containing their centre. The spec's "sampled at the target's cells by `wdm_covariance`" is the same band mean without the O(N · Nt) evaluation. Cost at 3 channels, 256 → 512 bands at 2^20 samples: 3.4 s the first time (builds the matrix), 0.8 s after.
- **Measured costs** (3 channels × 2^20 samples, this machine): each translation direction 0.01–0.15 s; translation plus the validating construction of the result 0.20–0.25 s; `Covariance.copy` alone 0.12 s. The README states these, not the spec's 0.12 s guess.
- **The `Block` protocol member is `role: str`**, declared like `name` (a read-only property under `TYPE_CHECKING`, a plain annotation otherwise). mypy infers the one-line `role = "noise"` the spec prescribes as `str`, which a `Literal` member would reject (verified: a `Literal` member fails the spec's own form and a dataclass field; the `str` member accepts a class attribute, a `Final`, a dataclass field and an explicit subclass, and still rejects a class with no `role`). `Role` and `ROLES` stay in `wheel.py`, unexported; `add` validates the value at runtime and narrows it to `Role`.
- **`add` reads the role after the protocol checks** (name, duplicate name, `sample`, `initial_sample`), before `_block_inputs`: `TypeError` "block 'x' declares no role; a Block needs name, role ('signal' or 'noise'), initial_sample(...) and sample(...)." for a missing or `None` attribute, `ValueError` "block 'x' declares role 'nosie'; a role is 'signal' or 'noise'" otherwise. `check_block` refuses a `noise_blocks` entry whose `role` is not `"noise"` with `ValueError` naming it, before building the Wheel.
- **A returned template is validated on the block's grid, then transformed.** `_adopt` runs `check_on_grid` and the finiteness check against the block's grid with the `"<block>.<method> template"` context (the messages blocks on the arrival grid get today), then `transform_channels` onto the home grid; `_canonical_state` is gone. The residual for a block is formed on the home grid and transformed once to the block's grid.
- **`transform(covariance, ...)` ignores `reference_data`** the way the `L1Data` path does; the catch-all message becomes "transform expects L1Data, BlockState or Covariance".
- **`Covariance.__add__` keeps refusing two grids** ("a covariance does not add across grids" in `covariance.py` and `tests/test_covariance.py` stay); only the Wheel's two refusals go.

**Spec:** `docs/superpowers/specs/2026-11-21-wdm-home-grid-design.md` (committed on this branch as 5ef8d85). **Branch:** `wdm-home`, cut from `main` at c669da8 (PR #12). Work in the main checkout `~/Documents/lisa/enchilada` (the editable install imports `src/enchilada` from here). Commit trailer, every commit:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P
```

**Tooling rules (binding for every task):**
- Run checks as `.venv/bin/python -m pytest -q -p no:cacheprovider ...`, `.venv/bin/python -m ruff check src tests examples benchmarks forge`, `.venv/bin/python -m ruff format --check src tests examples benchmarks forge` (formatting itself: `.venv/bin/python -m ruff format <the .py files you changed>` with explicit paths; **never a bare `ruff format`, never a notebook, never the README**), `.venv/bin/python -m mypy`. Never `uv run`, `uvx`, `uv sync`, `uv lock`, prek. No dependency changes anywhere in this plan, so `uv.lock` is untouched.
- Full suite: `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider --cov=enchilada` (today 687 passed, 5 skipped, coverage gate 95 %).
- Temporary files under `$TMPDIR`. Stage named paths only (never `git add -A`, `.` or `-u`). Before every git operation run `git branch --show-current` and stop if it does not print `wdm-home`: other Claude sessions share this checkout (a worktree `.claude/worktrees/l2data` belongs to another session; leave it alone).
- Notebooks: edit `create_block.ipynb`, `domain_translation.ipynb`, `gb_block_eryn.ipynb`, `forge/block.ipynb` and `forge/wheel.ipynb` by cell id with a JSON script (`json.load`, set `cell["source"]` to a list of lines each ending in `\n` except the last, `json.dump(nb, f, indent=1, ensure_ascii=False)` plus a trailing newline; verify a no-op dump round-trips byte for byte before editing). `demo.ipynb` does **not** round-trip: edit it by exact text substitution on the raw file. Every notebook code line must stay ≤ 88 columns and formatter-shaped (`ruff format --check examples forge` checks notebooks).
- Subagent reports carry no user authority; a decision the header does not record is not yours to make, flag it.

**Migration rules (Tasks 2–7 all meet these; state them once):**
- **Role keyword → attribute.** Delete every `role="signal"` / `role="noise"` keyword at `wheel.add(...)` and `check_block(...)`; give every block class one line `role = "signal"` or `role = "noise"` right after the class docstring (or as the first class statement when there is none). A block registered under both roles in different tests is split into two classes or gets its role set per instance (`block.role = ...` before `add` is fine: the attribute is read at `add`).
- **Divisions at construction.** Every `Wheel(data, ...)` on time or frequency data gains `num_frequency_divisions=8`, and so does every `check_block(...)` on such data, unless the test needs another grid. All test records are 64 samples or a larger power of two, so 8 rows is always valid (`N % 16 == 0`); WDM data (`transform(data, "wdm", ...)`) needs no divisions. Example sizes: README quickstart and `demo` 1024 → 32; `create_block.ipynb` 256 → 16; `domain_translation` 128 → 8; `toy_fit.py` 4096 → 64; `noise_floor.py` 2048 → 32; `gb_block_eryn.ipynb` 2^19 → 512; the benchmark takes a `--rows` option (default 128 for its default 32768 samples); `mojito_wdm_wheel.py` arrives on WDM. The forge reads `FORGE_ROWS` (Task 7).
- **Views pass through a transform.** `wheel.residual()`, `wheel.working_residual`, `wheel.contribution(...)` and anything derived from the home-grid residual come back through one exact inverse WDM transform: `np.testing.assert_array_equal` on them becomes `np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * scale)` (use `atol` alone for a zero expectation; `scale` is the size of the numbers involved, e.g. `atol=1e-6` for the ±1e8 cancellation test, whose float64 rounding is 1e-8). `wheel.observed_data` is the untouched arrival copy and stays exact.
- **`wheel.covariance` is on the home grid.** White noise of variance σ² per time sample is `dt * σ²` per WDM cell (`conftest.white_covariance` already knows the three conventions); compare `wheel.covariance.A` with `white_covariance(transform(data, "wdm", num_frequency_divisions=8), level).A` (allclose), or compare `transform(wheel.covariance, data.data_domain).A` with the time value. A covariance *received by a block* on its own grid, and a block's own native publication in `wheel.ledger[name].covariance`, keep today's values and grids (received sums off the home grid are translated, so compare them with allclose; a lone white floor translates exactly to rounding).
- **Ledger templates are on the home grid.** `wheel.ledger[name].template["A"]` is an `(8, 8)` array; a test that read template values on the arrival grid reads `wheel.contribution(name)["A"]` instead (allclose), or compares the ledger template with `transform(state, "wdm", num_frequency_divisions=8, reference_data=data).template`. `template is None` checks are unchanged.
- **The grid rule is gone.** Delete tests that pin "does not translate between grids", "does not add across grids" (the Wheel ones; `Covariance.__add__`'s stay), "the current covariance is on grid", "campaign covariance is on grid" and `_covariance_grid`; where such a test also pinned rollback or the hand-over, Task 2's `tests/test_wheel_home_grid.py` carries the replacement. A noise block publishing on another grid than its own registration grid is still refused ("<block>.<method> covariance is on grid ..."), so those tests stay.
- **Message rewording.** "is registered as a signal block" → "declares role 'signal'", "is registered as a noise block" → "declares role 'noise'".
- **The migration script** below does the mechanical part of a file; run it, then fix what it reports by hand (multi-line calls, other data names), then run the file's tests:

```python
# $TMPDIR/migrate.py — mechanical part of the role/home-grid migration. Usage:
#   .venv/bin/python $TMPDIR/migrate.py tests/test_wheel.py [more files]
import re
import sys

ROLE_KW = re.compile(r"""(,\s*)?\brole=(["'])(signal|noise)\2""")
ROLE_LINE = re.compile(r"""^[ \t]*role=(["'])(signal|noise)\1,?[ \t]*\n""", re.M)
# Wheel(<name>) / Wheel(<name>, ... / Wheel(observed_data=<name> ... for the
# 64-sample fixtures and locals: observed, obs, data, observations()
WHEEL = re.compile(
    r"""Wheel\((observed_data=)?(observed|obs|data|observations\(\))(\)|,\s)"""
)
CHECK = re.compile(r"""check_block\(([^()\n]*)\)""")

for path in sys.argv[1:]:
    text = open(path).read()
    text = ROLE_LINE.sub("", text)
    text = ROLE_KW.sub(lambda m: "" if m.group(1) else "", text)
    text = re.sub(r"\(\s*,\s*", "(", text)  # "(, x" left by a leading role=
    text = WHEEL.sub(
        lambda m: f"Wheel({m.group(1) or ''}{m.group(2)}, num_frequency_divisions=8"
        + (")" if m.group(3) == ")" else ", "),
        text,
    )
    text = CHECK.sub(
        lambda m: m.group(0)
        if "num_frequency_divisions" in m.group(1)
        else f"check_block({m.group(1)}, num_frequency_divisions=8)",
        text,
    )
    open(path, "w").write(text)
    for number, line in enumerate(text.splitlines(), 1):
        if "role=" in line or (
            ("Wheel(" in line or "check_block(" in line)
            and "num_frequency_divisions" not in line
            and "class " not in line
        ):
            print(f"{path}:{number}: check by hand: {line.strip()}")
```

After the script, `.venv/bin/python -m ruff format <file>` and `ruff check <file>` (unused imports such as `re`, `Role`, `_covariance_grid` references, line lengths), then pytest on the file.

**Task map and parallelism.** Task 1 first (suite stays green). Task 2 depends on Task 1 and is the only task that breaks the suite; Tasks 3, 4, 5, 6 and 7 depend only on Task 2 and touch disjoint files, so the controller may dispatch them in parallel (two or three at a time is comfortable). Task 8 depends on all of them. Each task's **Verify** names only files that pass at its commit; Task 8 restores the whole suite. Do not push before Task 8.

| Task | Files | Blocked by |
|---|---|---|
| 1 Translation kernels and the `transform` overload | `src/enchilada/_spectral.py` (new), `src/enchilada/translation.py`, `tests/test_covariance_translation.py` (new), `tests/test_translation.py` (one test) | — |
| 2 Home grid, hand-over by translation, declared roles | `src/enchilada/{wheel,block,fixed_noise,testing}.py`, `tests/test_wheel_home_grid.py` (new), `tests/{test_noise_sum,test_initial_sample,test_testing,test_typing,test_fixed_noise}.py` | 1 |
| 3 Wheel test migration | `tests/test_wheel.py`, `tests/test_wheel_translation.py` | 2 |
| 4 Remaining test migration | `tests/{test_state,test_block_state,test_block_state_contract,test_ledger,test_translation_review,test_data}.py` | 2 |
| 5 Examples, benchmark and their tests | `examples/{demo,toy_fit,noise_floor,domain_translation,gb_model,mojito_wdm_wheel}.py`, `benchmarks/benchmark_wheel.py`, `tests/{test_examples,test_toy_translation,test_gb_example_contract,test_noise_floor_example}.py` | 2 |
| 6 Notebooks | `examples/{create_block,demo,domain_translation,gb_block_eryn}.ipynb`, `tests/test_block_tutorial.py` | 2 |
| 7 Forge | `forge/{block.py,README.md,block.ipynb,wheel.ipynb}`, `tests/test_forge.py` | 2 |
| 8 README, CHANGELOG, `docs/wheel.svg`, wording sweep, full gates | `README.md`, `CHANGELOG.md`, `docs/wheel.svg`, anything the sweep finds | 3, 4, 5, 6, 7 |

---

### Task 1: Translation kernels and the `transform` overload

**Goal:** `src/enchilada/_spectral.py` holds the data-independent weights of a WDM grid and the six kernels that carry a diagonal covariance between time, frequency and WDM grids; `transform(covariance, target_domain, *, num_frequency_divisions=None, num_time_divisions=None)` returns a validated `Covariance` on the target grid (a copy on the same grid); both are pinned against domania's dense oracles; the existing suite stays green because the only public change is that `transform` now accepts a `Covariance` instead of refusing it.

**Files:**
- Create: `src/enchilada/_spectral.py`
- Modify: `src/enchilada/translation.py` (imports; a third `@overload`; the implementation's `isinstance(value, Covariance)` branch; the docstring; the catch-all message)
- Create: `tests/test_covariance_translation.py`
- Modify: `tests/test_translation.py:127-137` (`test_transform_rejects_a_covariance` becomes `test_transform_translates_a_covariance_and_refuses_other_types`)

**Acceptance Criteria:**
- [ ] `translate_blocks(blocks, source_grid, target_grid)` exists for every ordered pair of distinct domains (time, frequency, WDM) and for two different WDM grids; the spectral-model reading of each grid is the one in the spec's table.
- [ ] White noise round-trips exactly (to 1e-14 relative) through time → WDM → time, frequency → WDM → frequency, time ↔ frequency, WDM → time → WDM and WDM → frequency → WDM.
- [ ] A stationary PSD into WDM equals `domania.wdm_covariance` on every cell (DC and Nyquist cells included) to 1e-12 and equals `diag(W C_time Wᵀ)` from `domania.wilson.matrix`; frequency → time equals `diag(C_time)`; WDM → frequency equals the exact diagonal `|F W⁻¹|² C`; time-varying white noise into WDM equals `diag(W D Wᵀ)`; time → frequency is `dt² Σ σ²`; WDM → time preserves the trace and is constant for a stationary input.
- [ ] `transform(covariance, ...)` returns a `Covariance` on the target grid with the same `channel_names`, Hermitian and positive definite for a 3-channel input with a complex cross term, `complex128` on frequency and `float64` elsewhere; the same grid returns an equal, independent copy; division keywords follow the data rules.
- [ ] `tests/test_translation.py` no longer expects the `TypeError`; the full suite is green.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider --cov=enchilada` → all passed, 0 failed (5 skips acceptable), coverage ≥ 95 %; `.venv/bin/python -m mypy` → `Success: no issues found`; `.venv/bin/python -m ruff check src tests` and `ruff format --check src tests` clean.

**Steps:**

- [ ] **Step 1: Write the failing kernel tests**

Create `tests/test_covariance_translation.py`:

```python
"""Covariance translation between grids: the diagonal spectral-model rule.

Every covariance is read as a piecewise-constant one-sided spectral model
S(f, t) and the target grid takes weighted means of it with the Meyer
window's band weights and time weights. The kernels are pinned against
domania's dense oracles on a small grid (``build_W``, ``build_C_time``), so
the exact cases are exact and the smoothed cases preserve the trace.
"""

import domania
import numpy as np
import pytest
from domania.wilson import matrix

from enchilada import Covariance, Grid, WDMGrid, transform
from enchilada._spectral import _layout, translate_blocks

NF, NT, FS = 8, 32, 2.0
N = NF * NT
DT = 1.0 / FS
NUM_BINS = N // 2 + 1


def grid(domain, num_frequency_divisions=NF):
    wdm = (
        WDMGrid(
            num_frequency_divisions=num_frequency_divisions,
            num_time_divisions=N // num_frequency_divisions,
        )
        if domain == "wdm"
        else None
    )
    return Grid(domain, N, FS, wdm)


TIME, FREQUENCY, WDM = grid("time"), grid("frequency"), grid("wdm")


def one(values):
    """A single channel pair's blocks: shape (1, 1, *channel_shape)."""
    return np.asarray(values)[None, None]


def psd(frequency_hz):
    """A smooth coloured one-sided PSD, finite at f = 0."""
    f = np.asarray(frequency_hz, dtype=float)
    return 1e-2 * (1 + 1 / (1 + (f / (0.1 / DT)) ** 4)) * (1 + 0.5 * np.cos(40 * f))


@pytest.fixture(scope="module")
def dense():
    """The dense WDM matrix W (wdm = W x), its inverse and the dt * rfft matrix."""
    W = matrix.build_W(NF, NT, DT)
    k = np.arange(NUM_BINS)[:, None]
    j = np.arange(N)[None, :]
    F = DT * np.exp(-2j * np.pi * k * j / N)
    return W, W.T / DT, F


def relative(actual, expected):
    return np.abs(actual / expected - 1).max()


class TestWhiteRoundTrips:
    @pytest.mark.parametrize(
        "source, via, level",
        [
            (TIME, WDM, 2.0),
            (TIME, FREQUENCY, 2.0),
            (FREQUENCY, WDM, 3.0),
            (FREQUENCY, TIME, 3.0),
            (WDM, TIME, 0.7),
            (WDM, FREQUENCY, 0.7),
        ],
        ids=["t-w-t", "t-f-t", "f-w-f", "f-t-f", "w-t-w", "w-f-w"],
    )
    def test_white_noise_comes_back_exactly(self, source, via, level):
        values = np.full(source.channel_shape, level)
        if source.data_domain == "frequency":
            values = values + 0j
        there = translate_blocks(one(values), source, via)
        back = translate_blocks(there, via, source)
        assert back.shape == one(values).shape
        assert relative(back.real, level) < 1e-14


class TestExactCases:
    def test_stationary_psd_into_wdm_is_domanias_band_mean(self):
        f = np.fft.rfftfreq(N, DT)
        bins = one((N * DT / 2) * psd(f) + 0j)
        mine = translate_blocks(bins, FREQUENCY, WDM)[0, 0]
        theirs = domania.wdm_covariance(
            lambda ff, t: psd(ff)[None, None], Nf=NF, Nt=NT, dt=DT
        )[0, 0]
        assert relative(mine, theirs) < 1e-12  # every cell, DC and Nyquist too

    def test_stationary_psd_into_wdm_is_the_exact_cell_variance(self, dense):
        W, _, _ = dense
        f = np.fft.rfftfreq(N, DT)
        bins = one((N * DT / 2) * psd(f) + 0j)
        time_covariance = matrix.build_C_time(N, DT, psd)
        exact = np.diag(W @ time_covariance @ W.T).reshape(NF, NT)
        assert relative(translate_blocks(bins, FREQUENCY, WDM)[0, 0], exact) < 1e-12

    def test_frequency_to_time_is_the_diagonal_of_the_time_covariance(self):
        f = np.fft.rfftfreq(N, DT)
        bins = one((N * DT / 2) * psd(f) + 0j)
        exact = np.diag(matrix.build_C_time(N, DT, psd))
        assert relative(translate_blocks(bins, FREQUENCY, TIME)[0, 0], exact) < 1e-12

    def test_wdm_to_frequency_is_the_exact_diagonal(self, dense):
        _, W_inverse, F = dense
        rng = np.random.default_rng(3)
        cells = np.exp(0.3 * rng.normal(size=(NF, NT)))
        exact = (np.abs(F @ W_inverse) ** 2) @ cells.ravel()
        mine = translate_blocks(one(cells), WDM, FREQUENCY)[0, 0]
        assert mine.dtype == np.complex128
        assert relative(mine.real, exact) < 1e-12

    def test_time_varying_white_noise_into_wdm_is_the_exact_cell_variance(self, dense):
        W, _, _ = dense
        variance = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(N) / N)
        exact = ((W**2) @ variance).reshape(NF, NT)
        mine = translate_blocks(one(variance), TIME, WDM)[0, 0]
        assert relative(mine, exact) < 1e-12

    def test_time_to_frequency_is_the_mean_power(self):
        variance = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(N) / N)
        mine = translate_blocks(one(variance), TIME, FREQUENCY)[0, 0]
        assert mine.dtype == np.complex128
        assert relative(mine.real, DT**2 * variance.sum()) < 1e-14

    def test_wdm_to_time_preserves_the_trace_and_flattens_a_stationary_input(
        self, dense
    ):
        _, W_inverse, _ = dense
        rng = np.random.default_rng(4)
        cells = np.exp(0.3 * rng.normal(size=(NF, NT)))
        exact = (W_inverse**2) @ cells.ravel()
        mine = translate_blocks(one(cells), WDM, TIME)[0, 0]
        assert relative(mine.mean(), exact.mean()) < 1e-12  # trace preserved
        assert np.median(np.abs(mine / exact - 1)) < 0.05  # column-smoothed
        stationary = np.repeat(np.exp(0.5 * rng.normal(size=(NF, 1))), NT, axis=1)
        flat = translate_blocks(one(stationary), WDM, TIME)[0, 0]
        assert relative(flat, flat.mean()) < 1e-12
        assert (
            relative(flat.mean(), ((W_inverse**2) @ stationary.ravel()).mean()) < 1e-12
        )

    def test_wdm_to_wdm_is_the_band_mean_at_the_target_columns(self):
        rng = np.random.default_rng(5)
        cells = np.exp(0.3 * rng.normal(size=(NF, NT)))
        target = grid("wdm", 16)
        mine = translate_blocks(one(cells), WDM, target)[0, 0]
        source_layout, target_layout = _layout(NF, NT, DT), _layout(16, 16, DT)
        model = 2.0 * source_layout.padded(cells)
        by_hand = np.zeros((17, 16))
        for column in range(16):
            source_column = source_layout.column_of_sample[column * 16]
            by_hand[:, column] = target_layout.bands_from_bins(
                source_layout.bins_from_bands(model[:, source_column])
            )
        assert relative(mine, target_layout.folded(by_hand) / 2.0) < 1e-12
        white = translate_blocks(one(np.full((NF, NT), 0.7)), WDM, target)[0, 0]
        assert relative(white, 0.7) < 1e-14

    def test_time_varying_white_noise_cell_variances_match_realisations(self):
        rng = np.random.default_rng(6)
        variance = 1.0 + 0.8 * np.sin(2 * np.pi * np.arange(N) / N)
        draws = rng.standard_normal((4000, N)) * np.sqrt(variance)
        empirical = domania.transform(
            draws, "time", "wdm", N=N, dt=DT, target_wdm=(NF, NT)
        ).var(axis=0)
        predicted = translate_blocks(one(variance), TIME, WDM)[0, 0]
        rms = np.sqrt(np.mean((empirical / predicted - 1) ** 2))
        assert rms < 0.05  # per-cell Monte-Carlo noise is sqrt(2 / 4000) = 0.022


def three_channel_spectrum():
    """A 3-channel frequency covariance with a complex A-E cross term."""
    f = np.fft.rfftfreq(N, DT)
    blocks = np.zeros((3, 3, NUM_BINS), dtype=np.complex128)
    for index in range(3):
        blocks[index, index] = (index + 1) * (N * DT / 2) * psd(f)
    cross = 0.3 * np.sqrt(blocks[0, 0].real * blocks[1, 1].real)
    cross = cross * np.exp(1j * 10 * f)
    cross[0], cross[-1] = cross[0].real, cross[-1].real  # real DC and Nyquist
    blocks[0, 1], blocks[1, 0] = cross, cross.conj()
    return Covariance(blocks, channel_names=("A", "E", "T"), grid=FREQUENCY)


class TestTransformOverload:
    @pytest.mark.parametrize("target_domain", ["time", "wdm"])
    def test_a_correlated_covariance_stays_hermitian_and_positive_definite(
        self, target_domain
    ):
        source = three_channel_spectrum()
        translated = transform(
            source,
            target_domain,
            num_frequency_divisions=NF if target_domain == "wdm" else None,
        )
        assert type(translated) is Covariance
        assert translated.channel_names == ("A", "E", "T")
        assert translated.grid == grid(target_domain)
        assert translated.blocks.dtype == np.float64  # validated by the constructor
        back = transform(translated, "frequency")
        assert back.grid == FREQUENCY
        assert back.blocks.dtype == np.complex128
        assert np.all(back.blocks[0, 1].imag == 0)  # the diagonal model keeps Re
        assert np.all(back.A > 0)

    def test_pairs_are_carried_elementwise(self):
        data_grid = TIME
        blocks = np.zeros((2, 2, N))
        blocks[0, 0], blocks[1, 1], blocks[0, 1], blocks[1, 0] = 1.0, 4.0, 0.5, 0.5
        source = Covariance(blocks, channel_names=("A", "E"), grid=data_grid)
        cells = transform(source, "wdm", num_frequency_divisions=NF)
        np.testing.assert_allclose(cells.A, DT * 1.0, rtol=1e-14)
        np.testing.assert_allclose(cells.E, DT * 4.0, rtol=1e-14)
        np.testing.assert_allclose(cells.AE, DT * 0.5, rtol=1e-14)
        spectrum = transform(source, "frequency")
        np.testing.assert_allclose(spectrum.AE.real, N * DT**2 * 0.5, rtol=1e-14)

    def test_the_same_grid_returns_an_independent_equal_copy(self):
        source = three_channel_spectrum()
        copy = transform(source, "frequency")
        assert copy is not source
        assert copy.grid == source.grid
        np.testing.assert_array_equal(copy.blocks, source.blocks)
        assert not np.shares_memory(copy.blocks, source.blocks)
        cells = transform(source, "wdm", num_frequency_divisions=NF)
        again = transform(cells, "wdm")  # WDM to WDM without counts: its own grid
        assert again.grid == cells.grid
        np.testing.assert_array_equal(again.blocks, cells.blocks)
        same_counts = transform(cells, "wdm", num_frequency_divisions=NF)
        np.testing.assert_array_equal(same_counts.blocks, cells.blocks)

    def test_division_keywords_follow_the_data_rules(self):
        source = three_channel_spectrum()
        with pytest.raises(ValueError, match="division counts apply only"):
            transform(source, "time", num_frequency_divisions=NF)
        with pytest.raises(ValueError, match="num_frequency_divisions"):
            transform(source, "wdm")  # a WDM target from frequency needs a count
        with pytest.raises(ValueError, match="target_domain must be one of"):
            transform(source, "wavelet")
        by_columns = transform(source, "wdm", num_time_divisions=NT)
        assert by_columns.grid == WDM
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_covariance_translation.py`
Expected: collection error, `ModuleNotFoundError: No module named 'enchilada._spectral'`.

- [ ] **Step 3: Write `src/enchilada/_spectral.py`**

```python
"""Translate a diagonal covariance between grids through a spectral model.

Every covariance is read as a piecewise-constant one-sided spectral model
S(f, t) in the units `Covariance` uses on its grid: per-sample variances
sigma^2(t_j) on time (S = 2 dt sigma^2, flat in f), per-bin variances C_k on
frequency (S = 2 C_k / (N dt), flat in t) and per-cell variances C_mn on WDM
(S = 2 C_mn in band m at column n). The target grid's values are weighted
means of that model with non-negative weights: the Meyer window's squared
frequency samples across bands and its squared time profile across columns,
the same window domania's data transforms use, so Hermitian symmetry and
positive definiteness survive. White noise round-trips exactly through every
pair of grids; everything else is smoothed by a band mean (into and out of
WDM across frequency), a column mean (across time) or the flattening of the
time <-> frequency step. The user owns that approximation.

Reading bins into bands reflects the window's negative-frequency samples back
onto the positive bins (a one-sided PSD seen through a two-sided window, what
``domania.wdm_covariance`` computes); spreading bands onto bins, and the band
shares of a time mean, do not reflect, because the DC and Nyquist atoms put
their power on the positive bins once. Both are exact in their roles against
the dense transforms (see ``tests/test_covariance_translation.py``).
"""

from functools import lru_cache

import numpy as np
from domania.wilson.grid import cells, endpoint_weights
from domania.wilson.windows import MEYER, sample_window

from enchilada.domains import Grid


class _Layout:
    """Data-independent weights of one WDM grid.

    ``band`` and ``column`` map folded slots to bands 0..Nf and columns
    0..Nt-1 (``domania.wilson.grid.cells``). ``bins[m, i]`` is the signed rfft
    bin of window sample ``i`` of band ``m`` and ``inside`` marks the samples
    that fall on an rfft bin; ``reflected`` folds the others back.
    ``band_share[m]`` is band m's share of the two-sided frequency mean (half
    a share for the DC and Nyquist bands). The time profile ``phi2_t`` is the
    window's squared envelope centred on sample 0, kept as its rfft for the
    circular convolutions of the time direction.
    """

    def __init__(
        self,
        num_frequency_divisions: int,
        num_time_divisions: int,
        sample_interval_s: float,
    ) -> None:
        nf, nt, dt = num_frequency_divisions, num_time_divisions, sample_interval_s
        n = nf * nt
        self.num_frequency_divisions = nf
        self.num_time_divisions = nt
        self.num_time_samples = n
        self.num_bins = n // 2 + 1
        self.band, self.column = cells(nf, nt)
        phi2 = sample_window(MEYER, nf, nt, dt) ** 2  # (nt,), centred at nt // 2
        half = nt // 2
        offsets = np.arange(nt) - half
        bins = np.arange(nf + 1)[:, None] * half + offsets[None, :]
        self.phi2 = phi2
        self.bins = bins
        self.inside = (bins >= 0) & (bins <= n // 2)
        reflected = np.abs(bins)
        self.reflected = np.where(reflected > n // 2, n - reflected, reflected)
        weights = np.broadcast_to(phi2, bins.shape)
        bin_norm = np.zeros(self.num_bins)
        np.add.at(bin_norm, bins[self.inside], weights[self.inside])
        self.bin_norm = bin_norm
        eta = endpoint_weights(self.num_bins)
        safe = np.where(self.inside, bins, 0)
        share = (eta[safe] / bin_norm[safe]) * weights * self.inside
        self.band_share = share.sum(axis=1) / eta.sum()
        spectrum = np.zeros(n)
        spectrum[offsets % n] = np.sqrt(phi2)
        phi2_t = np.fft.ifft(spectrum).real ** 2
        self.phi2_t_hat = np.fft.rfft(phi2_t)
        self.phi2_t_sum = phi2_t.sum()
        self.column_centres = np.arange(nt) * nf
        self.column_of_sample = ((np.arange(n) + nf // 2) // nf) % nt
        impulses = np.zeros(n)
        impulses[self.column_centres] = 1.0
        self.column_weight_sum = self._convolve(impulses)

    # ---- frequency direction ------------------------------------------------
    def bands_from_bins(self, values: np.ndarray) -> np.ndarray:
        """Window band means of per-bin values, (..., num_bins) -> (..., Nf + 1)."""
        weighted = values[..., self.reflected] * self.phi2
        return weighted.sum(axis=-1) / self.phi2.sum()

    def bins_from_bands(self, values: np.ndarray) -> np.ndarray:
        """Per-bin values from per-band values, (..., Nf + 1) -> (..., num_bins)."""
        out = np.zeros(values.shape[:-1] + (self.num_bins,))
        for band in range(self.num_frequency_divisions + 1):
            selected = self.inside[band]
            low, high = self.bins[band, selected][0], self.bins[band, selected][-1] + 1
            out[..., low:high] += values[..., band, None] * self.phi2[selected]
        return out / self.bin_norm

    # ---- time direction ---------------------------------------------------------
    def _convolve(self, values: np.ndarray) -> np.ndarray:
        n = self.num_time_samples
        spectrum = np.fft.rfft(values, axis=-1) * self.phi2_t_hat
        return np.fft.irfft(spectrum, n=n, axis=-1)

    def columns_from_samples(self, values: np.ndarray) -> np.ndarray:
        """Window time means of per-sample values, (..., N) -> (..., Nt)."""
        return self._convolve(values)[..., self.column_centres] / self.phi2_t_sum

    def samples_from_columns(self, values: np.ndarray) -> np.ndarray:
        """Per-sample values from per-column values, (..., Nt) -> (..., N)."""
        impulses = np.zeros(values.shape[:-1] + (self.num_time_samples,))
        impulses[..., self.column_centres] = values
        return self._convolve(impulses) / self.column_weight_sum

    # ---- cell layout ----------------------------------------------------------------
    def padded(self, folded: np.ndarray) -> np.ndarray:
        """Folded cells (..., Nf, Nt) -> bands x columns (..., Nf + 1, Nt).

        The DC and Nyquist bands have cells at every other column; the missing
        columns repeat the previous one.
        """
        nf = self.num_frequency_divisions
        out = np.zeros(folded.shape[:-2] + (nf + 1, self.num_time_divisions))
        out[..., self.band, self.column] = folded
        out[..., 0, 1::2] = out[..., 0, 0::2]
        out[..., nf, 1::2] = out[..., nf, 0::2]
        return out

    def folded(self, padded: np.ndarray) -> np.ndarray:
        """Bands x columns (..., Nf + 1, Nt) -> folded cells (..., Nf, Nt)."""
        return padded[..., self.band, self.column]


@lru_cache(maxsize=8)
def _layout(
    num_frequency_divisions: int, num_time_divisions: int, sample_interval_s: float
) -> _Layout:
    return _Layout(num_frequency_divisions, num_time_divisions, sample_interval_s)


@lru_cache(maxsize=8)
def _band_matrix(source: _Layout, target: _Layout) -> np.ndarray:
    """(Nf_target + 1, Nf_source + 1): the target band means of each source band."""
    unit = np.eye(source.num_frequency_divisions + 1)
    columns = [target.bands_from_bins(source.bins_from_bands(e)) for e in unit]
    return np.stack(columns, axis=1)


def _layout_of(grid: Grid) -> _Layout:
    if grid.wdm_grid is None:
        raise ValueError(f"{grid} is not a WDM grid")
    return _layout(
        grid.wdm_grid.num_frequency_divisions,
        grid.wdm_grid.num_time_divisions,
        1.0 / grid.sample_rate_hz,
    )


def translate_blocks(blocks: np.ndarray, source: Grid, target: Grid) -> np.ndarray:
    """The covariance blocks ``(K, K, *source.channel_shape)`` read on ``target``.

    ``source`` and ``target`` share the record (sample count and rate) and
    differ in domain or WDM grid. Frequency targets come back ``complex128``,
    the others ``float64``; a complex cross term entering time or WDM keeps
    its real part (the mean power of the pair).
    """
    n = source.num_time_samples
    dt = 1.0 / source.sample_rate_hz
    pair_shape = blocks.shape[:2]
    if source.data_domain == "time":
        variance = blocks.real
        if target.data_domain == "frequency":
            level = n * dt**2 * variance.mean(axis=-1)
            shape = pair_shape + (n // 2 + 1,)
            return np.broadcast_to(level[..., None], shape).astype(np.complex128)
        layout = _layout_of(target)
        column_model = 2.0 * dt * layout.columns_from_samples(variance)
        shape = pair_shape + (
            layout.num_frequency_divisions + 1,
            layout.num_time_divisions,
        )
        return layout.folded(np.broadcast_to(column_model[..., None, :], shape)) / 2.0
    if source.data_domain == "frequency":
        eta = endpoint_weights(n // 2 + 1)
        model = 2.0 * blocks.real / (n * dt)
        if target.data_domain == "time":
            level = (model * eta).sum(axis=-1) / eta.sum() / (2.0 * dt)
            return np.broadcast_to(level[..., None], pair_shape + (n,)).copy()
        layout = _layout_of(target)
        band_model = layout.bands_from_bins(model)
        shape = pair_shape + (
            layout.num_frequency_divisions + 1,
            layout.num_time_divisions,
        )
        return layout.folded(np.broadcast_to(band_model[..., :, None], shape)) / 2.0
    layout = _layout_of(source)
    model = 2.0 * layout.padded(blocks.real)
    if target.data_domain == "frequency":
        spectrum = layout.bins_from_bands(model.mean(axis=-1))
        return ((n * dt / 2.0) * spectrum).astype(np.complex128)
    if target.data_domain == "time":
        column_model = np.einsum("m,...mn->...n", layout.band_share, model)
        return layout.samples_from_columns(column_model) / (2.0 * dt)
    other = _layout_of(target)
    band_matrix = _band_matrix(layout, other)
    source_columns = layout.column_of_sample[other.column_centres]
    regridded = np.einsum("pm,...mn->...pn", band_matrix, model[..., source_columns])
    return other.folded(regridded) / 2.0
```

The file is formatter-shaped as written; `.venv/bin/python -m ruff format src/enchilada/_spectral.py` is a no-op.

- [ ] **Step 4: Add the `Covariance` overload to `transform`**

In `src/enchilada/translation.py`, replace the imports block with:

```python
from copy import deepcopy
from dataclasses import replace
from typing import overload

from enchilada._spectral import translate_blocks
from enchilada._transforms import transform_channels
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid, WDMGrid, resolve_wdm_grid
```

After the second `@overload` (the `BlockState` one, lines 48-56) add a third:

```python
@overload
def transform(
    value: Covariance,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> Covariance: ...
```

Change the implementation's signature to `value: L1Data | BlockState | Covariance` and its return annotation to `-> L1Data | BlockState | Covariance`. Replace the docstring with:

```python
    """Return an independent representation of data, a block state or a covariance.

    Args:
        value: Object to translate. A BlockState with a template needs
            reference_data because its grid is intentionally external; its
            covariance is deep-copied unchanged. A Covariance carries its own
            grid and is translated through the diagonal spectral-model rule
            (see below).
        target_domain: "time", "frequency", or "wdm".
        num_frequency_divisions: WDM frequency divisions (Nf, the row count).
            Specify this or num_time_divisions on first conversion into WDM.
        num_time_divisions: WDM time divisions. The other count is derived from
            num_time_samples; both counts must be positive even integers. Both
            may be supplied if consistent. WDM-to-WDM defaults to its current grid.
        reference_data: Required for a BlockState with a template, as the
            L1Data on the template's source grid; ignored otherwise.

    Preserves observation metadata, orbit, published physical parameters and a
    block state's covariance. Signal arrays follow dt*rfft and domania's folded
    WDM layout.

    A Covariance is read as a piecewise-constant one-sided spectral model
    S(f, t): per-sample variances on time (flat in f), per-bin variances on
    frequency (flat in t), per-cell variances on WDM; the target grid takes
    weighted means of that model with the Meyer window's band weights across
    frequency and its time weights across columns, so the result is a
    validated Covariance with the same channels. This treats every covariance
    as diagonal in the grid index on every grid: white noise round-trips
    exactly, a coloured PSD comes back band-smoothed from WDM, a time-varying
    variance comes back column-smoothed, and time <-> frequency keeps only the
    mean power. The same grid in and out returns ``value.copy()``.
    """
```

Replace the `if isinstance(value, Covariance): raise TypeError(...)` block (the first statement of the body) with nothing, and insert this branch right before the final `raise TypeError(...)`:

```python
    if isinstance(value, Covariance):
        source_grid = value.grid
        target_grid = Grid(
            target_domain,
            source_grid.num_time_samples,
            source_grid.sample_rate_hz,
            _target_grid(
                target_domain,
                source_grid.num_time_samples,
                source_grid.wdm_grid,
                num_frequency_divisions,
                num_time_divisions,
            ),
        )
        if target_grid == source_grid:
            return value.copy()
        return Covariance(
            translate_blocks(value.blocks, source_grid, target_grid),
            channel_names=value.channel_names,
            grid=target_grid,
        )
    raise TypeError("transform expects L1Data, BlockState or Covariance")
```

The local names are `source_grid`/`target_grid` because `source` is already an `L1Data` in this function (mypy refuses the reuse). The two generic checks that precede the `isinstance(value, L1Data)` branch (`target_domain not in L1Data.DOMAINS`, division counts on a non-WDM target) stay where they are and run for a Covariance too; `_target_grid` repeats them harmlessly. `resolve_wdm_grid` and `WDMGrid` stay imported for `_target_grid`.

- [ ] **Step 5: Invert the `TypeError` test**

In `tests/test_translation.py`, replace `test_transform_rejects_a_covariance` (lines 127-137) with:

```python
def test_transform_translates_a_covariance_and_refuses_other_types():
    data = observations()
    covariance = white_covariance(data, 2.0)
    spectrum = transform(covariance, "frequency")
    assert type(spectrum) is Covariance
    assert spectrum.grid == transform(data, "frequency").grid
    np.testing.assert_allclose(
        spectrum.A, white_covariance(transform(data, "frequency"), 2.0).A
    )
    cells = transform(covariance, "wdm", num_frequency_divisions=8)
    assert cells.grid == transform(data, "wdm", num_frequency_divisions=8).grid
    np.testing.assert_allclose(cells.A, 2.0 / data.fs, rtol=1e-12)
    with pytest.raises(ValueError, match="division counts apply only"):
        transform(covariance, "frequency", num_frequency_divisions=8)
    with pytest.raises(TypeError, match="expects L1Data, BlockState or Covariance"):
        transform({"A": np.zeros(4)}, "time")
```

and change the file's import line `from enchilada import BlockState, L1Data, WDMGrid` to `from enchilada import BlockState, Covariance, L1Data, WDMGrid`.

- [ ] **Step 6: Run the new tests and the suite**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_covariance_translation.py tests/test_translation.py`
Expected: all passed.

Run the full **Verify** command, mypy, ruff. Expected: green, coverage ≥ 95 %.

- [ ] **Step 7: Commit**

```bash
git branch --show-current   # wdm-home
git add src/enchilada/_spectral.py src/enchilada/translation.py tests/test_covariance_translation.py tests/test_translation.py
git commit -m "transform: translate a Covariance between grids through a diagonal spectral model" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

### Task 2: Home grid, hand-over by translation, declared roles

**Goal:** `Wheel` keeps the campaign on a WDM home grid chosen at construction, serves every block on its own grid (data by exact transforms, the noise sum by `transform(covariance, ...)`), stores templates on the home grid and noise publications natively, and reads each block's `role` attribute once at `add`; `Block`, `FixedNoiseBlock`, `EchoBlock` and `check_block` carry the role; the one-grid rule and the `role=` keyword are gone. The new behaviour is pinned by `tests/test_wheel_home_grid.py` and the five core test files are brought to the new API (the rest of the suite stays red until Tasks 3–7).

**Files:**
- Modify (replace whole file): `src/enchilada/wheel.py`, `src/enchilada/block.py`, `src/enchilada/fixed_noise.py`, `src/enchilada/testing.py`
- Create: `tests/test_wheel_home_grid.py`
- Modify (patch): `tests/test_noise_sum.py`, `tests/test_initial_sample.py`, `tests/test_testing.py`, `tests/test_typing.py`, `tests/test_fixed_noise.py`

**Acceptance Criteria:**
- [ ] `Wheel(observed_data, *, num_frequency_divisions=None, num_time_divisions=None, random_seed=None)`: time/frequency observations without a count raise `ValueError` naming both keywords; WDM observations set the home grid and disagreeing counts raise `ValueError` naming both grids; `wheel.home_grid` is the WDM `Grid`; `observed_data` is the arrival copy (bit-exact) and `working_residual`, `residual(...)`, `contribution(...)` come back on the arrival grid through one exact inverse transform.
- [ ] `add(block, *, data_domain=None, num_frequency_divisions=None, num_time_divisions=None)` reads `block.role` after the name/duplicate/`sample`/`initial_sample` checks: missing or `None` → `TypeError` "block 'x' declares no role; a Block needs name, role ('signal' or 'noise'), ..."; any other value outside `("signal", "noise")` → `ValueError` "block 'x' declares role 'nosie'; a role is 'signal' or 'noise'"; the role is recorded in `_roles` and never re-read.
- [ ] A block off the home grid receives its residual by `transform` and the covariance by `transform(sum, its grid)` (a signal block the whole sum, a noise block the others' sum, `None` when empty); on the home grid it receives deep copies. Its template is validated on its grid (`"<block>.<method> template"` context, finiteness) and transformed onto the home grid for the ledger and the signal sum; its covariance is validated by `Covariance.copy`, refused off its registration grid with the existing message, kept natively in the ledger and translated onto the home grid for the noise sum; a translation `ValueError` is re-raised as "<block>.<method> covariance does not translate onto the home grid <grid>: <error>" with full rollback.
- [ ] Role-violation messages read "declares role 'signal'" / "declares role 'noise'"; `_covariance_grid`, `_canonical_state`, `_observed_data` and both grid refusals are gone; `Role`/`ROLES` stay in `wheel.py`.
- [ ] `Block` has the `role: str` member (property under `TYPE_CHECKING`); `FixedNoiseBlock.role == "noise"`, `EchoBlock.role == "signal"`; `check_block(block, data, num_cycles=2, *, noise_blocks=(), random_seed=0, num_frequency_divisions=None, num_time_divisions=None)` refuses a `noise_blocks` entry whose role is not `"noise"` before building the Wheel.
- [ ] `tests/test_wheel_home_grid.py` (30 tests) and the five patched files pass; mypy and ruff are clean.
- [ ] Review probes (spec §6, performed by the reviewer, not committed): with the hand-over returning the home-grid sum instead of `transform(sum, its grid)`, `test_a_signal_block_off_the_home_grid_receives_exactly_the_translated_sum` fails; with the ledger storing the translated copy instead of the native publication, `test_the_ledger_keeps_the_native_publication_and_the_sum_lives_at_home` fails; with the role read at every adoption instead of at `add`, the `turncoat`/`echo` cases of `test_role_violations_are_caught` in `tests/test_wheel.py` fail. Each mutation is reverted before the review verdict.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wheel_home_grid.py tests/test_noise_sum.py tests/test_initial_sample.py tests/test_testing.py tests/test_typing.py tests/test_fixed_noise.py tests/test_covariance_translation.py tests/test_translation.py tests/test_covariance.py tests/test_sum_tree.py` → all passed (162 in the eight API files); `.venv/bin/python -m mypy` → `Success: no issues found`; `.venv/bin/python -m ruff check src tests` and `ruff format --check src tests` clean. (The rest of `tests/`, the examples, notebooks and forge are expected red until Tasks 3–7; do not touch them here.)

**Steps:**

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wheel_home_grid.py`:

```python
"""The WDM home grid: one campaign grid, blocks on any grid, the sum translated."""

import re
from dataclasses import dataclass, field

import enchilada.wheel
import numpy as np
import pytest
from enchilada import (
    Block,
    BlockState,
    Covariance,
    FixedNoiseBlock,
    Grid,
    WDMGrid,
    Wheel,
    transform,
)
from enchilada.testing import EchoBlock, check_block

from conftest import white_covariance

NF = 8  # home-grid rows for the 64-sample fixture (fs = 0.5, so dt = 2.0)
CLOSE = {"rtol": 1e-12, "atol": 1e-12}  # one exact WDM transform and back


def home_of(data):
    return transform(data, "wdm", num_frequency_divisions=NF)


def level_state(conditional_residual, value):
    """A template of `value` in every time sample, expressed on the residual's grid."""
    time_reference = transform(conditional_residual, "time")
    start = time_reference.block_state(
        {
            ch: np.full(time_reference.num_time_samples, value)
            for ch in time_reference.channel_names
        }
    )
    grid = conditional_residual.wdm_grid
    options = (
        {}
        if grid is None
        else {"num_frequency_divisions": grid.num_frequency_divisions}
    )
    return transform(
        start,
        conditional_residual.data_domain,
        reference_data=time_reference,
        **options,
    )


class LevelSignal:
    """A signal block adding `value` to every time sample; records its inputs."""

    role = "signal"

    @dataclass
    class State:
        received: list = field(default_factory=list)  # (residual, covariance)
        states: list = field(default_factory=list)  # current_block_state per sample

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.state = self.State()

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self.state.received.append((conditional_residual, covariance))
        return level_state(conditional_residual, self.value)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.state.received.append((conditional_residual, covariance))
        self.state.states.append(current_block_state)
        return level_state(conditional_residual, self.value)


class WhiteNoise:
    """A noise block publishing white noise of `level`; records its inputs."""

    role = "noise"

    def __init__(self, name, level):
        self.name = name
        self.level = level
        self.received = []  # (covariance, current_block_state or None) per call
        self.published = []  # the covariance returned by each call

    def _publish(self, conditional_residual):
        return white_covariance(conditional_residual, self.level)

    def initial_sample(self, conditional_residual, covariance, *, rng):
        self.received.append((covariance, None))
        self.published.append(self._publish(conditional_residual))
        return BlockState(covariance=self.published[-1])

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        self.received.append((covariance, current_block_state))
        self.published.append(self._publish(conditional_residual))
        return BlockState(covariance=self.published[-1])


class ColouredNoise(WhiteNoise):
    """A noise block whose variance rises along its grid's last axis."""

    def _publish(self, conditional_residual):
        shape = conditional_residual.grid.channel_shape
        ramp = self.level * (1.0 + np.linspace(0.0, 1.0, shape[-1]))
        values = np.broadcast_to(ramp, shape).astype(np.float64)
        if conditional_residual.data_domain == "frequency":
            values = values.astype(np.complex128)
        return Covariance.like(
            conditional_residual,
            {ch: values.copy() for ch in conditional_residual.channel_names},
        )


def snapshot(wheel):
    """Everything a refused adoption must leave alone."""
    total = wheel.covariance
    return {
        "names": list(wheel.ledger),
        "roles": dict(wheel._roles),
        "grids": dict(wheel._block_domains),
        "residual": wheel._working_residual.channel_data["A"].copy(),
        "covariance": None if total is None else total.A.copy(),
        "rng": wheel._rng.bit_generator.state,
        "signal_sum": wheel._signal_sum,
        "noise_sum": wheel._noise_sum,
        "ledger": {
            name: (
                None if entry.template is None else entry.template["A"].copy(),
                None if entry.covariance is None else entry.covariance.A.copy(),
                entry.parameters,
            )
            for name, entry in wheel.ledger.snapshot().items()
        },
    }


def _same_array(after, before):
    if before is None:
        assert after is None
    else:
        np.testing.assert_array_equal(after, before)


def assert_unchanged(wheel, before):
    after = snapshot(wheel)
    assert after["names"] == before["names"]
    assert after["roles"] == before["roles"]
    assert after["grids"] == before["grids"]
    assert after["rng"] == before["rng"]
    assert after["signal_sum"] is before["signal_sum"]
    assert after["noise_sum"] is before["noise_sum"]
    np.testing.assert_array_equal(after["residual"], before["residual"])
    _same_array(after["covariance"], before["covariance"])
    assert after["ledger"].keys() == before["ledger"].keys()
    for name, (template, covariance, parameters) in before["ledger"].items():
        after_template, after_covariance, after_parameters = after["ledger"][name]
        _same_array(after_template, template)
        _same_array(after_covariance, covariance)
        assert after_parameters == parameters


class TestConstruction:
    def test_time_or_frequency_observations_need_a_division_count(self, observed):
        for data in (observed, observed.to_frequency()):
            with pytest.raises(ValueError) as info:
                Wheel(data)
            message = str(info.value)
            assert "num_frequency_divisions" in message
            assert "num_time_divisions" in message
            assert f"data_domain={data.data_domain!r}" in message

    def test_the_home_grid_is_resolved_from_either_count(self, observed):
        expected = Grid(
            "wdm", 64, 0.5, WDMGrid(num_frequency_divisions=8, num_time_divisions=8)
        )
        assert Wheel(observed, num_frequency_divisions=8).home_grid == expected
        assert Wheel(observed, num_time_divisions=8).home_grid == expected
        both = Wheel(observed, num_frequency_divisions=8, num_time_divisions=8)
        assert both.home_grid == expected
        by_columns = Wheel(observed.to_frequency(), num_time_divisions=16)
        assert by_columns.home_grid.wdm_grid.num_frequency_divisions == 4
        with pytest.raises(ValueError, match="product"):
            Wheel(observed, num_frequency_divisions=8, num_time_divisions=4)
        with pytest.raises(ValueError, match="positive even"):
            Wheel(observed, num_frequency_divisions=3)
        with pytest.raises(ValueError, match="divide"):
            Wheel(observed, num_frequency_divisions=6)

    def test_wdm_observations_bring_their_own_home_grid(self, observed):
        cells = home_of(observed)
        assert Wheel(cells).home_grid == cells.grid
        assert Wheel(cells, num_frequency_divisions=8).home_grid == cells.grid
        assert Wheel(cells, num_time_divisions=8).home_grid == cells.grid
        with pytest.raises(ValueError) as info:
            Wheel(cells, num_frequency_divisions=4)
        message = str(info.value)
        assert "num_frequency_divisions=8" in message
        assert "num_frequency_divisions=4" in message

    def test_the_old_signatures_stay_refused(self, observed):
        with pytest.raises(TypeError, match="positional"):
            Wheel(observed, 8)
        wheel = Wheel(observed, num_frequency_divisions=8)
        assert not hasattr(wheel, "_covariance_grid")
        assert not hasattr(wheel, "_observed_data")
        with pytest.raises(TypeError, match="role"):
            wheel.add(EchoBlock("echo"), role="signal")

    @pytest.mark.parametrize("domain", ["time", "frequency", "wdm"])
    def test_public_views_come_back_on_the_arrival_grid(self, observed, domain):
        data = (
            observed
            if domain == "time"
            else observed.to_frequency()
            if domain == "frequency"
            else home_of(observed)
        )
        wheel = Wheel(data, num_frequency_divisions=NF)
        assert wheel.observed_data.grid == data.grid
        np.testing.assert_array_equal(
            wheel.observed_data.channel_data["A"], data.channel_data["A"]
        )
        for view in (wheel.working_residual, wheel.residual()):
            assert view.grid == data.grid
            assert view.channel_data["A"].dtype == data.channel_data["A"].dtype
            np.testing.assert_allclose(
                view.channel_data["A"], data.channel_data["A"], **CLOSE
            )
            assert not np.shares_memory(
                view.channel_data["A"], wheel._working_residual.channel_data["A"]
            )
        if domain == "wdm":  # on the home grid the views are bit-exact copies
            np.testing.assert_array_equal(
                wheel.residual().channel_data["A"], data.channel_data["A"]
            )


class TestMixedGridCampaign:
    def campaign(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=1)
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        blocks = {
            "time": LevelSignal("time", 1.0),
            "spectral": LevelSignal("spectral", 2.0),
            "home": LevelSignal("home", 4.0),
            "coarse": LevelSignal("coarse", 8.0),
        }
        wheel.add(blocks["time"])
        wheel.add(blocks["spectral"], data_domain="frequency")
        wheel.add(blocks["home"], data_domain="wdm")
        wheel.add(blocks["coarse"], data_domain="wdm", num_frequency_divisions=4)
        return wheel, blocks

    def test_blocks_on_four_grids_share_one_campaign(self, observed):
        wheel, blocks = self.campaign(observed)
        wheel.run(2)
        total = 1.0 + 2.0 + 4.0 + 8.0
        np.testing.assert_allclose(
            wheel.residual().channel_data["A"],
            observed.channel_data["A"] - total,
            **CLOSE,
        )
        np.testing.assert_allclose(
            wheel.residual("coarse").channel_data["A"],
            observed.channel_data["A"] - (total - 8.0),
            **CLOSE,
        )
        for name, block in blocks.items():
            contribution = wheel.contribution(name)["E"]
            assert contribution.shape == (64,)  # the arrival grid
            np.testing.assert_allclose(contribution, block.value, rtol=1e-12)
            residual, _ = block.state.received[-1]
            expected_grid = wheel._block_domains[name]
            assert residual.grid == expected_grid
            np.testing.assert_allclose(
                transform(residual, "time").channel_data["A"],
                observed.channel_data["A"] - (total - block.value),
                **CLOSE,
            )
            current = block.state.states[-1]
            assert current.template["A"].shape == expected_grid.channel_shape
            np.testing.assert_allclose(
                current.template["A"],
                level_state(residual, block.value).template["A"],
                **CLOSE,
            )
        assert blocks["spectral"].state.received[-1][0].data_domain == "frequency"
        assert (
            blocks["coarse"].state.received[-1][0].wdm_grid.num_frequency_divisions == 4
        )
        assert blocks["home"].state.received[-1][0].grid == wheel.home_grid

    def test_the_ledger_keeps_templates_on_the_home_grid(self, observed):
        wheel, blocks = self.campaign(observed)
        for name, block in blocks.items():
            template = wheel.ledger[name].template["A"]
            assert template.shape == (NF, 64 // NF)
            np.testing.assert_allclose(
                template,
                level_state(home_of(observed), block.value).template["A"],
                **CLOSE,
            )
        assert wheel.ledger["floor"].template is None

    def test_every_block_receives_the_white_floor_on_its_grid(self, observed):
        wheel, blocks = self.campaign(observed)
        wheel.run(1)
        for block in blocks.values():
            for residual, covariance in block.state.received:
                assert type(covariance) is Covariance
                assert covariance.grid == residual.grid
                assert not covariance.blocks.flags.writeable
                np.testing.assert_allclose(
                    covariance.A, white_covariance(residual, 1.0).A, rtol=1e-12
                )
        assert wheel.covariance.grid == wheel.home_grid
        np.testing.assert_allclose(wheel.covariance.A, 1.0 / observed.fs, rtol=1e-12)


class TestNoiseSum:
    def test_a_signal_block_off_the_home_grid_receives_exactly_the_translated_sum(
        self, observed
    ):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        wheel.add(WhiteNoise("white", 1.0))
        wheel.add(ColouredNoise("coloured", 2.0), data_domain="frequency")
        spectral, coarse = LevelSignal("spectral", 1.0), LevelSignal("coarse", 1.0)
        wheel.add(spectral, data_domain="frequency")
        wheel.add(coarse, data_domain="wdm", num_frequency_divisions=4)
        wheel.run(1)
        total = wheel.covariance
        assert total.grid == wheel.home_grid
        for block, target, options in (
            (spectral, "frequency", {}),
            (coarse, "wdm", {"num_frequency_divisions": 4}),
        ):
            expected = transform(total, target, **options)
            for _, received in block.state.received:
                assert received.grid == expected.grid
                np.testing.assert_array_equal(received.blocks, expected.blocks)

    def test_a_noise_block_receives_the_others_translated_and_its_own_as_published(
        self, observed
    ):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        white = WhiteNoise("white", 1.0)
        coloured = ColouredNoise("coloured", 2.0)
        wheel.add(white)
        wheel.add(coloured, data_domain="frequency")
        wheel.run(2)
        spectrum = observed.to_frequency()
        assert white.received[0][0] is None  # alone at registration
        assert len(coloured.received) == len(coloured.published) == 3
        for received, _ in coloured.received:  # white's floor on its grid
            assert received.grid == spectrum.grid
            np.testing.assert_allclose(
                received.A, white_covariance(spectrum, 1.0).A, rtol=1e-12
            )
        for (_, current), published in zip(
            coloured.received[1:], coloured.published[:-1], strict=True
        ):
            assert current.template is None
            assert current.covariance.grid == spectrum.grid
            np.testing.assert_array_equal(current.covariance.blocks, published.blocks)
        for (received, _), published in zip(
            white.received[1:], coloured.published[:-1], strict=True
        ):  # coloured's accepted model: onto the home grid, then to the time grid
            assert received.grid == observed.grid
            via_home = transform(published, "wdm", num_frequency_divisions=NF)
            np.testing.assert_array_equal(
                received.blocks, transform(via_home, "time").blocks
            )

    def test_the_ledger_keeps_the_native_publication_and_the_sum_lives_at_home(
        self, observed
    ):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        wheel.add(WhiteNoise("white", 1.0))
        coloured = ColouredNoise("coloured", 2.0)
        wheel.add(coloured, data_domain="frequency")
        native = wheel.ledger["coloured"].covariance
        assert native.grid == observed.to_frequency().grid
        np.testing.assert_array_equal(native.blocks, coloured.published[-1].blocks)
        assert wheel.ledger["white"].covariance.grid == observed.grid
        assert wheel.covariance.grid == wheel.home_grid
        expected = transform(
            wheel.ledger["white"].covariance, "wdm", num_frequency_divisions=NF
        )
        expected = expected + transform(native, "wdm", num_frequency_divisions=NF)
        np.testing.assert_allclose(wheel.covariance.blocks, expected.blocks, rtol=1e-12)
        # the hand-over copies are independent of the stored sum
        total = wheel.covariance
        total.blocks.setflags(write=True)
        total.A[:] = 99.0
        np.testing.assert_allclose(wheel.covariance.blocks, expected.blocks, rtol=1e-12)

    def test_a_republished_covariance_replaces_its_leaf(self, observed):
        class Growing(WhiteNoise):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                self.level += 1.0
                return super().sample(
                    conditional_residual, covariance, current_block_state, rng=rng
                )

        wheel = Wheel(observed, num_frequency_divisions=NF)
        wheel.add(Growing("growing", 1.0), data_domain="frequency")
        wheel.add(FixedNoiseBlock("fixed", white_covariance(observed, 10.0)))
        per_cell = 1.0 / observed.fs
        np.testing.assert_allclose(wheel.covariance.A, 11.0 * per_cell, rtol=1e-12)
        wheel.run(1)
        np.testing.assert_allclose(wheel.covariance.A, 12.0 * per_cell, rtol=1e-12)
        wheel.run(1)
        np.testing.assert_allclose(wheel.covariance.A, 13.0 * per_cell, rtol=1e-12)
        np.testing.assert_allclose(
            wheel.ledger["growing"].covariance.A,
            white_covariance(observed.to_frequency(), 3.0).A,
        )

    def test_wdm_observations_with_home_grid_blocks_are_bit_exact(self, observed):
        cells = home_of(observed)
        wheel = Wheel(cells)
        noise, signal = WhiteNoise("noise", 1.0), LevelSignal("signal", 1.0)
        wheel.add(noise)
        wheel.add(signal, data_domain="wdm")
        wheel.run(1)
        residual, covariance = signal.state.received[-1]
        assert residual.grid == cells.grid
        template = signal.state.states[-1].template["A"]
        np.testing.assert_array_equal(
            residual.channel_data["A"], cells.channel_data["A"]
        )  # its own template is left in
        np.testing.assert_array_equal(
            wheel.residual().channel_data["A"], cells.channel_data["A"] - template
        )
        np.testing.assert_array_equal(covariance.blocks, noise.published[-1].blocks)
        np.testing.assert_array_equal(
            wheel.covariance.blocks, noise.published[-1].blocks
        )
        np.testing.assert_array_equal(wheel.contribution("signal")["A"], template)


class TestDeclaredRoles:
    def test_the_role_is_read_once_at_add(self, observed):
        wheel = Wheel(observed, num_frequency_divisions=NF)
        block = LevelSignal("turncoat", 1.0)
        wheel.add(block)
        block.role = "noise"
        assert wheel._roles == {"turncoat": "signal"}
        wheel.run(1)  # still adopted as a signal block
        np.testing.assert_allclose(wheel.contribution("turncoat")["A"], 1.0, rtol=1e-12)
        block.role = "nosie"
        wheel.run(1)
        assert len(block.state.states) == 2

    def test_a_block_without_a_role_is_refused_after_the_protocol_checks(
        self, observed
    ):
        class NoRole:
            name = "plain"

            def initial_sample(self, conditional_residual, covariance, *, rng):
                raise AssertionError(
                    "add must refuse the role before starting the block"
                )

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                raise AssertionError("unreachable")

        class NoneRole(NoRole):
            role = None

        class NoSample:
            name = "nosample"

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        before = snapshot(wheel)
        for block in (NoRole(), NoneRole()):
            with pytest.raises(TypeError) as info:
                wheel.add(block)
            message = str(info.value)
            assert message.startswith("block 'plain' declares no role; ")
            assert "role ('signal' or 'noise')" in message
            assert "initial_sample(conditional_residual, covariance, *, rng)" in message
        with pytest.raises(TypeError, match="does not implement sample"):
            wheel.add(NoSample())  # the protocol checks still come first
        assert not isinstance(NoRole(), Block)
        assert isinstance(LevelSignal("x", 1.0), Block)
        assert_unchanged(wheel, before)

    @pytest.mark.parametrize("declared", ["nosie", "Signal", "noise ", "", 1])
    def test_an_invalid_role_is_refused_with_its_value(self, observed, declared):
        block = LevelSignal("typo", 1.0)
        block.role = declared
        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        before = snapshot(wheel)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"block 'typo' declares role {declared!r}; "
                "a role is 'signal' or 'noise'"
            ),
        ):
            wheel.add(block)
        assert block.state.received == []
        assert_unchanged(wheel, before)

    def test_check_block_refuses_a_signal_block_among_the_noise_blocks(self, observed):
        class NeverStarted(LevelSignal):
            def initial_sample(self, conditional_residual, covariance, *, rng):
                raise AssertionError("check_block must refuse noise_blocks first")

        with pytest.raises(
            ValueError,
            match=re.escape(
                "noise_blocks entry 'echo' declares role 'signal'; every block in "
                "noise_blocks must declare role 'noise'"
            ),
        ):
            check_block(
                NeverStarted("under_test", 1.0),
                observed,
                noise_blocks=(EchoBlock("echo"),),
                num_frequency_divisions=NF,
            )

    def test_check_block_passes_the_division_counts_to_the_wheel(self, observed):
        check_block(EchoBlock("echo"), observed, num_frequency_divisions=NF)
        check_block(EchoBlock("echo"), observed, num_time_divisions=16)
        check_block(EchoBlock("echo"), home_of(observed))
        with pytest.raises(ValueError, match="num_frequency_divisions or num_time"):
            check_block(EchoBlock("echo"), observed)
        with pytest.raises(ValueError, match="num_frequency_divisions=4"):
            check_block(EchoBlock("echo"), home_of(observed), num_frequency_divisions=4)


class TestRefusals:
    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_noise_block_publishing_off_its_grid_is_refused_with_rollback(
        self, observed, method
    ):
        class OffGrid(WhiteNoise):
            def __init__(self, name, level, offend_in):
                super().__init__(name, level)
                self.offend_in = offend_in

            def _publish(self, conditional_residual, offending):
                data = (
                    transform(conditional_residual, "wdm", num_frequency_divisions=4)
                    if offending
                    else conditional_residual
                )
                return BlockState(covariance=white_covariance(data, self.level))

            def initial_sample(self, conditional_residual, covariance, *, rng):
                return self._publish(
                    conditional_residual, self.offend_in == "initial_sample"
                )

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return self._publish(conditional_residual, self.offend_in == "sample")

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        block = OffGrid("spectral", 1.0, method)
        if method == "sample":
            wheel.add(block, data_domain="frequency")
        wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 1.0)))
        before = snapshot(wheel)
        with pytest.raises(ValueError) as info:
            if method == "initial_sample":
                wheel.add(block, data_domain="frequency")
            else:
                wheel.run(1)
        message = str(info.value)
        assert message.startswith(f"spectral.{method} covariance is on grid ")
        assert "data_domain='wdm'" in message and "data_domain='frequency'" in message
        assert (
            "publish the covariance on the grid of the data the block received"
            in message
        )
        assert_unchanged(wheel, before)

    def test_a_covariance_that_fails_to_translate_is_refused_with_rollback(
        self, observed, monkeypatch
    ):
        real_transform = enchilada.wheel.transform

        def failing(value, target_domain, **options):
            if isinstance(value, Covariance) and target_domain == "wdm":
                raise ValueError("synthetic: the translated blocks must be finite")
            return real_transform(value, target_domain, **options)

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(
            FixedNoiseBlock("floor", white_covariance(observed, 1.0))
        )  # time grid
        before = snapshot(wheel)
        monkeypatch.setattr(enchilada.wheel, "transform", failing)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "spectral.initial_sample covariance does not translate onto the "
                f"home grid {wheel.home_grid}: synthetic: the translated blocks "
                "must be finite"
            ),
        ):
            wheel.add(WhiteNoise("spectral", 1.0), data_domain="frequency")
        assert_unchanged(wheel, before)
        assert "spectral" not in wheel._roles
        with pytest.raises(
            ValueError, match="floor.sample covariance does not translate"
        ):
            wheel.run(1)  # floor republishes on the time grid: translated at adoption
        assert_unchanged(wheel, before)

    @pytest.mark.parametrize("method", ["initial_sample", "sample"])
    def test_a_template_is_validated_on_the_block_grid(self, observed, method):
        class WrongShape(LevelSignal):
            def __init__(self, name, offend_in):
                super().__init__(name, 1.0)
                self.offend_in = offend_in

            def _publish(self, conditional_residual, offending):
                if not offending:
                    return level_state(conditional_residual, 1.0)
                return BlockState(
                    template={
                        ch: np.zeros(64) for ch in conditional_residual.channel_names
                    }
                )

            def initial_sample(self, conditional_residual, covariance, *, rng):
                return self._publish(
                    conditional_residual, self.offend_in == "initial_sample"
                )

            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                return self._publish(conditional_residual, self.offend_in == "sample")

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        block = WrongShape("spectral", method)
        if method == "sample":
            wheel.add(block, data_domain="frequency")
        wheel.add(LevelSignal("time", 1.0))
        before = snapshot(wheel)
        with pytest.raises(
            ValueError,
            match=re.escape(f"spectral.{method} template") + ".*length 64, expected 33",
        ):
            if method == "initial_sample":
                wheel.add(block, data_domain="frequency")
            else:
                wheel.run(1)
        assert_unchanged(wheel, before)

    def test_a_non_finite_template_is_refused_before_it_is_transformed(self, observed):
        class NaNSignal(LevelSignal):
            def sample(
                self, conditional_residual, covariance, current_block_state, *, rng
            ):
                state = level_state(conditional_residual, 1.0)
                state.template["A"][0] = np.nan
                return state

        wheel = Wheel(observed, num_frequency_divisions=NF, random_seed=3)
        wheel.add(NaNSignal("spectral", 1.0), data_domain="frequency")
        before = snapshot(wheel)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "spectral.sample returned non-finite samples in channel 'A'"
            ),
        ):
            wheel.run(1)
        assert_unchanged(wheel, before)
```

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wheel_home_grid.py 2>&1 | tail -3`
Expected: errors and failures (`Wheel.__init__() got an unexpected keyword argument 'num_frequency_divisions'`, `add() missing 1 required keyword-only argument: 'role'`).

- [ ] **Step 2: Replace `src/enchilada/wheel.py`**

```python
"""Orchestrator-owned data, accepted physical estimates, and the two sums."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Literal

import numpy as np

from enchilada._sum_tree import NoiseSum, Signal, SignalSum
from enchilada._transforms import transform_channels
from enchilada.block import Block
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid, resolve_wdm_grid
from enchilada.ledger import Ledger
from enchilada.translation import _target_grid, transform

Role = Literal["signal", "noise"]
ROLES: tuple[Role, ...] = ("signal", "noise")

_PROTOCOL = (
    "a Block needs name, role ('signal' or 'noise'), "
    "initial_sample(conditional_residual, covariance, *, rng) and "
    "sample(conditional_residual, covariance, current_block_state, *, rng)."
)


class Wheel:
    """Run blocks while owning their accepted physical estimates and shared data.

    The campaign lives on one WDM grid, the home grid, chosen at construction:
    time or frequency observations are transformed onto it once, exactly, and
    WDM observations bring their own. The pristine observation is kept twice,
    as it arrived (for `observed_data`) and on the home grid (the values every
    residual is built from). A second home-grid L1Data stores the full working
    residual, rebuilt after every accepted update. For block i, the input is
    d - sum(j != i, T_j); the block's own signal remains in its input. An
    absent TDI signal contributes zero, including when a block previously
    returned a signal. Ledger holds complete BlockState snapshots: a signal
    block's template on the home grid, a noise block's covariance as it was
    published. The public data views (`observed_data`, `working_residual`,
    `residual`, `contribution`) come back on the arrival grid through one exact
    inverse transform; `covariance` is on the home grid. Public data and
    covariance access returns copies. The ledger property returns a live
    handle; indexing and snapshot() copy its contents. Treat orbit resources as
    immutable.

    Every block declares its role in the attribute `role`, "signal" or
    "noise", read once at `add`. Signal blocks publish templates, which Wheel
    sums into the residuals; noise blocks publish covariances, which Wheel sums
    into the campaign covariance. A signal block receives the sum over every
    noise block (None until one has published); a noise block receives the sum
    over the other noise blocks (None when it is alone) and its own last
    publication on current_block_state.covariance. A noise block must return a
    covariance on every call and may not return a template; a signal block may
    not return a covariance.

    A block is served on the grid it asks for at `add`: its residual and
    current state by exact data transforms, the covariance it receives by
    `transform(covariance, ...)`, which reads every covariance as a diagonal
    spectral model (see `transform`). A template it returns is transformed
    onto the home grid and a covariance it returns is translated onto the home
    grid before entering the sums; the ledger keeps the covariance as
    published. A block on the home grid pays no transform. The diagonal
    approximation behind the covariance translation is the user's to assess;
    Wheel does not restrict grid choices on its account.

    Registration calls the block's `initial_sample` with the residual of the
    blocks registered so far on the block's grid, the covariance its role
    receives, and the campaign RNG, then adopts the returned state like a
    `sample` return. Each cycle calls sample with those inputs and the
    current_block_state from the ledger. Sampling receives the supplied RNG and
    independent input snapshots. Failed calls/validation/copying leave the
    ledger, both sums and the Wheel RNG unchanged. Blocks own their sampler
    state and diagnostics; Wheel does not roll back changes to those internals.
    Use separate block instances for separate campaigns. Earlier successful
    updates in a failed cycle remain committed. Exceptions in on_cycle_complete
    occur after that entire cycle has committed.
    """

    def __init__(
        self,
        observed_data: L1Data,
        *,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize a campaign from observations on a WDM home grid.

        Args:
            observed_data: L1 observations and shared run settings. The Wheel
                keeps an independent arrival-grid copy and a home-grid copy.
            num_frequency_divisions: Row count of the home grid. Time or
                frequency observations need this or num_time_divisions; WDM
                observations supply their own grid, and counts given here
                must agree with it.
            num_time_divisions: Column count of the home grid. The missing
                count follows from the observation length; both counts must
                be positive even integers whose product is the sample count.
            random_seed: Seed for the campaign's NumPy generator. None requests
                nondeterministic initialization.

        The campaign has no covariance until a noise block publishes one at
        registration; a known noise model is a FixedNoiseBlock.
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
        arrival = self._copy_data(observed_data)
        counts_given = (
            num_frequency_divisions is not None or num_time_divisions is not None
        )
        if arrival.data_domain == "wdm":
            if counts_given:
                requested = resolve_wdm_grid(
                    arrival.num_time_samples,
                    num_frequency_divisions=num_frequency_divisions,
                    num_time_divisions=num_time_divisions,
                )
                if requested != arrival.wdm_grid:
                    raise ValueError(
                        f"observed_data is on WDM grid {arrival.wdm_grid}, which "
                        f"is the home grid, but {requested} was requested; omit "
                        "the division counts or pass ones that agree"
                    )
            home = self._copy_data(arrival)
        else:
            if not counts_given:
                raise ValueError(
                    f"observed_data is on data_domain={arrival.data_domain!r}; "
                    "the campaign lives on a WDM home grid, so pass "
                    "num_frequency_divisions or num_time_divisions to choose it"
                )
            home = transform(
                arrival,
                "wdm",
                num_frequency_divisions=num_frequency_divisions,
                num_time_divisions=num_time_divisions,
            )
        self._arrival_data = arrival
        self._home_data = home
        self._working_residual = self._copy_data(home)
        self._ledger = Ledger()
        self._blocks: list[tuple[str, Block]] = []
        self._block_indices: dict[str, int] = {}
        self._block_domains: dict[str, Grid] = {}
        self._roles: dict[str, Role] = {}
        self._signal_sum = SignalSum()
        self._noise_sum = NoiseSum()
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

    @staticmethod
    def _frequency_divisions(grid: Grid) -> int | None:
        if grid.wdm_grid is None:
            return None
        return grid.wdm_grid.num_frequency_divisions

    @property
    def home_grid(self) -> Grid:
        """The WDM grid the campaign lives on."""
        return self._home_data.grid

    @property
    def observed_data(self) -> L1Data:
        """A defensive copy of the pristine observation, as it arrived."""
        return self._copy_data(self._arrival_data)

    @property
    def working_residual(self) -> L1Data:
        """A copy of the current full residual on the arrival grid."""
        return self._on_grid(self._working_residual, self._arrival_data.grid)

    @property
    def covariance(self) -> Covariance | None:
        """The sum over every noise block's covariance, on the home grid.

        An independent copy; None until a noise block has published.
        """
        total = self._noise_sum.total
        if total is None:
            return None
        # Leaves were validated at adoption and sums were built by
        # Covariance.__add__ through the constructor: restore unvalidated.
        return deepcopy(total)

    @property
    def ledger(self) -> Ledger:
        """Live state store; indexing copies entries, snapshot() copies all entries."""
        return self._ledger

    def _on_grid(self, home_data: L1Data, grid: Grid) -> L1Data:
        """Home-grid data on `grid`: an independent copy or one exact transform."""
        if grid == self._home_data.grid:
            return self._copy_data(home_data)
        return transform(
            home_data,
            grid.data_domain,
            num_frequency_divisions=self._frequency_divisions(grid),
        )

    def add(
        self,
        block_to_register: Block,
        *,
        data_domain: str | None = None,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
    ) -> None:
        """Register a block under its declared role and adopt its initial_sample.

        block_to_register must have a unique name, a `role` attribute that is
        "signal" or "noise", and implement initial_sample and sample. The role
        is read here, once. A signal block publishes a template and never a
        covariance; a noise block publishes a covariance on every call and
        never a template. Failed registration leaves the campaign unchanged:
        an exception from initial_sample, a return that is not a BlockState,
        an off-grid or non-finite template, a field the role forbids or lacks,
        a covariance on another grid than the block's, a covariance that does
        not translate onto the home grid, or a state that cannot be
        deep-copied leaves the ledger, the block list, the residual, both sums
        and the Wheel RNG as they were.

        initial_sample receives the current working residual (the observations
        minus the adopted starts of every block registered so far) on this
        block's grid, the covariance its role receives (the sum over every
        noise block for a signal block, the sum over the other noise blocks for
        a noise block; None while that sum is empty; translated to this block's
        grid when that is not the home grid) and a copy of the campaign RNG
        that is committed with the state. Blocks therefore start in
        registration order: register the noise blocks before the blocks whose
        starts need their covariance. The returned state is validated and
        adopted exactly like a sample return: its template must be on this
        block's grid, and Wheel transforms it onto the home grid for storage;
        its covariance must be on this block's grid, and Wheel translates it
        onto the home grid for the sum while the ledger keeps the original. A
        block that must start at a chosen point takes that point in its
        constructor and returns it from initial_sample.

        data_domain selects time, frequency or wdm for this block's residual,
        current state and published covariance. None means the arrival grid of
        the observations. "wdm" without division counts means the home grid,
        which costs no transform; with a count it is another WDM grid, and the
        other count follows from the observation length (both must be even).
        """
        block_grid = self._block_grid(
            data_domain, num_frequency_divisions, num_time_divisions
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
                f"block {block_name!r} does not implement sample; {_PROTOCOL}"
            )
        if not callable(getattr(block_to_register, "initial_sample", None)):
            raise TypeError(
                f"block {block_name!r} does not implement initial_sample; {_PROTOCOL}"
            )
        role = self._declared_role(block_to_register, block_name)
        conditional_residual, covariance = self._block_inputs(
            block_name, block_grid, role=role, registered=False
        )
        candidate_rng = deepcopy(self._rng)
        new_block_state: object = block_to_register.initial_sample(
            conditional_residual=conditional_residual,
            covariance=covariance,
            rng=candidate_rng,
        )
        self._adopt(
            block_name,
            new_block_state,
            "initial_sample",
            candidate_rng,
            role=role,
            block_to_register=block_to_register,
            block_grid=block_grid,
        )

    def _block_grid(
        self,
        data_domain: str | None,
        num_frequency_divisions: int | None,
        num_time_divisions: int | None,
    ) -> Grid:
        """The grid a block asks for: the arrival grid, or one on the home record."""
        if data_domain is None:
            if num_frequency_divisions is not None or num_time_divisions is not None:
                raise ValueError("WDM division counts require data_domain='wdm'")
            return self._arrival_data.grid
        home = self._home_data.grid
        return Grid(
            data_domain,
            home.num_time_samples,
            home.sample_rate_hz,
            _target_grid(
                data_domain,
                home.num_time_samples,
                home.wdm_grid,
                num_frequency_divisions,
                num_time_divisions,
            ),
        )

    @staticmethod
    def _declared_role(block_to_register: Block, block_name: str) -> Role:
        """The block's `role` attribute, narrowed to a Role."""
        declared = getattr(block_to_register, "role", None)
        if declared is None:
            raise TypeError(f"block {block_name!r} declares no role; {_PROTOCOL}")
        if declared not in ROLES:
            raise ValueError(
                f"block {block_name!r} declares role {declared!r}; "
                "a role is 'signal' or 'noise'"
            )
        return "signal" if declared == "signal" else "noise"

    def _block_inputs(
        self, block_name: str, block_grid: Grid, *, role: Role, registered: bool
    ) -> tuple[L1Data, Covariance | None]:
        """Conditional residual on the block's grid, and the covariance its role gets.

        A registered block's residual leaves its own signal in; a block being
        registered has no signal yet, so it receives the working residual. A
        signal block receives the sum over every noise block; a noise block
        receives the sum over the other noise blocks, which is the whole sum
        while it is being registered because its own slot does not exist yet.
        Both are formed on the home grid and brought to the block's grid once:
        the residual by an exact transform, the covariance by translation.
        """
        residual_on_home = (
            self._form_residual(
                self._signal_sum.excluding(self._block_indices[block_name])
            )
            if registered
            else self._working_residual
        )
        conditional_residual = self._on_grid(residual_on_home, block_grid)
        if role == "noise" and registered:
            total = self._noise_sum.excluding(self._block_indices[block_name])
        else:
            total = self._noise_sum.total
        if total is None:
            covariance = None
        elif block_grid == self._home_data.grid:
            # Stored covariances were validated when published or summed:
            # restore an independent read-only copy without revalidating.
            covariance = deepcopy(total)
        else:
            covariance = transform(
                total,
                block_grid.data_domain,
                num_frequency_divisions=self._frequency_divisions(block_grid),
            )
        return conditional_residual, covariance

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
                role = self._roles[block_name]
                conditional_residual, covariance = self._block_inputs(
                    block_name, block_grid, role=role, registered=True
                )
                current_block_state = self._ledger[block_name]
                if block_grid != self._home_data.grid:
                    current_block_state = transform(
                        current_block_state,
                        block_grid.data_domain,
                        reference_data=self._home_data,
                        num_frequency_divisions=self._frequency_divisions(block_grid),
                    )
                new_block_state: object = block.sample(
                    conditional_residual=conditional_residual,
                    covariance=covariance,
                    current_block_state=current_block_state,
                    rng=candidate_rng,
                )
                self._adopt(
                    block_name,
                    new_block_state,
                    "sample",
                    candidate_rng,
                    role=role,
                    block_grid=block_grid,
                )
            if on_cycle_complete is not None:
                on_cycle_complete(cycle_index, self)

    def _form_residual(self, signal_sum: Signal) -> L1Data:
        """Subtract an aggregate signal from the home-grid observations."""
        channel_data = {}
        for ch, observed_channel_values in self._home_data.channel_data.items():
            if signal_sum is None:
                residual_channel_values = observed_channel_values.copy()
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_channel_values = observed_channel_values - signal_sum[ch]
            if not np.isfinite(residual_channel_values).all():
                raise ValueError(f"residual in channel {ch!r} became non-finite")
            channel_data[ch] = residual_channel_values
        return replace(self._home_data, channel_data=channel_data)

    def residual(self, exclude_block_name: str | None = None) -> L1Data:
        """Return a full or conditional residual on the arrival grid, as a copy.

        exclude_block_name identifies a registered block whose signal is left
        in the observations. None subtracts every block's signal.
        """
        if exclude_block_name is None:
            return self.working_residual
        self._require_block_name(exclude_block_name)
        return self._on_grid(
            self._form_residual(
                self._signal_sum.excluding(self._block_indices[exclude_block_name])
            ),
            self._arrival_data.grid,
        )

    def contribution(self, block_name: str) -> dict[str, np.ndarray]:
        """Copy the registered block's summed TDI signal on the arrival grid.

        block_name must identify a registered block. An absent contribution
        returns fresh zero arrays on the arrival grid.
        """
        self._require_block_name(block_name)
        signal = self._ledger._states_by_block_name[block_name].template
        arrival = self._arrival_data
        if signal is None:
            return {ch: np.zeros_like(arr) for ch, arr in arrival.channel_data.items()}
        home = self._home_data
        if arrival.grid == home.grid:
            return {ch: arr.copy() for ch, arr in signal.items()}
        return transform_channels(
            signal,
            source_domain="wdm",
            target_domain=arrival.data_domain,
            num_time_samples=home.num_time_samples,
            sample_rate_hz=home.sample_rate_hz,
            source_wdm_grid=home.wdm_grid,
            target_wdm_grid=arrival.wdm_grid,
        )

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
        role: Role,
        block_to_register: Block | None = None,
        block_grid: Grid,
    ) -> None:
        """Validate a candidate state and RNG before committing campaign state.

        source_method_name names the block method whose return is being adopted,
        for diagnostics. role decides which fields the state may and must carry.
        block_to_register is supplied only when adopting a new block's initial state.
        block_grid is the grid the block is registered on: a published template
        or covariance must be on it, and is brought onto the home grid here.
        """
        if not isinstance(new_block_state, BlockState):
            raise TypeError(
                f"{block_name}.{source_method_name} must return a BlockState, got "
                f"{type(new_block_state).__name__}"
            )
        # Snapshot BEFORE validation so subsequent use only touches owned state.
        candidate_block_state = deepcopy(new_block_state)
        candidate_block_state.__post_init__()
        home = self._home_data
        signal = candidate_block_state.template
        if signal is not None:
            check_on_grid(
                signal,
                channel_names=home.channel_names,
                num_time_samples=home.num_time_samples,
                data_domain=block_grid.data_domain,
                wdm_grid=block_grid.wdm_grid,
                context_label=f"{block_name}.{source_method_name} template",
            )
            for ch, arr in signal.items():
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"{block_name}.{source_method_name} returned non-finite "
                        f"samples in channel {ch!r}"
                    )
        block_index = (
            len(self._blocks)
            if block_to_register is not None
            else self._block_indices[block_name]
        )
        candidate_noise_sum = self._noise_sum
        if role == "signal":
            if candidate_block_state.covariance is not None:
                raise ValueError(
                    f"{block_name}.{source_method_name} returned a covariance but "
                    f"{block_name!r} declares role 'signal'"
                )
        else:
            if signal is not None:
                raise ValueError(
                    f"{block_name}.{source_method_name} returned a template but "
                    f"{block_name!r} declares role 'noise'"
                )
            if candidate_block_state.covariance is None:
                raise ValueError(
                    f"{block_name}.{source_method_name} must return a covariance "
                    f"because {block_name!r} declares role 'noise'"
                )
            # Revalidate: callers can edit arrays inside a frozen dataclass.
            # Unbound call: copy without trusting a subclass's overridden methods.
            candidate_covariance = Covariance.copy(candidate_block_state.covariance)
            if candidate_covariance.channel_names != tuple(home.channel_names):
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance channel_names "
                    f"{candidate_covariance.channel_names} do not match the data's "
                    f"{tuple(home.channel_names)}"
                )
            if candidate_covariance.grid != block_grid:
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance is on grid "
                    f"{candidate_covariance.grid} but the block is registered on "
                    f"grid {block_grid}; publish the covariance on the grid of the "
                    "data the block received"
                )
            if block_grid == home.grid:
                home_covariance = candidate_covariance
            else:
                try:
                    home_covariance = transform(
                        candidate_covariance,
                        "wdm",
                        num_frequency_divisions=self._frequency_divisions(home.grid),
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{block_name}.{source_method_name} covariance does not "
                        f"translate onto the home grid {home.grid}: {error}"
                    ) from error
            # The ledger keeps the publication as made; the sum takes it on home.
            candidate_block_state = replace(
                candidate_block_state, covariance=candidate_covariance
            )
            candidate_noise_sum = self._noise_sum.with_covariance(
                block_index, home_covariance
            )
        if signal is not None and block_grid != home.grid:
            signal = transform_channels(
                signal,
                source_domain=block_grid.data_domain,
                target_domain="wdm",
                num_time_samples=home.num_time_samples,
                sample_rate_hz=home.sample_rate_hz,
                source_wdm_grid=block_grid.wdm_grid,
                target_wdm_grid=home.wdm_grid,
            )
            candidate_block_state = replace(candidate_block_state, template=signal)
        candidate_block_states = {
            **self._ledger._states_by_block_name,
            block_name: candidate_block_state,
        }
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
        candidate_roles = (
            self._roles
            if block_to_register is None
            else {**self._roles, block_name: role}
        )
        next_rng = deepcopy(candidate_rng)
        # All potentially failing work is done.
        self._ledger._states_by_block_name = candidate_block_states
        self._working_residual = candidate_residual
        self._signal_sum = candidate_signal_sum
        self._noise_sum = candidate_noise_sum
        self._block_indices = candidate_block_indices
        self._block_domains = candidate_block_domains
        self._roles = candidate_roles
        self._rng = next_rng
        self._blocks = next_blocks
```

- [ ] **Step 3: Replace `src/enchilada/block.py`**

```python
"""The boundary between a model's local state and its published estimates."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data


@runtime_checkable
class Block(Protocol):
    """A model that publishes physical estimates while owning its sampler state.

    Keep sampler continuation and diagnostics in a block-owned State object.
    BlockState contains only template, covariance, and physical parameters.
    A block declares what it publishes in its `role` attribute, "signal" or
    "noise", usually one class-level line (`role = "noise"`); `Wheel.add`
    reads it once at registration. `add` calls `initial_sample` once with the
    block's starting inputs and adopts the returned state; `sample` then
    advances the latest accepted `current_block_state` and returns a complete
    new state. Both receive the conditional residual on the block's registered
    grid, the covariance the block's role receives, and the campaign RNG. A
    block that starts at a known point takes that point in its constructor and
    returns it from `initial_sample`; a block that starts from its prior draws
    from `rng`.

    `conditional_residual` is the observed data minus every OTHER block's current
    signal; this block's signal is still present. The block does not perform
    cross-block subtraction or add-back. `covariance` is a `Covariance` on the
    same grid as `conditional_residual`, or None: for a signal block the sum
    over every noise block, for a noise block the sum over the other noise
    blocks (its own last publication is on `current_block_state.covariance`).
    Wheel keeps the campaign on a WDM home grid and brings these inputs to the
    block's grid: the data by exact transforms, the covariance by
    `transform(covariance, ...)`, which treats every covariance as diagonal in
    its grid index.

    Signal and noise blocks share this interface; `role` says which one a
    block is. A signal block publishes a template (None means zero
    contribution, replacing any previous signal) and never a covariance. A
    noise block publishes its complete covariance on every call, on the grid
    of the data it received, and never a template; Wheel adds the noise
    blocks' covariances, so each block publishes only its own component. A
    joint model is two blocks.

    Each call, `initial_sample` included, receives independent snapshots of the
    conditional residual, covariance, and current state. Use the supplied RNG;
    Wheel commits its progress with the state. Treat the shared orbit as
    immutable.
    Wheel does not copy or roll back block-local state; blocks manage their own
    failure recovery. Use separate block instances for independent campaigns.
    Register a preferred data_domain with Wheel.add to receive matching inputs.
    Validate unsupported conventions explicitly.

    Static type checkers validate signatures; runtime protocol checks only test
    member presence. Neither establishes statistical correctness.
    """

    if TYPE_CHECKING:

        @property
        def name(self) -> str:
            """Stable registration name; immutable configurations are supported."""
            ...

        @property
        def role(self) -> str:
            """``"signal"`` or ``"noise"``: what this block publishes."""
            ...
    else:
        # Avoid an inherited property setter blocking dataclass construction
        # when an implementation explicitly subclasses this protocol.
        name: str
        role: str

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw or construct this block's starting physical state on the input grid."""
        ...

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Advance the supplied state and return the complete new block state."""
        ...
```

- [ ] **Step 4: Replace `src/enchilada/fixed_noise.py`**

```python
"""A noise block that publishes one fixed covariance on every call."""

from copy import deepcopy

import numpy as np

from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data


class FixedNoiseBlock:
    """Publish a known covariance as a noise block; nothing is sampled.

    ``wheel.add(FixedNoiseBlock("floor", covariance))`` adds ``covariance`` to
    the campaign's noise sum at registration and republishes it every cycle,
    so signal blocks receive it and other noise blocks add their own models on
    top. The covariance must be on the grid the block is registered on
    (``data_domain`` at ``add``), like any noise publication; Wheel translates
    it onto the home grid for the sum. ``initial_sample`` and ``sample``
    ignore their inputs and draw nothing. The constructor keeps an
    independent, revalidated copy.
    """

    role = "noise"

    def __init__(self, name: str, covariance: Covariance) -> None:
        if not isinstance(covariance, Covariance):
            raise TypeError(
                "FixedNoiseBlock covariance must be a Covariance, got "
                f"{type(covariance).__name__}"
            )
        self.name = name
        # Unbound call: copy without trusting a subclass's overridden methods.
        self._covariance = Covariance.copy(covariance)

    def _publish(self) -> BlockState:
        # deepcopy restores a validated Covariance read-only without revalidating.
        return BlockState(covariance=deepcopy(self._covariance))

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Publish the configured covariance; the inputs and ``rng`` are unused."""
        return self._publish()

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Publish the configured covariance again; inputs and ``rng`` are unused."""
        return self._publish()
```

- [ ] **Step 5: Replace `src/enchilada/testing.py`**

```python
"""Helpers for checking the Wheel/Block exchange contract end to end."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from enchilada.block import Block
from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.wheel import Wheel


class EchoBlock:
    """Zero-signal diagnostic block that prints residual RMS and advances a counter.

    A signal block: ``wheel.add(block)`` calls ``initial_sample``, which
    publishes an empty state, and every ``sample`` call does the same after
    printing. Inspect ``block.state.num_sample_calls`` for the counter;
    diagnostics belong to ``block.state.metadata``. Use a separate instance per
    campaign.
    """

    role = "signal"

    @dataclass
    class State:
        """Mutable continuation state and diagnostics owned by this block."""

        num_sample_calls: int = 0
        metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str):
        self.name = name
        self.state = self.State()

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Publish an empty physical state; the counter and the RNG are untouched."""
        return BlockState()

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Print the residual RMS, advance the local counter, and publish no signal."""
        first_channel_name = conditional_residual.channel_names[0]
        residual_rms = float(
            np.sqrt(
                np.mean(
                    np.abs(conditional_residual.channel_data[first_channel_name]) ** 2
                )
            )
        )
        num_sample_calls = self.state.num_sample_calls
        print(
            f"[{self.name}] sample {num_sample_calls}: "
            f"residual RMS on {first_channel_name!r} = {residual_rms:.4e}"
        )
        self.state.num_sample_calls += 1
        return BlockState()


def check_block(
    block_under_test: Block,
    observed_data: L1Data,
    num_cycles: int = 2,
    *,
    noise_blocks: Sequence[Block] = (),
    random_seed: int | None = 0,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
) -> None:
    """Run a block on a scratch Wheel and validate each returned BlockState.

    Registers each block in ``noise_blocks`` in order, then
    ``block_under_test``, which calls its ``initial_sample`` on the observation
    grid, then calls ``sample`` for ``num_cycles``, passing accepted state into
    each call. Wheel validates the start and every sample alike: the signal
    grid, finiteness, the declared role's rules and any published Covariance.

    Args:
        block_under_test: Configured block whose exchange contract is checked.
            A block that must start at a chosen point is constructed with it.
        observed_data: L1 observations and shared run settings for the scratch run.
        num_cycles: Number of sampling cycles after registration; defaults to 2.
        noise_blocks: Noise blocks registered before ``block_under_test``; each
            must declare ``role == "noise"``. A signal block that needs a
            covariance gets it from a ``FixedNoiseBlock`` here; a noise block
            under test receives their sum as its ``covariance``. Their
            covariances must be on ``observed_data``'s grid: to test with a
            frequency-grid covariance, pass ``transform(observed_data,
            "frequency")`` as ``observed_data``.
        random_seed: Seed for the scratch Wheel's NumPy generator, which also
            drives ``initial_sample``. Defaults to 0 for reproducibility; None
            requests nondeterministic initialization.
        num_frequency_divisions: Row count of the scratch Wheel's home grid;
            time or frequency ``observed_data`` need this or
            ``num_time_divisions``, WDM observations need neither.
        num_time_divisions: Column count of the scratch Wheel's home grid.

    This advances the supplied blocks' local state. It checks the exchange
    contract, not statistical correctness or block-local failure recovery.
    Test posterior recovery and independent campaigns separately;
    ``examples/toy_fit.py`` provides a pattern. ``check_block`` does not select
    a separate block domain: ``initial_sample`` and ``sample`` receive
    ``observed_data``'s grid.
    """
    for noise_block in noise_blocks:
        declared = getattr(noise_block, "role", None)
        if declared != "noise":
            raise ValueError(
                f"noise_blocks entry {getattr(noise_block, 'name', noise_block)!r} "
                f"declares role {declared!r}; every block in noise_blocks must "
                "declare role 'noise'"
            )
    wheel = Wheel(
        observed_data=observed_data,
        num_frequency_divisions=num_frequency_divisions,
        num_time_divisions=num_time_divisions,
        random_seed=random_seed,
    )
    for noise_block in noise_blocks:
        wheel.add(block_to_register=noise_block)
    wheel.add(block_to_register=block_under_test)
    wheel.run(num_cycles=num_cycles)
```

All four files are formatter-shaped; `ruff format` is a no-op on them.

- [ ] **Step 6: Run the new tests**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wheel_home_grid.py tests/test_fixed_noise.py`
Expected: `test_wheel_home_grid.py` all 30 passed; `test_fixed_noise.py` passes except that the role assertion is not there yet (added by the patch below).

- [ ] **Step 7: Bring the five core test files to the new API**

Save the patch below to `$TMPDIR/core-tests.patch` (a heredoc with a quoted delimiter, `cat > "$TMPDIR/core-tests.patch" <<'PATCH' ... PATCH`, keeps every character) and apply it from the repository root with `git apply "$TMPDIR/core-tests.patch"`. It was generated against the files as they are on `wdm-home` at 5ef8d85 and applies cleanly (`git apply --check` first). If a hunk does not apply because the file moved, make the same change by hand: every hunk is a plain substitution. What the patch does, so a reviewer can check it rather than read it line by line:

- `tests/test_noise_sum.py`: `home_level(observed, variance) = variance / observed.fs` (a white time variance on the home grid); `role = "signal"` on `ConstSignal`, `role = "noise"` on `WhiteNoise`, `offender(declared, ...)` sets `role = declared`; `snapshot`/`assert_unchanged` lose `_covariance_grid`; the two role-keyword tests become `test_the_role_is_read_from_the_block_after_the_protocol_checks`; `_check_refusal(..., declared=...)`; messages reworded to "declares role ..."; every `Wheel(observed...)` gets `num_frequency_divisions=8`; every `role=` keyword goes; handed-over values and `wheel.covariance.A` are compared with `assert_allclose` (the latter against `home_level(...)`), ledger entries stay exact; `test_the_sum_describes_the_observations` checks the home-grid sum against the WDM observations and its time translation against the time observations; `TestCovarianceGrid` is deleted (its surviving concern, a noise block publishing off its grid, lives in `tests/test_wheel_home_grid.py`); `TestRemovedApi` gains `test_the_role_keyword_and_the_one_grid_rule_are_gone`.
- `tests/test_initial_sample.py`: `role` attributes on `StartingBlock` (signal), `NoiseStart`, `NoiseOrNothing` (per parameter), `DropsNoise`, `RememberedNoise` (noise); messages reworded; `test_covariance_on_another_grid_is_refused_at_add` becomes `test_a_signal_block_on_another_grid_receives_the_translated_covariance`; divisions at every `Wheel(...)`/`check_block(...)`; `role=` keywords gone; views and received values via `assert_allclose`; `wheel.covariance.A` compared with `level / observed.fs` and, where the block published on frequency, `transform(wheel.covariance, "frequency").A` with the published 100.0.
- `tests/test_testing.py`: `FlatNoiseBlock.role = "noise"`; `test_role_is_required` becomes `test_a_signal_block_among_the_noise_blocks_is_refused` (and keeps a probe that `role=` is a `TypeError`); `test_role_violations_are_caught` sets the wrong role on an instance; divisions everywhere; `test_must_be_registered_on_its_covariance_grid` ends with the home-grid sum translated back to the frequency floor; `transform` imported.
- `tests/test_typing.py`: `CONSUMER` drops every `role=`, passes `num_frequency_divisions=2` to `Wheel`/`check_block`, asserts the `Covariance` overload's return type, gives `SampleOnly`/`FrozenBlock`/`ExplicitBlock` a `role`, and adds `RoleLess` (no role: not a `Block`); the role-keyword assertion becomes the `RoleLess` assignment error (still 5 errors, 3 `attr-defined`); the runtime tests gain a `RoleLess` isinstance check and `role = "signal"` on the explicit subclass.
- `tests/test_fixed_noise.py`: asserts `block.role == "noise"` and `FixedNoiseBlock.role == "noise"`.

```diff
--- a/tests/test_noise_sum.py
+++ b/tests/test_noise_sum.py
@@ -1,4 +1,4 @@
-"""Roles at Wheel.add: signal blocks sum templates, noise blocks sum covariances."""
+"""Declared roles: signal blocks sum templates, noise blocks sum covariances."""

 import re
 from dataclasses import dataclass
@@ -13,9 +13,16 @@
 from enchilada.testing import EchoBlock


+def home_level(observed, variance):
+    """White time variance `variance` as it reads on the WDM home grid: dt * sigma^2."""
+    return variance / observed.fs
+
+
 class ConstSignal:
     """A signal block publishing a constant template; `state.updates` counts samples."""

+    role = "signal"
+
     @dataclass
     class State:
         updates: int = 0
@@ -47,6 +54,8 @@
     first; `sample` republishes the last accepted state unchanged.
     """

+    role = "noise"
+
     def __init__(self, name, level, seen):
         self.name = name
         self.level = level
@@ -82,8 +91,8 @@
         )


-def offender(role, method, *, with_template, with_covariance):
-    """A block of `role` whose `method` returns the given fields.
+def offender(declared, method, *, with_template, with_covariance):
+    """A block declaring role `declared` whose `method` returns the given fields.

     The other method returns what the role allows, so the block registers and
     samples normally until `method` is called.
@@ -91,10 +100,11 @@

     class Offender:
         name = "offender"
+        role = declared

         @staticmethod
         def _allowed(conditional_residual):
-            if role == "noise":
+            if declared == "noise":
                 return BlockState(
                     covariance=white_covariance(conditional_residual, 1.0)
                 )
@@ -133,7 +143,6 @@
         "rng": wheel._rng.bit_generator.state,
         "signal_sum": wheel._signal_sum,
         "noise_sum": wheel._noise_sum,
-        "covariance_grid": wheel._covariance_grid,
         "ledger": {
             name: (
                 None if entry.template is None else entry.template["A"].copy(),
@@ -158,7 +167,6 @@
     assert after["rng"] == before["rng"]
     assert after["signal_sum"] is before["signal_sum"]
     assert after["noise_sum"] is before["noise_sum"]
-    assert after["covariance_grid"] == before["covariance_grid"]
     np.testing.assert_array_equal(after["residual"], before["residual"])
     _same_array(after["covariance"], before["covariance"])
     for name, (template, covariance, parameters) in before["ledger"].items():
@@ -169,32 +177,27 @@


 class TestRoles:
-    def test_role_is_validated_before_the_block_is_inspected(self, observed):
-        class Untouchable:
-            @property
-            def name(self):
-                raise AssertionError(
-                    "add must refuse the role before reading the block"
-                )
+    def test_the_role_is_read_from_the_block_after_the_protocol_checks(self, observed):
+        class RoleLess(ConstSignal):
+            role = None

-        wheel = Wheel(observed)
-        for role in ("Signal", "noise ", "", None, 1):
-            with pytest.raises(ValueError, match="role must be 'signal' or 'noise'"):
-                wheel.add(Untouchable(), role=role)
-        with pytest.raises(TypeError, match="role"):
-            wheel.add(ConstSignal("a", 1.0))
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        with pytest.raises(TypeError, match="block 'a' declares no role"):
+            wheel.add(RoleLess("a", 1.0))
+        typo = ConstSignal("b", 1.0)
+        typo.role = "Noise"
+        with pytest.raises(
+            ValueError, match=re.escape("block 'b' declares role 'Noise'; a role is")
+        ):
+            wheel.add(typo)
         assert len(wheel.ledger) == 0
         assert wheel._blocks == []
         assert wheel._roles == {}

-    def test_the_role_message_shows_the_offending_value(self, observed):
-        with pytest.raises(ValueError, match=re.escape("got 'Noise'")):
-            Wheel(observed).add(EchoBlock("e"), role="Noise")
-
     def test_roles_are_recorded_per_block(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(WhiteNoise("noise", 1.0, {}), role="noise")
-        wheel.add(ConstSignal("signal", 1.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(WhiteNoise("noise", 1.0, {}))
+        wheel.add(ConstSignal("signal", 1.0))
         assert wheel._roles == {"noise": "noise", "signal": "signal"}

     @pytest.mark.parametrize("method", ["initial_sample", "sample"])
@@ -202,12 +205,12 @@
         self._check_refusal(
             observed,
             method,
-            role="signal",
+            declared="signal",
             with_template=True,
             with_covariance=True,
             message=(
-                f"offender.{method} returned a covariance but 'offender' is "
-                "registered as a signal block"
+                f"offender.{method} returned a covariance but 'offender' "
+                "declares role 'signal'"
             ),
         )

@@ -216,12 +219,12 @@
         self._check_refusal(
             observed,
             method,
-            role="noise",
+            declared="noise",
             with_template=True,
             with_covariance=True,
             message=(
-                f"offender.{method} returned a template but 'offender' is "
-                "registered as a noise block"
+                f"offender.{method} returned a template but 'offender' "
+                "declares role 'noise'"
             ),
         )

@@ -230,32 +233,35 @@
         self._check_refusal(
             observed,
             method,
-            role="noise",
+            declared="noise",
             with_template=False,
             with_covariance=False,
             message=(
                 f"offender.{method} must return a covariance because 'offender' "
-                "is registered as a noise block"
+                "declares role 'noise'"
             ),
         )

     @staticmethod
     def _check_refusal(
-        observed, method, *, role, with_template, with_covariance, message
+        observed, method, *, declared, with_template, with_covariance, message
     ):
-        wheel = Wheel(observed, random_seed=3)
+        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
         block = offender(
-            role, method, with_template=with_template, with_covariance=with_covariance
+            declared,
+            method,
+            with_template=with_template,
+            with_covariance=with_covariance,
         )
         if method == "sample":
             # First in the cycle, so nothing else commits before the refusal.
-            wheel.add(block, role=role)
-        wheel.add(WhiteNoise("noise", 1.0, {}), role="noise")
-        wheel.add(ConstSignal("signal", 1.0), role="signal")
+            wheel.add(block)
+        wheel.add(WhiteNoise("noise", 1.0, {}))
+        wheel.add(ConstSignal("signal", 1.0))
         before = snapshot(wheel)
         with pytest.raises(ValueError, match=re.escape(message)):
             if method == "initial_sample":
-                wheel.add(block, role=role)
+                wheel.add(block)
             else:
                 wheel.run(1)
         assert_unchanged(wheel, before)
@@ -263,7 +269,7 @@
             assert "offender" not in wheel._roles
             assert "offender" not in wheel._block_indices
         else:
-            assert wheel._roles["offender"] == role
+            assert wheel._roles["offender"] == declared

     def test_a_noise_block_cannot_smuggle_a_non_finite_covariance_past_revalidation(
         self, observed
@@ -282,17 +288,17 @@
                     covariance=published, parameters={"level": self.level}
                 )

-        wheel = Wheel(observed, random_seed=3)
+        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
         # The only noise block, so no Covariance.__add__ in the sum revalidates it:
         # the refusal can only come from the copy Wheel takes at adoption.
-        wheel.add(Smuggler("smuggler", 1.0, {}), role="noise")
-        wheel.add(ConstSignal("signal", 0.5), role="signal")
+        wheel.add(Smuggler("smuggler", 1.0, {}))
+        wheel.add(ConstSignal("signal", 0.5))
         before = snapshot(wheel)
         with pytest.raises(ValueError, match=re.escape("blocks must be finite")):
             wheel.run(1)
         assert_unchanged(wheel, before)
         assert wheel._roles == {"smuggler": "noise", "signal": "signal"}
-        np.testing.assert_array_equal(wheel.covariance.A, 1.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 1.0))


 class TestNoiseSum:
@@ -300,58 +306,57 @@
         self, observed
     ):
         seen = {}
-        wheel = Wheel(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
         assert wheel.covariance is None
-        wheel.add(RecordingSignal("signal", 0.5, seen), role="signal")
+        wheel.add(RecordingSignal("signal", 0.5, seen))
         wheel.run(1)
         assert seen["signal"] == [None, None]
         assert wheel.covariance is None
-        assert wheel._covariance_grid is None

     def test_two_noise_blocks_each_receive_the_other_and_wheel_holds_the_sum(
         self, observed
     ):
         seen = {}
-        wheel = Wheel(observed)
-        wheel.add(WhiteNoise("instrument", 1.0, seen), role="noise")
-        wheel.add(WhiteNoise("confusion", 2.0, seen), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(WhiteNoise("instrument", 1.0, seen))
+        wheel.add(WhiteNoise("confusion", 2.0, seen))
         assert seen["instrument"] == [None]  # alone at registration
-        np.testing.assert_array_equal(seen["confusion"][0], 1.0)
-        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
+        np.testing.assert_allclose(seen["confusion"][0], 1.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
         wheel.run(1)
-        np.testing.assert_array_equal(seen["instrument"][1], 2.0)
-        np.testing.assert_array_equal(seen["confusion"][1], 1.0)
+        np.testing.assert_allclose(seen["instrument"][1], 2.0)
+        np.testing.assert_allclose(seen["confusion"][1], 1.0)
         np.testing.assert_array_equal(wheel.ledger["instrument"].covariance.A, 1.0)
         np.testing.assert_array_equal(wheel.ledger["confusion"].covariance.A, 2.0)
-        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
-        assert wheel._covariance_grid == observed.grid
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
+        assert wheel.covariance.grid == wheel.home_grid

     def test_a_signal_block_receives_the_sum_over_every_noise_block(self, observed):
         seen = {}
-        wheel = Wheel(observed)
-        wheel.add(RecordingSignal("early", 0.5, seen), role="signal")
-        wheel.add(WhiteNoise("instrument", 1.0, seen), role="noise")
-        wheel.add(WhiteNoise("confusion", 2.0, seen), role="noise")
-        wheel.add(RecordingSignal("late", 0.5, seen), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(RecordingSignal("early", 0.5, seen))
+        wheel.add(WhiteNoise("instrument", 1.0, seen))
+        wheel.add(WhiteNoise("confusion", 2.0, seen))
+        wheel.add(RecordingSignal("late", 0.5, seen))
         assert seen["early"] == [None]  # registered before any noise block
-        np.testing.assert_array_equal(seen["late"][0], 3.0)
+        np.testing.assert_allclose(seen["late"][0], 3.0)
         wheel.run(2)
         assert len(seen["early"]) == len(seen["late"]) == 3
         for name in ("early", "late"):
             for received in seen[name][1:]:
-                np.testing.assert_array_equal(received, 3.0)
+                np.testing.assert_allclose(received, 3.0)

     def test_three_noise_blocks_each_receive_the_sum_of_the_other_two(self, observed):
         seen = {}
-        wheel = Wheel(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
         for name, level in (("a", 1.0), ("b", 2.0), ("c", 4.0)):
-            wheel.add(WhiteNoise(name, level, seen), role="noise")
-        np.testing.assert_array_equal(seen["c"][0], 3.0)
+            wheel.add(WhiteNoise(name, level, seen))
+        np.testing.assert_allclose(seen["c"][0], 3.0)
         wheel.run(1)
-        np.testing.assert_array_equal(seen["a"][1], 6.0)
-        np.testing.assert_array_equal(seen["b"][1], 5.0)
-        np.testing.assert_array_equal(seen["c"][1], 3.0)
-        np.testing.assert_array_equal(wheel.covariance.A, 7.0)
+        np.testing.assert_allclose(seen["a"][1], 6.0)
+        np.testing.assert_allclose(seen["b"][1], 5.0)
+        np.testing.assert_allclose(seen["c"][1], 3.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 7.0))

     def test_a_republished_covariance_replaces_that_block_s_slot(self, observed):
         class Growing(WhiteNoise):
@@ -361,27 +366,28 @@
                 self.level += 1.0
                 return self.initial_sample(conditional_residual, covariance, rng=rng)

-        wheel = Wheel(observed)
-        wheel.add(Growing("growing", 1.0, {}), role="noise")
-        wheel.add(
-            FixedNoiseBlock("fixed", white_covariance(observed, 10.0)), role="noise"
-        )
-        np.testing.assert_array_equal(wheel.covariance.A, 11.0)
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Growing("growing", 1.0, {}))
+        wheel.add(FixedNoiseBlock("fixed", white_covariance(observed, 10.0)))
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 11.0))
         wheel.run(1)
-        np.testing.assert_array_equal(wheel.covariance.A, 12.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 12.0))
         wheel.run(1)
-        np.testing.assert_array_equal(wheel.covariance.A, 13.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 13.0))
         np.testing.assert_array_equal(wheel.ledger["growing"].covariance.A, 3.0)
         np.testing.assert_array_equal(wheel.ledger["fixed"].covariance.A, 10.0)

     def test_noise_blocks_contribute_no_signal(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(WhiteNoise("a", 1.0, {}), role="noise")
-        wheel.add(WhiteNoise("b", 2.0, {}), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(WhiteNoise("a", 1.0, {}))
+        wheel.add(WhiteNoise("b", 2.0, {}))
         wheel.run(2)
         for name in observed.channel_names:
-            np.testing.assert_array_equal(
-                wheel.residual().channel_data[name], observed.channel_data[name]
+            np.testing.assert_allclose(
+                wheel.residual().channel_data[name],
+                observed.channel_data[name],
+                rtol=1e-12,
+                atol=1e-12,
             )
             np.testing.assert_array_equal(wheel.contribution("a")[name], 0.0)
         assert wheel.ledger["a"].template is None
@@ -402,26 +408,29 @@
                     conditional_residual, covariance, current_block_state, rng=rng
                 )

-        wheel = Wheel(observed)
-        wheel.add(WhiteNoise("a", 1.0, {}), role="noise")
-        wheel.add(WhiteNoise("b", 2.0, {}), role="noise")
-        wheel.add(Mutates("signal", 0.5), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(WhiteNoise("a", 1.0, {}))
+        wheel.add(WhiteNoise("b", 2.0, {}))
+        wheel.add(Mutates("signal", 0.5))
         wheel.run(2)
         assert len(received) == 2
         assert received[0] is not received[1]
-        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
         total = wheel.covariance
         total.blocks.setflags(write=True)
         total.A[:] = 99.0
-        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 3.0))
         assert type(wheel.covariance) is Covariance
         assert not wheel.covariance.blocks.flags.writeable

     def test_the_sum_describes_the_observations(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(WhiteNoise("a", 1.0, {}), role="noise")
-        wheel.add(WhiteNoise("b", 2.0, {}), role="noise")
-        wheel.covariance.check_compatible(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(WhiteNoise("a", 1.0, {}))
+        wheel.add(WhiteNoise("b", 2.0, {}))
+        wheel.covariance.check_compatible(
+            transform(observed, "wdm", num_frequency_divisions=8)
+        )
+        transform(wheel.covariance, "time").check_compatible(observed)

     def test_a_noise_block_cannot_corrupt_the_others_through_its_hand_over(
         self, observed
@@ -440,16 +449,17 @@
                 )

         seen = {}
-        wheel = Wheel(observed)
-        wheel.add(WhiteNoise("a", 2.0, seen), role="noise")
-        wheel.add(Corrupting("b", 3.0, seen), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(WhiteNoise("a", 2.0, seen))
+        wheel.add(Corrupting("b", 3.0, seen))
         wheel.run(1)
-        np.testing.assert_array_equal(seen["b"][0], 2.0)  # a's model at registration
+        np.testing.assert_allclose(seen["b"][0], 2.0)  # a's model at registration
         np.testing.assert_array_equal(seen["b"][1], 999.0)  # its copy was overwritten
         np.testing.assert_array_equal(wheel.ledger["a"].covariance.A, 2.0)
-        np.testing.assert_array_equal(wheel.covariance.A, 5.0)
-        np.testing.assert_array_equal(
-            wheel._noise_sum.excluding(wheel._block_indices["b"]).A, 2.0
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 5.0))
+        np.testing.assert_allclose(
+            wheel._noise_sum.excluding(wheel._block_indices["b"]).A,
+            home_level(observed, 2.0),
         )

     def test_a_refused_noise_block_keeps_its_siblings_new_publication(self, observed):
@@ -472,133 +482,24 @@
                 return const_block_state(conditional_residual, 0.5)

         seen = {}
-        wheel = Wheel(observed)
-        wheel.add(Rising("a", 2.0, seen), role="noise")
-        wheel.add(TurnsSignal("b", 1.0, seen), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Rising("a", 2.0, seen))
+        wheel.add(TurnsSignal("b", 1.0, seen))
         with pytest.raises(
             ValueError,
             match=re.escape(
-                "b.sample returned a template but 'b' is registered as a noise block"
+                "b.sample returned a template but 'b' declares role 'noise'"
             ),
         ):
             wheel.run(1)  # registration order: a is adopted, then b is refused
         assert len(seen["b"]) == 2
-        np.testing.assert_array_equal(seen["b"][1], 3.0)  # a's new leaf, handed to b
+        np.testing.assert_allclose(seen["b"][1], 3.0)  # a's new leaf, handed to b
         np.testing.assert_array_equal(wheel.ledger["a"].covariance.A, 3.0)
         np.testing.assert_array_equal(wheel.ledger["b"].covariance.A, 1.0)
         assert wheel.ledger["b"].template is None
-        np.testing.assert_array_equal(wheel.covariance.A, 4.0)
+        np.testing.assert_allclose(wheel.covariance.A, home_level(observed, 4.0))


-class TestCovarianceGrid:
-    """The first noise publication fixes the grid of the sum."""
-
-    def test_a_noise_block_on_another_grid_is_refused_before_initial_sample(
-        self, observed
-    ):
-        class NeverStarted:
-            name = "spectral"
-
-            def initial_sample(self, conditional_residual, covariance, *, rng):
-                raise AssertionError("add must refuse the grid first")
-
-            def sample(
-                self, conditional_residual, covariance, current_block_state, *, rng
-            ):
-                raise AssertionError("unreachable")
-
-        wheel = Wheel(observed, random_seed=3)
-        wheel.add(WhiteNoise("time", 1.0, {}), role="noise")
-        before = snapshot(wheel)
-        with pytest.raises(ValueError, match="does not add across grids") as info:
-            wheel.add(NeverStarted(), role="noise", data_domain="frequency")
-        message = str(info.value)
-        assert "'spectral'" in message
-        assert "campaign covariance is on grid" in message
-        assert "data_domain='frequency'" in message
-        assert "data_domain='time'" in message
-        assert_unchanged(wheel, before)
-        assert "spectral" not in wheel._roles
-
-    def test_a_signal_block_on_another_grid_is_refused_at_add_and_at_run(
-        self, observed
-    ):
-        wheel = Wheel(observed, random_seed=3)
-        wheel.add(ConstSignal("early", 0.5), role="signal")
-        wheel.add(WhiteNoise("time", 1.0, {}), role="noise")
-        wheel.add(ConstSignal("same_grid", 0.5), role="signal", data_domain="time")
-        with pytest.raises(ValueError, match="does not translate between grids"):
-            wheel.add(ConstSignal("late", 0.5), role="signal", data_domain="frequency")
-        wheel.run(1)  # the blocks on the time grid still sample
-        wheel = Wheel(observed, random_seed=3)
-        wheel.add(ConstSignal("spectral", 0.5), role="signal", data_domain="frequency")
-        wheel.add(WhiteNoise("time", 1.0, {}), role="noise")
-        with pytest.raises(
-            ValueError,
-            match="'spectral' is registered on grid .* but the current covariance",
-        ):
-            wheel.run(1)
-
-    def test_the_first_noise_publication_fixes_the_grid(self, observed):
-        spectrum = observed.to_frequency()
-        wheel = Wheel(observed)
-        wheel.add(
-            FixedNoiseBlock("spectral", white_covariance(spectrum, 1.0)),
-            role="noise",
-            data_domain="frequency",
-        )
-        assert wheel._covariance_grid == spectrum.grid
-        assert wheel.covariance.grid == spectrum.grid
-        wheel.add(
-            FixedNoiseBlock("more", white_covariance(spectrum, 2.0)),
-            role="noise",
-            data_domain="frequency",
-        )
-        np.testing.assert_array_equal(
-            wheel.covariance.A, white_covariance(spectrum, 3.0).A
-        )
-        with pytest.raises(ValueError, match="does not add across grids"):
-            wheel.add(
-                FixedNoiseBlock("time", white_covariance(observed, 1.0)), role="noise"
-            )
-
-    def test_a_refused_first_noise_publication_leaves_the_grid_unset(self, observed):
-        class Publishes:
-            """A noise block on the time grid, publishing on frequency if off_grid."""
-
-            name = "noise"
-
-            def __init__(self, off_grid):
-                self.off_grid = off_grid
-
-            def initial_sample(self, conditional_residual, covariance, *, rng):
-                published_on = (
-                    transform(conditional_residual, "frequency")
-                    if self.off_grid
-                    else conditional_residual
-                )
-                return BlockState(covariance=white_covariance(published_on, 1.0))
-
-            def sample(
-                self, conditional_residual, covariance, current_block_state, *, rng
-            ):
-                return current_block_state
-
-        wheel = Wheel(observed, random_seed=3)
-        before = snapshot(wheel)
-        with pytest.raises(
-            ValueError, match=re.escape("noise.initial_sample covariance is on grid")
-        ):
-            wheel.add(Publishes(off_grid=True), role="noise")
-        assert_unchanged(wheel, before)
-        assert wheel._covariance_grid is None
-        assert wheel.covariance is None
-        assert wheel._roles == {}
-        wheel.add(Publishes(off_grid=False), role="noise")
-        assert wheel._covariance_grid == observed.grid
-        np.testing.assert_array_equal(wheel.covariance.A, 1.0)
-
-
 class TestRemovedApi:
     def test_noise_overwritten_warning_is_gone(self):
         assert not hasattr(enchilada, "NoiseOverwrittenWarning")
@@ -613,5 +514,14 @@
             Wheel(observed, initial_covariance=covariance)
         with pytest.raises(TypeError, match="positional"):
             Wheel(observed, covariance)
-        assert not hasattr(Wheel(observed), "_covariance")
-        assert not hasattr(Wheel(observed), "_noise_owner_block_name")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        assert not hasattr(wheel, "_covariance")
+        assert not hasattr(wheel, "_noise_owner_block_name")
+
+    def test_the_role_keyword_and_the_one_grid_rule_are_gone(self, observed):
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        with pytest.raises(TypeError, match="role"):
+            wheel.add(EchoBlock("e"), **{"role": "signal"})  # the keyword is gone
+        assert not hasattr(wheel, "_covariance_grid")
+        assert not hasattr(wheel, "_canonical_state")
+        assert not hasattr(wheel, "_observed_data")
--- a/tests/test_initial_sample.py
+++ b/tests/test_initial_sample.py
@@ -38,6 +38,7 @@
     """

     name = "starting"
+    role = "signal"

     @dataclass
     class State:
@@ -84,6 +85,8 @@
     """A StartingBlock whose start publishes only white noise of `level`."""

     class NoiseStart(StartingBlock):
+        role = "noise"
+
         def initial_sample(self, conditional_residual, covariance, *, rng):
             self.state.starts.append((conditional_residual, covariance, rng))
             return BlockState(covariance=white_covariance(conditional_residual, level))
@@ -100,10 +103,10 @@

     assert not isinstance(SampleOnly(), Block)
     assert isinstance(StartingBlock(), Block)
-    wheel = Wheel(observed, random_seed=1)
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=1)
     before = deepcopy(wheel._rng.bit_generator.state)
     with pytest.raises(TypeError, match="does not implement initial_sample"):
-        wheel.add(SampleOnly(), role="signal")
+        wheel.add(SampleOnly())
     assert len(wheel.ledger) == 0
     assert wheel._blocks == []
     assert wheel._rng.bit_generator.state == before
@@ -123,16 +126,16 @@
             )
             return super().initial_sample(conditional_residual, covariance, rng=rng)

-    wheel = Wheel(observed, random_seed=5)
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=5)
     block = ChecksAliasing()
-    wheel.add(block, role="signal")
+    wheel.add(block)
     assert len(block.state.starts) == 1
     residual, covariance, rng = block.state.starts[0]
-    np.testing.assert_array_equal(residual.channel_data["A"], 5.0)
+    np.testing.assert_allclose(residual.channel_data["A"], 5.0)
     assert covariance is None
     assert isinstance(rng, np.random.Generator)
-    np.testing.assert_array_equal(wheel.contribution("starting")["A"], 2.0)
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 3.0)
+    np.testing.assert_allclose(wheel.contribution("starting")["A"], 2.0)
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 3.0)
     # The block received its own copy of the residual.
     assert shared == [False]
     wheel.run(1)
@@ -141,13 +144,13 @@


 def test_initial_sample_receives_the_campaign_covariance(observed):
-    wheel = Wheel(observed)
-    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 0.25)), role="noise")
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 0.25)))
     block = StartingBlock()
-    wheel.add(block, role="signal")
+    wheel.add(block)
     _, covariance, _ = block.state.starts[0]
     assert isinstance(covariance, Covariance)
-    np.testing.assert_array_equal(covariance.A, 0.25)
+    np.testing.assert_allclose(covariance.A, 0.25)
     assert covariance is not wheel._noise_sum.total
     assert not covariance.blocks.flags.writeable

@@ -159,8 +162,8 @@
             return super().initial_sample(conditional_residual, covariance, rng=rng)

     def start(seed, block):
-        wheel = Wheel(observed, random_seed=seed)
-        wheel.add(block, role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=seed)
+        wheel.add(block)
         wheel.run(1)
         return wheel.ledger["starting"].parameters["amplitude"], block.state.draw

@@ -195,9 +198,9 @@
     )
     canonical = transform(observed, canonical_domain, **canonical_options)
     block_options = {"num_time_divisions": 4} if block_domain == "wdm" else {}
-    wheel = Wheel(canonical, random_seed=23)
+    wheel = Wheel(canonical, num_frequency_divisions=8, random_seed=23)
     block = StartingBlock()
-    wheel.add(block, role="signal", data_domain=block_domain, **block_options)
+    wheel.add(block, data_domain=block_domain, **block_options)
     residual, _, _ = block.state.starts[0]
     assert residual.data_domain == (block_domain or canonical_domain)
     if block_domain == "wdm":
@@ -230,15 +233,16 @@

     class RememberedNoise(StartingBlock):
         name = "noise"
+        role = "noise"

         def initial_sample(self, conditional_residual, covariance, *, rng):
             start = BlockState(covariance=white_covariance(conditional_residual, 0.25))
             returned["noise"] = start
             return start

-    wheel = Wheel(observed)
-    wheel.add(RememberedNoise(), role="noise")
-    wheel.add(RememberedStart(), role="signal")
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(RememberedNoise())
+    wheel.add(RememberedStart())
     start = returned["start"]
     start.template["A"][:] = 100
     start.parameters["amplitude"] = 100
@@ -249,9 +253,9 @@
     accepted = wheel.ledger["starting"]
     assert accepted.parameters["amplitude"] == 2.0
     np.testing.assert_array_equal(accepted.parameters["positions"][0], [1, 2])
-    np.testing.assert_array_equal(wheel.contribution("starting")["A"], 2)
+    np.testing.assert_allclose(wheel.contribution("starting")["A"], 2)
     np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 0.25)
-    np.testing.assert_array_equal(wheel.covariance.A, 0.25)
+    np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)


 @pytest.mark.parametrize("block_domain", [None, "frequency"])
@@ -292,22 +296,22 @@
         "type": (TypeError, "starting.initial_sample must return a BlockState"),
         "shape": (ValueError, "length 3, expected"),
         "nonfinite": (ValueError, "finite"),
-        "covariance": (ValueError, "registered as a signal block"),
+        "covariance": (ValueError, "declares role 'signal'"),
         "copy": (RuntimeError, "start state cannot be copied"),
     }[failure]
-    wheel = Wheel(observed, random_seed=7)
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=7)
     before = deepcopy(wheel._rng.bit_generator.state)
     with pytest.raises(expected, match=message):
-        wheel.add(FailingStart(), role="signal", data_domain=block_domain)
+        wheel.add(FailingStart(), data_domain=block_domain)
     assert len(wheel.ledger) == 0
     assert wheel._blocks == []
     assert wheel._block_domains == {}
     assert wheel._roles == {}
     assert wheel._rng.bit_generator.state == before
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 5)
     # The same name can be retried, and the discarded draw never reached the RNG.
     block = StartingBlock()
-    wheel.add(block, role="signal", data_domain=block_domain)
+    wheel.add(block, data_domain=block_domain)
     wheel.run(1)
     assert block.state.draw == np.random.default_rng(7).normal()

@@ -315,6 +319,8 @@
 @pytest.mark.parametrize("with_noise", [False, True])
 def test_start_without_signal_can_sample_in_selected_domain(observed, with_noise):
     class NoiseOrNothing(StartingBlock):
+        role = "noise" if with_noise else "signal"
+
         def initial_sample(self, conditional_residual, covariance, *, rng):
             assert conditional_residual.data_domain == "frequency"
             # White noise with variance 0.25 on the block's frequency grid:
@@ -344,21 +350,24 @@
             self.state.counter += 1
             return current_block_state

-    wheel = Wheel(observed)
+    wheel = Wheel(observed, num_frequency_divisions=8)
     block = NoiseOrNothing()
-    wheel.add(block, role="noise" if with_noise else "signal", data_domain="frequency")
+    wheel.add(block, data_domain="frequency")
     wheel.run(1)
     assert block.state.counter == 1
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 5)
     if with_noise:
-        np.testing.assert_allclose(wheel.covariance.A, 100.0)
-        assert wheel.covariance.grid == observed.to_frequency().grid
+        assert wheel.covariance.grid == wheel.home_grid
+        np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)
+        np.testing.assert_allclose(transform(wheel.covariance, "frequency").A, 100.0)
     else:
         assert wheel.covariance is None


 def test_a_noise_block_must_publish_a_covariance_on_every_call(observed):
     class DropsNoise(StartingBlock):
+        role = "noise"
+
         def initial_sample(self, conditional_residual, covariance, *, rng):
             return BlockState(
                 covariance=white_covariance(conditional_residual, 0.25),
@@ -368,79 +377,78 @@
         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
             return BlockState(parameters={"variance": 0.25})

-    wheel = Wheel(observed)
-    wheel.add(DropsNoise(), role="noise")
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 5)
-    np.testing.assert_array_equal(wheel.covariance.A, 0.25)
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(DropsNoise())
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 5)
+    np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)
     with pytest.raises(
         ValueError,
-        match="starting.sample must return a covariance because 'starting' is "
-        "registered as a noise block",
+        match="starting.sample must return a covariance because 'starting' "
+        "declares role 'noise'",
     ):
         wheel.run(1)
     np.testing.assert_array_equal(wheel.ledger["starting"].covariance.A, 0.25)
-    np.testing.assert_array_equal(wheel.covariance.A, 0.25)
+    np.testing.assert_allclose(wheel.covariance.A, 0.25 / observed.fs)


 def test_two_noise_starts_add_and_the_second_receives_the_first(observed):
-    wheel = Wheel(observed, random_seed=3)
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
     first, second = noise_start(1.0), noise_start(2.0)
     first.name, second.name = "first", "second"
-    wheel.add(first, role="noise")
-    wheel.add(second, role="noise")
+    wheel.add(first)
+    wheel.add(second)
     assert first.state.starts[0][1] is None  # alone when it started
-    np.testing.assert_array_equal(second.state.starts[0][1].A, 1.0)
+    np.testing.assert_allclose(second.state.starts[0][1].A, 1.0)
     assert list(wheel.ledger) == ["first", "second"]
     np.testing.assert_array_equal(wheel.ledger["first"].covariance.A, 1.0)
     np.testing.assert_array_equal(wheel.ledger["second"].covariance.A, 2.0)
-    np.testing.assert_array_equal(wheel.covariance.A, 3.0)
+    np.testing.assert_allclose(wheel.covariance.A, 3.0 / observed.fs)


 def test_check_block_can_exercise_a_configured_start(observed):
     block = StartingBlock(amplitude=4.0)
-    check_block(block, observed, num_cycles=2, role="signal")
+    check_block(block, observed, num_cycles=2, num_frequency_divisions=8)
     assert len(block.state.starts) == 1
     assert block.state.counter == 2
-    np.testing.assert_array_equal(block.state.starts[0][0].channel_data["A"], 5.0)
+    np.testing.assert_allclose(block.state.starts[0][0].channel_data["A"], 5.0)


 def test_later_blocks_start_from_the_residual_of_earlier_starts(observed):
-    wheel = Wheel(observed)
+    wheel = Wheel(observed, num_frequency_divisions=8)
     first, second = StartingBlock(amplitude=2.0), StartingBlock(amplitude=1.0)
     first.name, second.name = "first", "second"
-    wheel.add(first, role="signal")
-    wheel.add(second, role="signal", data_domain="frequency")
+    wheel.add(first)
+    wheel.add(second, data_domain="frequency")
     first_residual = first.state.starts[0][0]
     second_residual = second.state.starts[0][0]
-    np.testing.assert_array_equal(first_residual.channel_data["A"], 5.0)
+    np.testing.assert_allclose(first_residual.channel_data["A"], 5.0)
     assert second_residual.data_domain == "frequency"
     np.testing.assert_allclose(transform(second_residual, "time").channel_data["A"], 3)
     np.testing.assert_allclose(wheel.residual().channel_data["A"], 2.0)


-def test_covariance_on_another_grid_is_refused_at_add(observed):
-    wheel = Wheel(observed, random_seed=3)
-    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 0.25)), role="noise")
+def test_a_signal_block_on_another_grid_receives_the_translated_covariance(observed):
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
+    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 0.25)))
     block = StartingBlock()
-    before = deepcopy(wheel._rng.bit_generator.state)
-    with pytest.raises(ValueError, match="does not translate between grids") as info:
-        wheel.add(block, role="signal", data_domain="frequency")
-    message = str(info.value)
-    assert "'starting'" in message
-    assert "data_domain='frequency'" in message and "data_domain='time'" in message
-    assert "Register the block on the covariance's grid" in message
-    assert block.state.starts == []
-    assert list(wheel.ledger) == ["noise"]
-    assert wheel._rng.bit_generator.state == before
-    assert wheel.covariance.grid == observed.grid
+    wheel.add(block, data_domain="frequency")
+    residual, covariance, _ = block.state.starts[0]
+    spectrum = observed.to_frequency()
+    assert residual.grid == spectrum.grid
+    assert covariance.grid == spectrum.grid
+    np.testing.assert_allclose(covariance.A, white_covariance(spectrum, 0.25).A)
+    assert list(wheel.ledger) == ["noise", "starting"]
+    assert wheel.covariance.grid == wheel.home_grid
+    wheel.run(1)
+    assert block.state.counter == 1


 def test_start_that_publishes_a_covariance_is_handed_to_the_next_block(observed):
-    wheel = Wheel(observed)
+    wheel = Wheel(observed, num_frequency_divisions=8)
     noise, signal = noise_start(0.5), StartingBlock()
     noise.name = "noise"
-    wheel.add(noise, role="noise")
-    wheel.add(signal, role="signal")
+    wheel.add(noise)
+    wheel.add(signal)
     _, covariance, _ = signal.state.starts[0]
-    np.testing.assert_array_equal(covariance.A, 0.5)
+    np.testing.assert_allclose(covariance.A, 0.5)
     assert covariance.grid == observed.grid
--- a/tests/test_testing.py
+++ b/tests/test_testing.py
@@ -4,7 +4,7 @@
 import pytest

 from conftest import make_observed, white_covariance
-from enchilada import BlockState, FixedNoiseBlock, Wheel
+from enchilada import BlockState, FixedNoiseBlock, Wheel, transform
 from enchilada.testing import EchoBlock, check_block


@@ -32,6 +32,8 @@
 class FlatNoiseBlock:
     """A noise block publishing white noise of `level` from both methods."""

+    role = "noise"
+
     def __init__(self, name, level, seen=None):
         self.name = name
         self.level = level
@@ -57,17 +59,24 @@
             block_under_test=EchoBlock(name="echo"),
             observed_data=observed,
             num_cycles=2,
-            role="signal",
+            num_frequency_divisions=8,
             noise_blocks=(FixedNoiseBlock("noise", white_covariance(observed, 1.0)),),
             random_seed=0,
         )

     def test_conforming_signal_block_passes_without_noise(self, observed):
-        check_block(EchoBlock(name="echo"), observed, role="signal")
+        check_block(EchoBlock(name="echo"), observed, num_frequency_divisions=8)

-    def test_role_is_required(self, observed):
+    def test_a_signal_block_among_the_noise_blocks_is_refused(self, observed):
+        with pytest.raises(ValueError, match="noise_blocks entry 'echo' declares role"):
+            check_block(
+                EchoBlock(name="under_test"),
+                observed,
+                noise_blocks=(EchoBlock(name="echo"),),
+                num_frequency_divisions=8,
+            )
         with pytest.raises(TypeError, match="role"):
-            check_block(EchoBlock(name="echo"), observed)
+            check_block(EchoBlock(name="echo"), observed, **{"role": "signal"})

     def test_conforming_in_frequency_domain(self, rng):
         n = 64
@@ -78,7 +87,7 @@
                 ch: rng.standard_normal(n // 2 + 1) + 0j for ch in ("A", "E", "T")
             },
         )
-        check_block(EchoBlock(name="echo"), obs, role="signal")
+        check_block(EchoBlock(name="echo"), obs, num_frequency_divisions=8)

     def test_non_block_state_return_caught(self, observed):
         class Bad(EchoBlock):
@@ -93,7 +102,7 @@
                 return {"A": np.zeros(1)}  # not a BlockState

         with pytest.raises(TypeError, match="must return a BlockState"):
-            check_block(Bad(name="bad"), observed, role="signal")
+            check_block(Bad(name="bad"), observed, num_frequency_divisions=8)

     def test_off_grid_block_state_caught(self, observed):
         class Drifter(EchoBlock):
@@ -112,10 +121,14 @@
                 )

         with pytest.raises(ValueError, match="length 3, expected"):
-            check_block(Drifter(name="drift"), observed, role="signal")
+            check_block(Drifter(name="drift"), observed, num_frequency_divisions=8)

     def test_conforming_noise_block_passes(self, observed):
-        check_block(FlatNoiseBlock("noise", 1.0), observed.to_frequency(), role="noise")
+        check_block(
+            FlatNoiseBlock("noise", 1.0),
+            observed.to_frequency(),
+            num_frequency_divisions=8,
+        )

     def test_noise_model_violating_contract_caught(self, observed):
         class BadNoise(EchoBlock):
@@ -123,13 +136,17 @@
                 return conditional_residual.block_state().with_covariance(object())

         with pytest.raises(TypeError, match="Covariance"):
-            check_block(BadNoise(name="bad"), observed, role="noise")
+            check_block(BadNoise(name="bad"), observed, num_frequency_divisions=8)

     def test_role_violations_are_caught(self, observed):
-        with pytest.raises(ValueError, match="registered as a signal block"):
-            check_block(FlatNoiseBlock("noise", 1.0), observed, role="signal")
-        with pytest.raises(ValueError, match="registered as a noise block"):
-            check_block(EchoBlock(name="echo"), observed, role="noise")
+        turncoat = FlatNoiseBlock("noise", 1.0)
+        turncoat.role = "signal"
+        with pytest.raises(ValueError, match="declares role 'signal'"):
+            check_block(turncoat, observed, num_frequency_divisions=8)
+        echo = EchoBlock(name="echo")
+        echo.role = "noise"
+        with pytest.raises(ValueError, match="declares role 'noise'"):
+            check_block(echo, observed, num_frequency_divisions=8)

     def test_noise_blocks_register_first_and_the_signal_block_receives_their_sum(
         self, observed
@@ -139,7 +156,7 @@
             RecordingEcho("echo", seen),
             observed,
             num_cycles=2,
-            role="signal",
+            num_frequency_divisions=8,
             noise_blocks=(
                 FixedNoiseBlock("floor", white_covariance(observed, 1.0)),
                 FlatNoiseBlock("excess", 2.0),
@@ -147,7 +164,7 @@
         )
         assert len(seen) == 3  # initial_sample and two samples
         for received in seen:
-            np.testing.assert_array_equal(received, 3.0)
+            np.testing.assert_allclose(received, 3.0)

     def test_a_noise_block_under_test_receives_the_other_noise_blocks(self, observed):
         seen = []
@@ -155,12 +172,12 @@
             FlatNoiseBlock("excess", 2.0, seen),
             observed,
             num_cycles=2,
-            role="noise",
+            num_frequency_divisions=8,
             noise_blocks=(FixedNoiseBlock("floor", white_covariance(observed, 1.0)),),
         )
         assert len(seen) == 3
         for received in seen:
-            np.testing.assert_array_equal(received, 1.0)  # the floor, not itself
+            np.testing.assert_allclose(received, 1.0)  # the floor, not itself

     def test_block_without_initial_sample_is_refused(self, observed):
         class SampleOnly:
@@ -177,7 +194,7 @@
                 return current_block_state

         with pytest.raises(TypeError, match="does not implement initial_sample"):
-            check_block(SampleOnly(), observed, role="signal")
+            check_block(SampleOnly(), observed, num_frequency_divisions=8)

     def test_failing_initial_sample_is_caught(self, observed):
         class BadStart(EchoBlock):
@@ -185,7 +202,7 @@
                 return {"A": np.zeros(1)}  # not a BlockState

         with pytest.raises(TypeError, match="initial_sample must return a BlockState"):
-            check_block(BadStart(name="bad"), observed, role="signal")
+            check_block(BadStart(name="bad"), observed, num_frequency_divisions=8)


 class TestFixedNoiseBlock:
@@ -193,39 +210,40 @@
         check_block(
             FixedNoiseBlock("floor", white_covariance(observed, 1.0)),
             observed,
-            role="noise",
+            num_frequency_divisions=8,
         )

     def test_gives_a_campaign_a_known_covariance(self, observed):
         covariance = white_covariance(observed, 7.0)
-        wheel = Wheel(observed)
-        wheel.add(FixedNoiseBlock("floor", covariance), role="noise")
-        np.testing.assert_array_equal(wheel.covariance.A, 7.0)
-        wheel.add(EchoBlock("echo"), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(FixedNoiseBlock("floor", covariance))
+        np.testing.assert_allclose(wheel.covariance.A, 7.0 / observed.fs)
+        wheel.add(EchoBlock("echo"))
         wheel.run(2)
-        np.testing.assert_array_equal(wheel.covariance.A, 7.0)
+        np.testing.assert_allclose(wheel.covariance.A, 7.0 / observed.fs)
         np.testing.assert_array_equal(wheel.ledger["floor"].covariance.A, 7.0)
         assert wheel.ledger["floor"].template is None

     def test_must_be_registered_on_its_covariance_grid(self, observed):
-        wheel = Wheel(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
         spectral = white_covariance(observed.to_frequency(), 1.0)
         with pytest.raises(
             ValueError, match="floor.initial_sample covariance is on grid"
         ):
-            wheel.add(FixedNoiseBlock("floor", spectral), role="noise")
+            wheel.add(FixedNoiseBlock("floor", spectral))
         assert len(wheel.ledger) == 0
-        wheel.add(
-            FixedNoiseBlock("floor", spectral), role="noise", data_domain="frequency"
+        wheel.add(FixedNoiseBlock("floor", spectral), data_domain="frequency")
+        assert wheel.covariance.grid == wheel.home_grid
+        np.testing.assert_allclose(
+            transform(wheel.covariance, "frequency").A, spectral.A, rtol=1e-12
         )
-        assert wheel.covariance.grid == observed.to_frequency().grid


 class TestEchoBlock:
     def test_keeps_sampling_counter_in_local_state(self, observed):
         echo = EchoBlock(name="echo")
-        wheel = Wheel(observed)
-        wheel.add(echo, role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(echo)
         wheel.run(3)
         assert echo.state.num_sample_calls == 3
         published = wheel.ledger["echo"]
@@ -235,12 +253,15 @@
         assert echo.state.metadata == {}

     def test_contributes_a_zero_signal(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(EchoBlock(name="echo"), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(EchoBlock(name="echo"))
         wheel.run(1)
         for ch in observed.channel_names:
-            np.testing.assert_array_equal(
-                wheel.residual().channel_data[ch], observed.channel_data[ch]
+            np.testing.assert_allclose(
+                wheel.residual().channel_data[ch],
+                observed.channel_data[ch],
+                rtol=1e-12,
+                atol=1e-12,
             )


--- a/tests/test_typing.py
+++ b/tests/test_typing.py
@@ -44,20 +44,21 @@
 assert_type(transform(r, "wdm", num_frequency_divisions=2).wdm_grid, WDMGrid | None)
 noise = Covariance.like(r, {"A": np.ones(8)})
 assert_type(noise.copy(), Covariance)
+assert_type(transform(noise, "wdm", num_frequency_divisions=2), Covariance)
 assert_type(transform(BlockState(), "time"), BlockState)
-wheel = Wheel(r)
-wheel.add(FixedNoiseBlock("noise", noise), role="noise")
-wheel.add(EchoBlock("wdm"), role="signal", data_domain="wdm", num_time_divisions=2)
-wheel.add(EchoBlock("plain"), role="signal")
-wheel.add(EchoBlock("typo"), role="nosie")   # not a Role: static error
-check_block(EchoBlock("checked"), r, role="signal")
-check_block(EchoBlock("floored"), r, role="signal",
+wheel = Wheel(r, num_frequency_divisions=2)
+wheel.add(FixedNoiseBlock("noise", noise))
+wheel.add(EchoBlock("wdm"), data_domain="wdm", num_time_divisions=2)
+wheel.add(EchoBlock("plain"))
+check_block(EchoBlock("checked"), r, num_frequency_divisions=2)
+check_block(EchoBlock("floored"), r, num_frequency_divisions=2,
             noise_blocks=(FixedNoiseBlock("floor", noise),))
-check_block(FixedNoiseBlock("probe", noise), r, role="noise")
+check_block(FixedNoiseBlock("probe", noise), r, num_frequency_divisions=2)
 from enchilada import NoiseOverwrittenWarning   # removed: static error

 class SampleOnly:
     name = "sample_only"
+    role = "signal"

     def sample(
         self, conditional_residual: L1Data,
@@ -67,10 +68,29 @@
         return current_block_state

 sample_only: Block = SampleOnly()   # no initial_sample: not a Block
+
+class RoleLess:
+    name = "roleless"
+
+    def initial_sample(
+        self, conditional_residual: L1Data,
+        covariance: Covariance | None, *, rng: np.random.Generator,
+    ) -> BlockState:
+        return BlockState()
+
+    def sample(
+        self, conditional_residual: L1Data,
+        covariance: Covariance | None,
+        current_block_state: BlockState, *, rng: np.random.Generator,
+    ) -> BlockState:
+        return current_block_state
+
+roleless: Block = RoleLess()   # no role: not a Block

 @dataclass(frozen=True)
 class FrozenBlock:
     name: str = "frozen"
+    role = "signal"

     def initial_sample(
         self, conditional_residual: L1Data,
@@ -86,12 +106,13 @@
         return current_block_state

 frozen: Block = FrozenBlock()
-wheel.add(FrozenBlock(), role="signal")
-check_block(FrozenBlock(), r, role="signal")
+wheel.add(FrozenBlock())
+check_block(FrozenBlock(), r, num_frequency_divisions=2)

 @dataclass(frozen=True)
 class ExplicitBlock(Block):
     name: str
+    role = "signal"

     def initial_sample(
         self, conditional_residual: L1Data,
@@ -107,7 +128,7 @@
         return current_block_state

 explicit: Block = ExplicitBlock("explicit")
-wheel.add(explicit, role="signal")
+wheel.add(explicit)
 """


@@ -135,8 +156,11 @@
         'Incompatible types in assignment (expression has type "SampleOnly", '
         'variable has type "Block")'
     ) in out, out
-    # ... a misspelt role is a static error ...
-    assert 'Argument "role" to "add" of "Wheel" has incompatible type' in out, out
+    # ... and neither is a class without a role ...
+    assert (
+        'Incompatible types in assignment (expression has type "RoleLess", '
+        'variable has type "Block")'
+    ) in out, out
     # ... and the removed warning class is not importable
     assert 'has no attribute "NoiseOverwrittenWarning"' in out, out
     assert out.count("error:") == 5, out
@@ -203,7 +227,7 @@


 def test_block_protocol_is_runtime_checkable():
-    from enchilada import Block
+    from enchilada import Block, BlockState
     from enchilada.testing import EchoBlock

     assert isinstance(EchoBlock("e"), Block)
@@ -221,7 +245,18 @@

     assert not isinstance(SampleOnly(), Block)

+    class RoleLess:
+        name = "r"

+        def initial_sample(self, conditional_residual, covariance, *, rng):
+            return BlockState()
+
+        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
+            return current_block_state
+
+    assert not isinstance(RoleLess(), Block)
+
+
 def test_frozen_dataclass_can_explicitly_inherit_block():
     from dataclasses import dataclass

@@ -232,6 +267,7 @@
     @dataclass(frozen=True)
     class ExplicitBlock(Block):
         name: str
+        role = "signal"

         def initial_sample(self, conditional_residual, covariance, *, rng):
             return BlockState(parameters={"value": 1})
@@ -248,8 +284,8 @@
         tdi_generation="2.0",
         physical_observable="strain",
     )
-    wheel = Wheel(observations)
-    wheel.add(block, role="signal")
+    wheel = Wheel(observations, num_frequency_divisions=2)
+    wheel.add(block)
     wheel.run(1)
     assert wheel.ledger["explicit"].parameters == {"value": 1}

--- a/tests/test_fixed_noise.py
+++ b/tests/test_fixed_noise.py
@@ -14,6 +14,8 @@
     block = FixedNoiseBlock("floor", white_covariance(observed, 2.0))
     assert isinstance(block, Block)
     assert block.name == "floor"
+    assert block.role == "noise"
+    assert FixedNoiseBlock.role == "noise"
     start = block.initial_sample(observed, None, rng=rng)
     later = block.sample(observed, None, BlockState(), rng=rng)
     for published in (start, later):
```

- [ ] **Step 8: Run the Verify command, mypy and ruff**

Expected: 162 passed across the eight API test files, `test_covariance.py` and `test_sum_tree.py` green, `mypy` clean, `ruff check src tests` and `ruff format --check src tests` clean. Do not run the full suite as a gate here (it is red by design until Task 7); you may run it to list what the migrations will have to fix, but fix nothing outside this task's files.

- [ ] **Step 9: Commit**

```bash
git branch --show-current   # wdm-home
git add src/enchilada/wheel.py src/enchilada/block.py src/enchilada/fixed_noise.py src/enchilada/testing.py tests/test_wheel_home_grid.py tests/test_noise_sum.py tests/test_initial_sample.py tests/test_testing.py tests/test_typing.py tests/test_fixed_noise.py
git commit -m "Wheel: a WDM home grid, hand-over by covariance translation, block-declared roles" -m "Blocks are served on the grid they ask for; the noise sum is translated through the diagonal spectral model; role is a block attribute read at add. Core tests follow; the remaining suites, examples, notebooks and forge are migrated in the next commits." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

### Task 3: Wheel test migration

**Goal:** `tests/test_wheel.py` and `tests/test_wheel_translation.py` run against the home-grid Wheel of Task 2: roles are class attributes, every campaign on time or frequency data names its home grid, views are compared to rounding, covariances are read natively in the ledger and on the home grid (or translated back) in the sum, the three grid-rule tests are gone, and the overflow test provokes its overflow in a home-grid sum of finite leaves.

**Files:**
- Modify: `tests/test_wheel.py`
- Modify: `tests/test_wheel_translation.py`

**Acceptance Criteria:**
- [ ] `grep -n "role=" tests/test_wheel.py tests/test_wheel_translation.py` prints nothing, and every block class in both files carries one line `role = "signal"` or `role = "noise"` (`EchoNoise` in `test_wheel_translation.py` is the noise-role `EchoBlock`).
- [ ] Every `Wheel(...)` on time or frequency data passes `num_frequency_divisions=8`; `tests/test_wheel_translation.py` has no test matching "does not translate between grids", "does not add across grids" or `_covariance_grid`, and `test_noise_blocks_on_several_grids_add_on_the_home_grid` shows a frequency floor adding `1.0 / data.fs` to two WDM floors on the home grid.
- [ ] `test_cached_signal_sums_follow_updates_withdrawals_and_failures` provokes `ValueError` "non-finite" from a sum of finite home-grid leaves and leaves `wheel.contribution("4")` unchanged.
- [ ] Views (`residual()`, `working_residual`, `contribution(...)`, values handed to blocks) are compared with `assert_allclose` at `rtol=1e-12`; `wheel.observed_data` and ledger entries stay under `assert_array_equal`.
- [ ] The two files pass (81 tests), ruff check and format clean.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wheel.py tests/test_wheel_translation.py` → `81 passed`; `.venv/bin/python -m ruff check tests/test_wheel.py tests/test_wheel_translation.py` → `All checks passed!`; `.venv/bin/python -m ruff format --check tests/test_wheel.py tests/test_wheel_translation.py` → `2 files already formatted`.

**Steps:**

- [ ] **Step 1: Confirm the branch and the starting point**

```bash
git branch --show-current          # wdm-home, or stop
git status --short tests/test_wheel.py tests/test_wheel_translation.py   # prints nothing: Task 2 left these files as on main
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wheel.py tests/test_wheel_translation.py 2>&1 | tail -3
```

Expected before the migration: most tests fail with `TypeError: block '...' declares no role` or `ValueError: ... num_frequency_divisions ...`; that is the red state Task 2 left on purpose.

- [ ] **Step 2: Save and apply the patch**

Save the patch below to `$TMPDIR/task3-tests.patch` (a heredoc with a quoted delimiter, `cat > "$TMPDIR/task3-tests.patch" <<'PATCH' ... PATCH`, keeps every character) and apply it from the repository root:

```bash
git apply --check "$TMPDIR/task3-tests.patch" && git apply "$TMPDIR/task3-tests.patch"
```

It was generated against the files as they are on `wdm-home` at 5ef8d85 (identical to `main` at c669da8 for these two files) and applies cleanly. If a hunk does not apply because the file moved, make the same change by hand: every hunk is a plain substitution, and the list below says what each file's hunks do. What the patch does:

- `tests/test_wheel.py`: a module docstring naming the home grid; `transform` imported; `CLOSE = {"rtol": 1e-12, "atol": 1e-12}`; `role = "signal"` on `ConstBlock`, `InPlace`, `Minimal` and `PartlyZero`, `role = "noise"` on `LevelNoiseBlock`, `PublishesUnchecked` and `PublishesCompatible`; every `Wheel(observed, ...)` gains `num_frequency_divisions=8` (`Wheel(spectrum, num_frequency_divisions=8)` in `test_fixed_noise_block_supplies_a_known_covariance`); every `role=` keyword is gone; residuals, contributions, received residuals and levels are compared with `assert_allclose(..., **CLOSE)`; `test_covariance_refreshes_each_cycle` reads `wheel.covariance.A` as `1.0 / observed.fs` then `3.0 / observed.fs` (a time variance per WDM cell) and the ledger entry exactly; `test_fixed_noise_block_supplies_a_known_covariance` checks the ledger entry natively (`.grid == spectrum.grid`), `wheel.covariance.grid == wheel.home_grid` and `transform(wheel.covariance, "frequency").A` against the published covariance before and after the run; `test_cached_signal_sums_follow_updates_withdrawals_and_failures` is rewritten around a module helper `level_on(residual, value)` (a time constant expressed on the residual's grid), registers every `Changing` block on the home grid (`data_domain="wdm"`) and adds an `Enormous` leaf (`np.finfo(float).max` on every cell) so the candidate sum, not a transform, overflows; the float32 cancellation test compares to `0.0` with `atol=1e-6`; the subclass test checks `wheel.covariance.A.shape == wheel.home_grid.channel_shape` and `1.0 / observed.fs`.
- `tests/test_wheel_translation.py`: a module docstring describing the home grid; a module-level `class EchoNoise(EchoBlock)` with `role = "noise"` replaces `starting_at(EchoBlock("noise"), start)`; `role = "signal"` on `ConstantSignal`, `RandomSignal` and `NeverCalled`, `role = "noise"` on `PixelNoise`, `TimeNoiseFromFrequency` and `FrequencyNoise`; the three grid-rule tests (`test_block_on_another_grid_than_the_covariance_is_refused_at_add`, `test_wdm_block_on_a_time_covariance_is_refused_before_the_transform`, `test_default_block_is_refused_once_a_frequency_noise_block_has_published`) are deleted (Task 2's `tests/test_wheel_home_grid.py` carries the hand-over on several grids); `Wheel(canonical, num_frequency_divisions=8, random_seed=42)` and every other `Wheel(...)` on time or frequency data gets `num_frequency_divisions=8`; `role=` keywords are gone; "registered as a signal block" → "declares role 'signal'"; the wrong-grid test calls `monkeypatch.undo()` before reading the residual (the view is an inverse WDM transform) and its comments describe the (4, 16) block grid; `test_covariance_returned_on_another_grid...` checks the ledger entry natively, `wheel.covariance.grid == wheel.home_grid` and `transform(wheel.covariance, "frequency").A`; `test_covariance_published_on_the_block_grid_is_held_there` becomes `..._is_held_there_in_the_ledger` (entry exact, sum translated); `test_noise_only_start_on_the_block_grid_is_adopted_there` reads the entry natively and `transform(wheel.covariance, domain, **options).A`; `test_two_wdm_noise_blocks_add_on_the_wdm_grid` becomes `test_noise_blocks_on_several_grids_add_on_the_home_grid`: the frequency floor now adds, `total = 3.0 + 1.0 / data.fs`, the received sum is `total`, the ledger order is `["floor", "excess", "spectral", "wdm_signal"]`; the unused `const_block_state` import is dropped.

<!-- patch: task3-tests.patch -->
```diff
--- a/tests/test_wheel.py
+++ b/tests/test_wheel.py
@@ -1,15 +1,23 @@
-"""Wheel: the block-state ledger, the data-minus-others handoff, and its guards."""
+"""Wheel: the block-state ledger, the data-minus-others handoff, and its guards.

+Every campaign here lives on an 8-row WDM home grid while the observations
+arrive on the time grid, so the public views (residuals, contributions, the
+covariance translated back) come back through one exact inverse transform and
+are compared to rounding (`CLOSE`); ledger entries and `observed_data` stay exact.
+"""
+
 from dataclasses import dataclass, field, replace

 import numpy as np
 import pytest

 from conftest import const_block_state, make_observed, starting_at, white_covariance
-from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel
+from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel, transform
 from enchilada.testing import EchoBlock

+CLOSE = {"rtol": 1e-12, "atol": 1e-12}

+
 class ConstBlock:
     """Returns a block state with the same signal value in every sample.

@@ -18,6 +26,8 @@
     template, so its ledger entry carries that constant signal.
     """

+    role = "signal"
+
     @dataclass
     class State:
         updates: int = 0
@@ -47,6 +57,8 @@
 class LevelNoiseBlock:
     """A covariance level driven by a sampling-call counter."""

+    role = "noise"
+
     @dataclass
     class State:
         updates: int = 0
@@ -70,10 +82,10 @@
 class TestLedger:
     def test_full_residual_subtracts_every_signal_template(self, rng):
         obs = make_observed(rng)
-        wheel = Wheel(observed_data=obs)
-        wheel.add(block_to_register=ConstBlock("a", 1.0), role="signal")
-        wheel.add(ConstBlock("b", 10.0), role="signal")
-        wheel.add(ConstBlock("c", 100.0), role="signal")
+        wheel = Wheel(observed_data=obs, num_frequency_divisions=8)
+        wheel.add(block_to_register=ConstBlock("a", 1.0))
+        wheel.add(ConstBlock("b", 10.0))
+        wheel.add(ConstBlock("c", 100.0))
         wheel.run(num_cycles=3)
         full = wheel.residual()
         for ch in obs.channel_names:
@@ -83,10 +95,10 @@

     def test_exclude_leaves_that_block_in(self, rng):
         obs = make_observed(rng)
-        wheel = Wheel(obs)
-        wheel.add(ConstBlock("a", 1.0), role="signal")
-        wheel.add(ConstBlock("b", 10.0), role="signal")
-        wheel.add(ConstBlock("c", 100.0), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(ConstBlock("a", 1.0))
+        wheel.add(ConstBlock("b", 10.0))
+        wheel.add(ConstBlock("c", 100.0))
         wheel.run(2)
         # residual(exclude_block_name=b) = data minus a and c (110), b left in
         seen = wheel.residual(exclude_block_name="b")
@@ -97,8 +109,8 @@

     def test_contribution_is_the_returned_signal_template(self, rng):
         obs = make_observed(rng)
-        wheel = Wheel(obs)
-        wheel.add(ConstBlock("a", 7.0), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(ConstBlock("a", 7.0))
         wheel.run(2)
         for ch in obs.channel_names:
             np.testing.assert_allclose(wheel.contribution(block_name="a")[ch], 7.0)
@@ -110,6 +122,7 @@
         obs = make_observed(rng)

         class InPlace:
+            role = "signal"
             name = "ip"

             def initial_sample(self, conditional_residual, covariance, *, rng):
@@ -131,8 +144,8 @@
                     conditional_residual.channel_data
                 )  # ...and claim them

-        wheel = Wheel(obs)
-        wheel.add(InPlace(), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(InPlace())
         wheel.run(2)
         for ch in obs.channel_names:
             np.testing.assert_allclose(wheel.contribution("ip")[ch], 3.0)
@@ -141,8 +154,8 @@
             )

     def test_unknown_exclude_or_contribution_rejected(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(ConstBlock("a", 1.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(ConstBlock("a", 1.0))
         with pytest.raises(ValueError, match="unknown block"):
             wheel.residual(exclude_block_name="ghost")
         with pytest.raises(ValueError, match="unknown block"):
@@ -172,8 +185,8 @@
                     rng=rng,
                 )

-        wheel = Wheel(obs)
-        wheel.add(Mutator("mut", 0.0), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(Mutator("mut", 0.0))
         wheel.run(2)
         for ch, arr in snapshot.items():
             np.testing.assert_array_equal(
@@ -185,8 +198,8 @@
     ):
         block = ConstBlock("a", 1.0)
         state = block.state
-        wheel = Wheel(observed)
-        wheel.add(block, role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(block)
         assert block.state.updates == 0
         wheel.run(4)
         accepted = wheel.ledger["a"]
@@ -199,23 +212,23 @@
         assert block.value == 1.0

     def test_zero_blocks_is_a_noop(self, observed):
-        wheel = Wheel(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
         wheel.run(5)
         for ch in observed.channel_names:
-            np.testing.assert_array_equal(
-                wheel.residual().channel_data[ch], observed.channel_data[ch]
+            np.testing.assert_allclose(
+                wheel.residual().channel_data[ch], observed.channel_data[ch], **CLOSE
             )

     def test_run_rejects_bad_turn_counts(self, observed):
-        wheel = Wheel(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
         with pytest.raises(ValueError, match="num_cycles"):
             wheel.run(-1)
         with pytest.raises(ValueError, match="num_cycles"):
             wheel.run(True)  # bool is not a cycle count

     def test_run_accepts_numpy_integers(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(ConstBlock("a", 1.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(ConstBlock("a", 1.0))
         wheel.run(np.int64(2))


@@ -246,9 +259,9 @@
                     rng=rng,
                 )

-        wheel = Wheel(obs)
-        wheel.add(Recorder("a", 2.0), role="signal")
-        wheel.add(Recorder("b", 5.0), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(Recorder("a", 2.0))
+        wheel.add(Recorder("b", 5.0))
         wheel.run(1)
         # a sees data minus b (5); b sees data minus a (2)
         for ch in obs.channel_names:
@@ -277,9 +290,9 @@
                     ).template,
                 )

-        wheel = Wheel(observed)
-        wheel.add(Growing("a", 2.0), role="signal")
-        wheel.add(Growing("b", 5.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Growing("a", 2.0))
+        wheel.add(Growing("b", 5.0))
         wheel.run(2)
         assert [name for name, _ in seen] == ["a", "b", "a", "b"]
         for (_, handed), subtracted in zip(seen, [5.0, 3.0, 6.0, 4.0], strict=True):
@@ -311,34 +324,32 @@
                     rng=rng,
                 )

-        wheel = Wheel(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
         block = Recorder("rec", 2.0)
-        wheel.add(block, role="signal")
+        wheel.add(block)
         assert calls == ["initial_sample"]
         assert block.state.updates == 0
-        np.testing.assert_array_equal(wheel.contribution("rec")["A"], 2.0)
+        np.testing.assert_allclose(wheel.contribution("rec")["A"], 2.0, **CLOSE)
         wheel.run(2)
         assert calls == ["initial_sample", "sample", "sample"]

     def test_duplicate_name_rejected(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(EchoBlock(name="x"), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(EchoBlock(name="x"))
         with pytest.raises(ValueError, match="already registered"):
-            wheel.add(EchoBlock(name="x"), role="signal")
+            wheel.add(EchoBlock(name="x"))

     def test_empty_name_rejected(self, observed):
         with pytest.raises(ValueError, match="non-empty"):
-            Wheel(observed).add(EchoBlock(name=""), role="signal")
+            Wheel(observed, num_frequency_divisions=8).add(EchoBlock(name=""))

     def test_invalid_start_leaves_wheel_untouched(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(ConstBlock("ok", 1.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(ConstBlock("ok", 1.0))
         with pytest.raises(
             TypeError, match="bad.initial_sample must return a BlockState"
         ):
-            wheel.add(
-                starting_at(ConstBlock("bad", 0.0), "not a result"), role="signal"
-            )
+            wheel.add(starting_at(ConstBlock("bad", 0.0), "not a result"))
         with pytest.raises(ValueError, match="unknown block"):
             wheel.contribution("bad")  # not registered
         wheel.contribution("ok")  # the good one still is
@@ -359,8 +370,8 @@
             ):
                 return conditional_residual if return_input_data else {"A": np.zeros(1)}

-        wheel = Wheel(observed)
-        wheel.add(Bad("bad", 0.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Bad("bad", 0.0))
         with pytest.raises(TypeError, match="bad.sample must return a BlockState"):
             wheel.run(1)

@@ -380,8 +391,8 @@
                     }
                 )

-        wheel = Wheel(observed)
-        wheel.add(Drifter("drift", 0.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Drifter("drift", 0.0))
         with pytest.raises(ValueError, match="length 2, expected"):
             wheel.run(1)

@@ -399,24 +410,25 @@
                     template={"A": np.zeros(conditional_residual.num_time_samples)}
                 )

-        wheel = Wheel(observed)
-        wheel.add(Partial("partial", 0.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Partial("partial", 0.0))
         with pytest.raises(ValueError, match="must match the run's channels"):
             wheel.run(1)


 class TestExplicitNoise:
     def test_covariance_refreshes_each_cycle(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(LevelNoiseBlock("noise"), role="noise")
-        np.testing.assert_array_equal(wheel.covariance.A, 1.0)
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(LevelNoiseBlock("noise"))
+        # The sum is on the WDM home grid: a time variance of 1 is 1 / fs per cell.
+        np.testing.assert_allclose(wheel.covariance.A, 1.0 / observed.fs, **CLOSE)
         wheel.run(2)
-        np.testing.assert_array_equal(wheel.covariance.A, 3.0)
+        np.testing.assert_allclose(wheel.covariance.A, 3.0 / observed.fs, **CLOSE)
         np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 3.0)

     def test_noise_block_contributes_zero_without_signal(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(LevelNoiseBlock("noise"), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(LevelNoiseBlock("noise"))
         wheel.run(1)
         assert wheel.ledger["noise"].template is None
         for ch in observed.channel_names:
@@ -435,11 +447,11 @@
                 seen["covariance"] = covariance
                 return super().initial_sample(conditional_residual, covariance, rng=rng)

-        wheel = Wheel(observed)
-        wheel.add(LevelNoiseBlock("noise"), role="noise")
-        wheel.add(Reader("rec", 1.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(LevelNoiseBlock("noise"))
+        wheel.add(Reader("rec", 1.0))
         assert isinstance(seen["covariance"], Covariance)
-        np.testing.assert_array_equal(seen["covariance"].A, 1.0)
+        np.testing.assert_allclose(seen["covariance"].A, 1.0, **CLOSE)

     def test_sample_sees_current_noise(self, observed):
         levels = []
@@ -461,11 +473,11 @@
                     rng=rng,
                 )

-        wheel = Wheel(observed)
-        wheel.add(LevelNoiseBlock("noise"), role="noise")  # samples first each cycle
-        wheel.add(Recorder("rec", 0.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(LevelNoiseBlock("noise"))  # samples first each cycle
+        wheel.add(Recorder("rec", 0.0))
         wheel.run(3)
-        assert levels == [2.0, 3.0, 4.0]
+        np.testing.assert_allclose(levels, [2.0, 3.0, 4.0], **CLOSE)

     def test_copied_covariance_arrays_stay_read_only(self, observed):
         seen = []
@@ -487,8 +499,8 @@
                     rng=rng,
                 )

-        wheel = Wheel(observed)
-        wheel.add(Inspector("noise"), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Inspector("noise"))
         wheel.run(2)
         assert len(seen) == 2
         assert not any(arr.flags.writeable for arr in seen)
@@ -498,14 +510,21 @@
     def test_fixed_noise_block_supplies_a_known_covariance(self, observed):
         spectrum = observed.to_frequency()
         covariance = white_covariance(spectrum, 7.0)
-        wheel = Wheel(spectrum)
+        wheel = Wheel(spectrum, num_frequency_divisions=8)
         assert wheel.covariance is None
-        wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
-        np.testing.assert_array_equal(wheel.covariance.A, covariance.A)
-        assert wheel.covariance.grid == spectrum.grid
-        wheel.add(EchoBlock("echo"), role="signal")
+        wheel.add(FixedNoiseBlock("noise", covariance))
+        # The publication is kept natively; the sum is its home-grid translation.
+        np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, covariance.A)
+        assert wheel.ledger["noise"].covariance.grid == spectrum.grid
+        assert wheel.covariance.grid == wheel.home_grid
+        np.testing.assert_allclose(
+            transform(wheel.covariance, "frequency").A, covariance.A, **CLOSE
+        )
+        wheel.add(EchoBlock("echo"))
         wheel.run(2)
-        np.testing.assert_array_equal(wheel.covariance.A, covariance.A)
+        np.testing.assert_allclose(
+            transform(wheel.covariance, "frequency").A, covariance.A, **CLOSE
+        )
         assert wheel.ledger["noise"].template is None

     def test_fixed_noise_block_republishes_each_cycle(self, observed):
@@ -521,21 +540,21 @@
                 )

         covariance = white_covariance(observed, 7.0)
-        wheel = Wheel(observed_data=observed)
-        wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
-        wheel.add(Reader("rec", 1.0), role="signal")
+        wheel = Wheel(observed_data=observed, num_frequency_divisions=8)
+        wheel.add(FixedNoiseBlock("noise", covariance))
+        wheel.add(Reader("rec", 1.0))
         wheel.run(2)
         assert len(seen) == 2
         for received in seen:
-            np.testing.assert_array_equal(received, 7.0)
+            np.testing.assert_allclose(received, 7.0, **CLOSE)
         np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 7.0)


 class TestOnCycle:
     def test_callback_called_per_turn_with_wheel(self, observed):
         calls = []
-        wheel = Wheel(observed)
-        wheel.add(ConstBlock("a", 1.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(ConstBlock("a", 1.0))
         wheel.run(
             num_cycles=3, on_cycle_complete=lambda i, w: calls.append((i, w is wheel))
         )
@@ -550,10 +569,10 @@
         exactly what the noise blocks published."""
         obs = make_observed(rng).to_frequency()
         covariance = white_covariance(obs, 3.0)
-        wheel = Wheel(obs)
-        wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(FixedNoiseBlock("noise", covariance))
         expected = wheel.covariance.A.copy()
-        wheel.add(ConstBlock("signal", 1.0), role="signal")
+        wheel.add(ConstBlock("signal", 1.0))
         wheel.run(2)
         np.testing.assert_array_equal(wheel.covariance.A, expected)

@@ -574,8 +593,8 @@
                     }
                 )

-        wheel = Wheel(observed)
-        wheel.add(Blowup("boom", 0.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Blowup("boom", 0.0))
         with pytest.raises(ValueError, match="non-finite sample"):
             wheel.run(1)

@@ -584,10 +603,11 @@
             name = "nosample"

         with pytest.raises(TypeError, match="does not implement sample"):
-            Wheel(observed).add(NoSample(), role="signal")
+            Wheel(observed, num_frequency_divisions=8).add(NoSample())

     def test_block_with_name_initial_sample_and_sample_can_register(self, observed):
         class Minimal:
+            role = "signal"
             name = "minimal"

             def initial_sample(self, conditional_residual, covariance, *, rng):
@@ -603,8 +623,8 @@
             ):
                 return current_block_state

-        wheel = Wheel(observed)
-        wheel.add(Minimal(), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Minimal())
         wheel.run(1)
         assert list(wheel.ledger) == ["minimal"]

@@ -624,8 +644,8 @@
             ):
                 return conditional_residual.zero_block_state(parameters={"value": 0.0})

-        wheel = Wheel(observed)
-        wheel.add(Death("rj", 1.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(Death("rj", 1.0))
         with _w.catch_warnings():
             _w.simplefilter("error")
             wheel.run(1)
@@ -635,8 +655,8 @@
         # EchoBlock contributes a zero template on every sampling call
         import warnings as _w

-        wheel = Wheel(observed)
-        wheel.add(EchoBlock("echo"), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(EchoBlock("echo"))
         with _w.catch_warnings():
             _w.simplefilter("error")  # any warning becomes a failure
             wheel.run(3)
@@ -675,8 +695,8 @@
                     }
                 )

-        wheel = Wheel(obs)
-        wheel.add(Wider("w", 0.0), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(Wider("w", 0.0))
         wheel.run(1)
         assert wheel.residual().channel_data["A"].dtype == np.float64
         assert wheel.observed_data.channel_data["A"].dtype == np.float32  # left alone
@@ -686,55 +706,94 @@
     observed = make_observed(
         rng, channel_data={name: np.ones(64) for name in ("A", "E", "T")}
     )
-    wheel = Wheel(observed)
-    wheel.add(ConstBlock("positive", 1e16), role="signal")
-    wheel.add(ConstBlock("negative", -1e16), role="signal")
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], np.ones(64))
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(ConstBlock("positive", 1e16))
+    wheel.add(ConstBlock("negative", -1e16))
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 1.0, **CLOSE)
     wheel.run(2)
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], np.ones(64))
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 1.0, **CLOSE)


+def level_on(residual, value):
+    """A time-constant template of `value`, expressed on `residual`'s grid."""
+    zeros = replace(
+        residual,
+        channel_data={
+            name: np.zeros_like(x) for name, x in residual.channel_data.items()
+        },
+    )
+    time_reference = transform(zeros, "time")
+    options = (
+        {}
+        if residual.wdm_grid is None
+        else {"num_frequency_divisions": residual.wdm_grid.num_frequency_divisions}
+    )
+    return transform(
+        const_block_state(time_reference, value),
+        residual.data_domain,
+        reference_data=time_reference,
+        **options,
+    )
+
+
 def test_cached_signal_sums_follow_updates_withdrawals_and_failures(rng):
     observed = make_observed(
         rng, channel_data={name: np.full(64, 100.0) for name in ("A", "E", "T")}
     )

     class Changing(ConstBlock):
+        def _block_state(self, residual):
+            return level_on(residual, self.value)
+
         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
             self.state.updates += 1
             count = self.state.updates
             if count == 2 and self.name == "4":
                 # Signal leaves are finite individually, but their sum overflows.
-                return conditional_residual.block_state(
-                    {name: np.full(64, np.finfo(float).max) for name in ("A", "E", "T")}
-                )
+                return const_block_state(conditional_residual, np.finfo(float).max)
             if count == 2:
                 return BlockState()
-            return conditional_residual.block_state(
-                {name: np.full(64, self.value + count) for name in ("A", "E", "T")},
-            )
+            return level_on(conditional_residual, self.value + count)

-    wheel = Wheel(observed)
+    class Enormous:
+        """A home-grid leaf at the largest float: finite alone, infinite in a sum."""
+
+        role = "signal"
+        name = "overflow_guard"
+
+        def initial_sample(self, conditional_residual, covariance, *, rng):
+            return const_block_state(conditional_residual, np.finfo(float).max)
+
+        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
+            return current_block_state
+
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    # Every block works on the home grid itself (data_domain="wdm" without
+    # counts), so the enormous leaf below is summed but never translated; the
+    # levels are time constants expressed on that grid.
     for index in range(9):
-        wheel.add(Changing(str(index), float(index + 1)), role="signal")
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 55.0)
+        wheel.add(Changing(str(index), float(index + 1)), data_domain="wdm")
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 55.0, **CLOSE)
     wheel.run(1)
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 46.0)
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 46.0, **CLOSE)
     for index in range(9):
-        np.testing.assert_array_equal(
+        np.testing.assert_allclose(
             wheel.residual(exclude_block_name=str(index)).channel_data["A"],
             48.0 + index,
+            **CLOSE,
         )
     # Force an overflow in the candidate tree after its leaves pass validation.
-    wheel.add(ConstBlock("overflow_guard", np.finfo(float).max), role="signal")
+    wheel.add(Enormous(), data_domain="wdm")
     before_failed_block = wheel.contribution("4")["A"]
     with pytest.raises(ValueError, match="non-finite"):
         wheel.run(1)
     np.testing.assert_array_equal(wheel.contribution("4")["A"], before_failed_block)
     # The first four withdrawals were accepted; the rejected candidate never
     # enters the cached sum. Excluding the enormous guard recovers the small sum.
-    np.testing.assert_array_equal(
-        wheel.residual(exclude_block_name="overflow_guard").channel_data["A"], 60.0
+    np.testing.assert_allclose(
+        wheel.residual(exclude_block_name="overflow_guard").channel_data["A"],
+        60.0,
+        **CLOSE,
     )


@@ -752,10 +811,11 @@
                 }
             )

-    wheel = Wheel(observed)
+    wheel = Wheel(observed, num_frequency_divisions=8)
     for index, value in enumerate((1e8, 1.0, -1e8)):
-        wheel.add(NarrowSignal(str(index), value), role="signal")
-    np.testing.assert_array_equal(wheel.residual().channel_data["A"], 0.0)
+        wheel.add(NarrowSignal(str(index), value))
+    # The leaves are translated in float64; the 1e8 pair cancels to its rounding.
+    np.testing.assert_allclose(wheel.residual().channel_data["A"], 0.0, atol=1e-6)


 class TestDocumentedContracts:
@@ -765,8 +825,8 @@
         """A new block starts from the observations minus the blocks registered
         before it."""
         obs = make_observed(rng)
-        wheel = Wheel(obs)
-        wheel.add(ConstBlock("a", 1.0), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(ConstBlock("a", 1.0))
         seen = {}

         class Reader(ConstBlock):
@@ -774,7 +834,7 @@
                 seen["residual"] = conditional_residual
                 return super().initial_sample(conditional_residual, covariance, rng=rng)

-        wheel.add(Reader("b", 2.0), role="signal")
+        wheel.add(Reader("b", 2.0))
         np.testing.assert_allclose(
             seen["residual"].channel_data["A"], obs.channel_data["A"] - 1.0
         )
@@ -786,16 +846,16 @@
         """residual()'s docstring: 'callers may mutate freely'."""
         obs = make_observed(rng)
         snapshot = obs.channel_data["A"].copy()
-        wheel = Wheel(obs)  # no blocks: the aliasing case
+        wheel = Wheel(obs, num_frequency_divisions=8)  # no blocks: the aliasing case
         wheel.residual().channel_data["A"][0] = -999.0
         np.testing.assert_array_equal(obs.channel_data["A"], snapshot)
-        wheel.add(ConstBlock("a", 1.0), role="signal")
+        wheel.add(ConstBlock("a", 1.0))
         wheel.residual(exclude_block_name="a").channel_data["A"][0] = -999.0
         np.testing.assert_array_equal(obs.channel_data["A"], snapshot)

     def test_contribution_returns_a_copy_of_the_ledger_entry(self, observed):
-        wheel = Wheel(observed)
-        wheel.add(ConstBlock("a", 2.0), role="signal")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(ConstBlock("a", 2.0))
         got = wheel.contribution("a")
         got["A"][:] = 100.0
         np.testing.assert_allclose(wheel.contribution("a")["A"], 2.0)
@@ -821,8 +881,8 @@
                 tdi[channel][3] = bad
                 return conditional_residual.block_state(tdi)

-        wheel = Wheel(obs)
-        wheel.add(Blowup("b", 0.0), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(Blowup("b", 0.0))
         with pytest.raises(ValueError, match="non-finite sample"):
             wheel.run(1)

@@ -839,6 +899,7 @@
             )

         class PartlyZero:
+            role = "signal"
             name = "p"

             def initial_sample(self, conditional_residual, covariance, *, rng):
@@ -854,8 +915,8 @@
             ):
                 return partly_zero_result(conditional_residual)

-        wheel = Wheel(obs)
-        wheel.add(PartlyZero(), role="signal")
+        wheel = Wheel(obs, num_frequency_divisions=8)
+        wheel.add(PartlyZero())
         wheel.run(2)
         np.testing.assert_allclose(wheel.contribution("p")["T"], 0.0)
         np.testing.assert_allclose(wheel.contribution("p")["A"], 1.0)
@@ -867,7 +928,7 @@
         block = Numbered("x", 0.0)
         block.name = 5
         with pytest.raises(ValueError, match="non-empty string"):
-            Wheel(observed).add(block, role="signal")
+            Wheel(observed, num_frequency_divisions=8).add(block)

     def test_missing_name_gets_the_protocol_message(self, observed):
         class Anonymous:
@@ -885,7 +946,7 @@
                 return conditional_residual.zero_block_state()

         with pytest.raises((ValueError, TypeError), match="name"):
-            Wheel(observed).add(Anonymous(), role="signal")
+            Wheel(observed, num_frequency_divisions=8).add(Anonymous())


 class TestObservedDataIsChecked:
@@ -898,7 +959,7 @@
         with pytest.raises(
             ValueError, match=r"observed_data.channel_data\['E'\] has non-finite"
         ):
-            Wheel(obs)
+            Wheel(obs, num_frequency_divisions=8)

     def test_the_message_points_at_the_missing_gap_support(self, rng):
         """NaN is how a user marks a gap today, and gaps are not in the
@@ -906,10 +967,10 @@
         obs = make_observed(rng)
         obs.channel_data["A"][:3] = np.nan
         with pytest.raises(ValueError, match="no data-quality mask"):
-            Wheel(obs)
+            Wheel(obs, num_frequency_divisions=8)

     def test_finite_data_constructs_normally(self, observed):
-        assert Wheel(observed).residual() is not None
+        assert Wheel(observed, num_frequency_divisions=8).residual() is not None

     def test_covariance_subclass_cannot_skip_grid_validation(self, observed):
         """Wheel and FixedNoiseBlock copy through Covariance's own methods, so
@@ -930,13 +991,15 @@
             channel_names=covariance.channel_names,
             grid=covariance.grid,
         )
-        wheel = Wheel(observed)
+        wheel = Wheel(observed, num_frequency_divisions=8)
         with pytest.raises(ValueError, match="grid"):
-            wheel.add(FixedNoiseBlock("fixed", unchecked), role="noise")
+            wheel.add(FixedNoiseBlock("fixed", unchecked))
         assert len(wheel.ledger) == 0
         assert wheel.covariance is None

         class PublishesUnchecked(EchoBlock):
+            role = "noise"
+
             def initial_sample(self, conditional_residual, covariance, *, rng):
                 return BlockState(
                     covariance=white_covariance(conditional_residual, 1.0)
@@ -947,12 +1010,12 @@
             ):
                 return BlockState(covariance=unchecked)

-        wheel = Wheel(observed)
-        wheel.add(PublishesUnchecked("noise"), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(PublishesUnchecked("noise"))
         with pytest.raises(ValueError, match="grid"):
             wheel.run(1)
-        assert wheel.covariance.A.shape == observed.channel_data["A"].shape
-        np.testing.assert_array_equal(wheel.covariance.A, 1.0)
+        assert wheel.covariance.A.shape == wheel.home_grid.channel_shape
+        np.testing.assert_allclose(wheel.covariance.A, 1.0 / observed.fs, **CLOSE)

         # On the right grid, validation passes; only the copy keeps the stored
         # object a plain Covariance rather than the caller's subclass.
@@ -961,13 +1024,15 @@
             channel_names=observed.channel_names,
             grid=observed.grid,
         )
-        wheel = Wheel(observed)
-        wheel.add(FixedNoiseBlock("fixed", compatible), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(FixedNoiseBlock("fixed", compatible))
         assert type(wheel._noise_sum.total) is Covariance
         assert wheel._noise_sum.total is not compatible
         assert type(wheel.ledger["fixed"].covariance) is Covariance

         class PublishesCompatible(EchoBlock):
+            role = "noise"
+
             def initial_sample(self, conditional_residual, covariance, *, rng):
                 return BlockState(covariance=compatible)

@@ -976,8 +1041,8 @@
             ):
                 return BlockState(covariance=compatible)

-        wheel = Wheel(observed)
-        wheel.add(PublishesCompatible("noise"), role="noise")
+        wheel = Wheel(observed, num_frequency_divisions=8)
+        wheel.add(PublishesCompatible("noise"))
         assert type(wheel._noise_sum.total) is Covariance
         assert type(wheel.ledger["noise"].covariance) is Covariance
         wheel.run(1)
--- a/tests/test_wheel_translation.py
+++ b/tests/test_wheel_translation.py
@@ -1,9 +1,9 @@
-"""Wheel keeps accepted signals on its observation grid.
+"""Wheel serves each block on its own grid from a WDM home grid.

-The noise sum lives on the grid of the first noise publication. A block
-registered on that grid receives it; a noise block on another grid is refused
-at add, and a signal block on another grid is refused at add or at the next
-run once a noise block has published elsewhere.
+Templates are kept on the home grid and the views come back on the observation
+grid to rounding. A noise block's publication is kept natively in the ledger and
+translated onto the home grid for the sum, which every block receives translated
+onto its own grid; a noise block must still publish on the grid it registered on.
 """

 from dataclasses import dataclass, replace
@@ -11,7 +11,7 @@
 import numpy as np
 import pytest

-from conftest import const_block_state, starting_at, white_covariance
+from conftest import starting_at, white_covariance
 from enchilada import (
     BlockState,
     Covariance,
@@ -24,6 +24,12 @@
 from enchilada.testing import EchoBlock


+class EchoNoise(EchoBlock):
+    """An EchoBlock declaring the noise role; its start supplies the covariance."""
+
+    role = "noise"
+
+
 def observations():
     return L1Data(
         channel_data={name: np.full(64, 5.0) for name in ("A", "E")},
@@ -49,6 +55,8 @@


 class ConstantSignal:
+    role = "signal"
+
     @dataclass
     class State:
         updates: int = 0
@@ -102,7 +110,7 @@
     data = observations()
     options = {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
     canonical = transform(data, canonical_domain, **options)
-    wheel = Wheel(canonical, random_seed=42)
+    wheel = Wheel(canonical, num_frequency_divisions=8, random_seed=42)
     domains = ("time", "frequency", third_domain)
     blocks = {}
     for name, amplitude, domain in zip(
@@ -111,7 +119,7 @@
         options = {"num_time_divisions": 4} if domain == "wdm" else {}
         block = ConstantSignal(name, amplitude, domain)
         blocks[name] = block
-        wheel.add(block, role="signal", data_domain=domain, **options)
+        wheel.add(block, data_domain=domain, **options)
     wheel.run(3)
     assert wheel.working_residual.data_domain == canonical_domain
     restored = transform(wheel.residual(), "time")
@@ -123,6 +131,7 @@

 def test_wdm_noise_publication_is_held_on_the_wdm_grid():
     class PixelNoise:
+        role = "noise"
         name = "noise"

         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
@@ -141,20 +150,19 @@
             wdm_data, {name: variances for name in data.channel_names}
         )
     )
-    wheel = Wheel(data)
+    wheel = Wheel(data, num_frequency_divisions=8)
     wheel.add(
         starting_at(PixelNoise(), initial),
-        role="noise",
         data_domain="wdm",
         num_frequency_divisions=8,
     )
     assert wheel.covariance is not None
     assert wheel.ledger["noise"].template is None
     wheel.run(2)
-    np.testing.assert_array_equal(
-        wheel.residual().channel_data["A"], data.channel_data["A"]
+    np.testing.assert_allclose(
+        wheel.residual().channel_data["A"], data.channel_data["A"], rtol=1e-12
     )
-    assert wheel.covariance.grid == wdm_data.grid
+    assert wheel.covariance.grid == wdm_data.grid == wheel.home_grid
     np.testing.assert_array_equal(wheel.covariance.A, variances)


@@ -163,6 +171,7 @@
     fail = [True]

     class RandomSignal:
+        role = "signal"
         name = "random"

         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
@@ -177,20 +186,17 @@
             return BlockState(parameters={"amplitude": value})

     actual, expected = (
-        Wheel(observations(), random_seed=7),
-        Wheel(observations(), random_seed=7),
+        Wheel(observations(), num_frequency_divisions=8, random_seed=7),
+        Wheel(observations(), num_frequency_divisions=8, random_seed=7),
     )
     for wheel in (actual, expected):
         wheel.add(
             starting_at(RandomSignal(), BlockState(parameters={"amplitude": 0.0})),
-            role="signal",
             data_domain="frequency",
         )
     before = actual.ledger["random"].parameters.copy()
     match = (
-        "registered as a signal block"
-        if failure == "signal_publishes_covariance"
-        else None
+        "declares role 'signal'" if failure == "signal_publishes_covariance" else None
     )
     with pytest.raises(ValueError, match=match):
         actual.run(1)
@@ -200,13 +206,16 @@
     actual.run(1)
     expected.run(1)
     assert actual.ledger["random"].parameters == expected.ledger["random"].parameters
-    np.testing.assert_array_equal(
-        actual.residual().channel_data["A"], observations().channel_data["A"]
+    np.testing.assert_allclose(
+        actual.residual().channel_data["A"],
+        observations().channel_data["A"],
+        rtol=1e-12,
     )


 def test_invalid_domain_selection_does_not_register_or_sample():
     class NeverCalled:
+        role = "signal"
         name = "unregistered"

         def initial_sample(self, conditional_residual, covariance, *, rng):
@@ -215,22 +224,22 @@
         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
             raise AssertionError

-    wheel = Wheel(observations())
+    wheel = Wheel(observations(), num_frequency_divisions=8)
     with pytest.raises(ValueError, match="data_domain|target_domain"):
-        wheel.add(NeverCalled(), role="signal", data_domain="unknown")
+        wheel.add(NeverCalled(), data_domain="unknown")
     with pytest.raises(ValueError, match="data_domain|wdm"):
-        wheel.add(NeverCalled(), role="signal", num_time_divisions=8)
+        wheel.add(NeverCalled(), num_time_divisions=8)
     assert len(wheel.ledger) == 0


 def test_a_start_on_the_wrong_wdm_grid_is_refused(monkeypatch):
-    # add transforms the residual to WDM before initial_sample runs; the wrong
-    # template is then caught by the grid check inside transform, before any
-    # WDM conversion of the template itself, so domania never sees it. The patch
-    # is installed from inside initial_sample because add legitimately uses
-    # domania to build the residual first.
+    # add serves the residual on the block's (4, 16) grid before initial_sample
+    # runs; the wrong template is then caught by the grid check on that grid,
+    # before any translation of the template onto the home grid, so domania
+    # never sees it. The patch is installed from inside initial_sample because
+    # add legitimately uses domania to build the residual first.
     data = observations()
-    wheel = Wheel(data)
+    wheel = Wheel(data, num_frequency_divisions=8)
     # Both grids represent 64 samples, but the selected (Nf, Nt) = (4, 16)
     # requires shape (4, 16), not the (8, 8) coefficients of an (8, 8) grid.
     wrong = BlockState(template={name: np.zeros((8, 8)) for name in data.channel_names})
@@ -249,14 +258,14 @@
     with pytest.raises(ValueError, match="shape"):
         wheel.add(
             WrongGrid("wrong_grid", 1.0, "wdm"),
-            role="signal",
             data_domain="wdm",
             num_frequency_divisions=4,
         )
     assert len(wheel.ledger) == 0
+    monkeypatch.undo()  # the view below is an inverse WDM transform
     for name in data.channel_names:
-        np.testing.assert_array_equal(
-            wheel.residual().channel_data[name], data.channel_data[name]
+        np.testing.assert_allclose(
+            wheel.residual().channel_data[name], data.channel_data[name], rtol=1e-12
         )


@@ -272,157 +281,29 @@
     for name in data.channel_names:
         data.channel_data[name] = np.zeros((8, 8))
     with pytest.raises(ValueError, match="shape"):
-        Wheel(data)
+        Wheel(data, num_frequency_divisions=8)
     assert data.wdm_grid is original_grid


 @pytest.mark.parametrize("domain", ["time", "frequency"])
 def test_explicit_domain_revalidates_mutated_start_arrays(domain):
     data = observations()
-    wheel = Wheel(data)
+    wheel = Wheel(data, num_frequency_divisions=8)
     start = transform(data, domain).zero_block_state()
     start.template["A"] = start.template["A"].tolist()
     with pytest.raises(TypeError, match="numpy array"):
         wheel.add(
             starting_at(ConstantSignal("invalid_array", 1.0, domain), start),
-            role="signal",
             data_domain=domain,
         )
     assert len(wheel.ledger) == 0


-@pytest.mark.parametrize("role", ["signal", "noise"])
-def test_block_on_another_grid_than_the_covariance_is_refused_at_add(role):
-    """The Wheel neither translates nor adds a covariance across grids: a block
-    registered on another grid than the noise sum is refused before
-    initial_sample is called, with a message for its role."""
-
-    class NeverSampled:
-        name = "frequency_block"
-
-        def initial_sample(self, conditional_residual, covariance, *, rng):
-            raise AssertionError(
-                "the Wheel must refuse before initial_sample is called"
-            )
-
-        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
-            raise AssertionError(
-                "unreachable: sample exists only to pass add's callable check"
-            )
-
-    data = observations()
-    wheel = Wheel(data, random_seed=3)
-    wheel.add(FixedNoiseBlock("time_noise", white_covariance(data, 0.25)), role="noise")
-    rng_before = wheel._rng.bit_generator.state
-    if role == "signal":
-        match = "does not translate between grids"
-    else:
-        match = "does not add across grids"
-    with pytest.raises(ValueError, match=match) as info:
-        wheel.add(NeverSampled(), role=role, data_domain="frequency")
-    message = str(info.value)
-    assert "'frequency_block'" in message
-    assert "data_domain='frequency'" in message and "data_domain='time'" in message
-    if role == "signal":
-        assert "Register the block on the covariance's grid" in message
-    else:
-        assert "campaign covariance is on grid" in message
-    assert wheel._rng.bit_generator.state == rng_before
-    assert list(wheel.ledger) == ["time_noise"]
-    assert "frequency_block" not in wheel._roles
-    assert wheel.covariance.grid == data.grid
-    np.testing.assert_array_equal(wheel.covariance.A, 0.25)
-
-
-def test_wdm_block_on_a_time_covariance_is_refused_before_the_transform(monkeypatch):
-    # The covariance-grid check in _block_inputs runs at add, before the
-    # residual is transformed and before initial_sample.
-    data = observations()
-    wheel = Wheel(data)
-    wheel.add(FixedNoiseBlock("noise", white_covariance(data, 1.0)), role="noise")
-
-    def unexpected_transform(*args, **kwargs):
-        raise AssertionError("the grid check must come before any WDM transform")
-
-    monkeypatch.setattr("enchilada._transforms._wdm_transform", unexpected_transform)
-    with pytest.raises(ValueError, match="does not translate between grids"):
-        wheel.add(
-            EchoBlock("wdm_block"),
-            role="signal",
-            data_domain="wdm",
-            num_frequency_divisions=8,
-        )
-    with pytest.raises(ValueError, match="does not add across grids"):
-        wheel.add(
-            FixedNoiseBlock("wdm_noise", white_covariance(data, 1.0)),
-            role="noise",
-            data_domain="wdm",
-            num_frequency_divisions=8,
-        )
-    assert list(wheel.ledger) == ["noise"]
-    assert wheel.covariance.grid == data.grid
-
-
-def test_default_block_is_refused_once_a_frequency_noise_block_has_published():
-    """A signal block registered without data_domain is on the observation
-    grid; once a noise block registered on the frequency grid publishes there,
-    the signal block is refused at run."""
-
-    class TimeSignal:
-        name = "signal"
-
-        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
-            raise AssertionError("refused at run before sample is reached")
-
-    class FrequencyNoise:
-        name = "noise"
-
-        def initial_sample(self, conditional_residual, covariance, *, rng):
-            assert conditional_residual.data_domain == "frequency"
-            return BlockState(
-                covariance=white_covariance(conditional_residual, 1.0),
-                parameters={"level": 1.0},
-            )
-
-        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
-            raise AssertionError("the signal block fails first in the cycle")
-
-    data = observations()
-    wheel = Wheel(data, random_seed=3)
-    start = const_block_state(data, 0.5)
-    start.parameters["amplitude"] = 0.5
-    wheel.add(starting_at(TimeSignal(), start), role="signal")
-    wheel.add(FrequencyNoise(), role="noise", data_domain="frequency")
-    assert wheel.covariance.grid == transform(data, "frequency").grid
-    signal_before = wheel.ledger["signal"]
-    covariance_before = wheel.covariance
-    residual_before = {
-        name: values.copy()
-        for name, values in wheel.working_residual.channel_data.items()
-    }
-    rng_before = wheel._rng.bit_generator.state
-    with pytest.raises(
-        ValueError,
-        match="is registered on grid .* but the current covariance is on grid",
-    ) as info:
-        wheel.run(1)
-    assert "'signal'" in str(info.value)
-    assert wheel._rng.bit_generator.state == rng_before
-    entry = wheel.ledger["signal"]
-    assert entry.parameters == signal_before.parameters
-    for values in entry.template.values():
-        np.testing.assert_array_equal(values, 0.5)
-    for name, values in residual_before.items():
-        np.testing.assert_array_equal(wheel.working_residual.channel_data[name], values)
-    assert wheel.ledger["noise"].parameters == {"level": 1.0}
-    assert wheel.covariance.grid == covariance_before.grid
-    np.testing.assert_array_equal(wheel.covariance.blocks, covariance_before.blocks)
-
-
 def test_covariance_returned_on_another_grid_than_the_block_is_refused():
     """A noise block must publish on the grid of the data it received."""

     class TimeNoiseFromFrequency:
+        role = "noise"
         name = "frequency_noise"

         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
@@ -439,10 +320,9 @@
     start = BlockState(
         covariance=white_covariance(spectrum, 0.5), parameters={"level": 0.5}
     )
-    wheel = Wheel(data, random_seed=3)
+    wheel = Wheel(data, num_frequency_divisions=8, random_seed=3)
     wheel.add(
         starting_at(TimeNoiseFromFrequency(), start),
-        role="noise",
         data_domain="frequency",
     )
     rng_before = wheel._rng.bit_generator.state
@@ -454,15 +334,22 @@
     entry = wheel.ledger["frequency_noise"]
     assert entry.parameters == {"level": 0.5}
     assert entry.template is None
-    assert wheel.covariance.grid == spectrum.grid
-    np.testing.assert_array_equal(wheel.covariance.A, white_covariance(spectrum, 0.5).A)
-    np.testing.assert_array_equal(
-        wheel.residual().channel_data["A"], data.channel_data["A"]
+    assert entry.covariance.grid == spectrum.grid
+    np.testing.assert_array_equal(entry.covariance.A, white_covariance(spectrum, 0.5).A)
+    assert wheel.covariance.grid == wheel.home_grid
+    np.testing.assert_allclose(
+        transform(wheel.covariance, "frequency").A,
+        white_covariance(spectrum, 0.5).A,
+        rtol=1e-12,
     )
+    np.testing.assert_allclose(
+        wheel.residual().channel_data["A"], data.channel_data["A"], rtol=1e-12
+    )


-def test_covariance_published_on_the_block_grid_is_held_there():
+def test_covariance_published_on_the_block_grid_is_held_there_in_the_ledger():
     class FrequencyNoise:
+        role = "noise"
         name = "frequency_noise"

         def _publish(self, conditional_residual, level):
@@ -485,18 +372,24 @@
             return self._publish(conditional_residual, 1.0 + abs(rng.normal()))

     data = observations()
-    wheel = Wheel(data, random_seed=3)
-    wheel.add(FrequencyNoise(), role="noise", data_domain="frequency")
+    wheel = Wheel(data, num_frequency_divisions=8, random_seed=3)
+    wheel.add(FrequencyNoise(), data_domain="frequency")
     wheel.run(2)
-    assert wheel.covariance.grid == data.to_frequency().grid
-    level = wheel.ledger["frequency_noise"].parameters["level"]
-    np.testing.assert_array_equal(wheel.covariance.A, level)
+    entry = wheel.ledger["frequency_noise"]
+    level = entry.parameters["level"]
+    assert entry.covariance.grid == data.to_frequency().grid
+    np.testing.assert_array_equal(entry.covariance.A, level)
+    assert wheel.covariance.grid == wheel.home_grid
+    np.testing.assert_allclose(
+        transform(wheel.covariance, "frequency").A, level, rtol=1e-12
+    )
     assert wheel.working_residual.data_domain == "time"


 @pytest.mark.parametrize("domain", ["frequency", "wdm"])
 def test_noise_only_start_on_the_block_grid_is_adopted_there(domain):
-    """A start covariance is on the block's grid; the Wheel keeps it there."""
+    """A start covariance is on the block's grid; the ledger keeps it there and
+    the sum carries its translation."""
     data = observations()
     reference, options = block_reference(data, domain)
     variances = np.ones(reference.grid.channel_shape)
@@ -505,16 +398,18 @@
             reference, {name: variances for name in data.channel_names}
         )
     )
-    wheel = Wheel(data)
+    wheel = Wheel(data, num_frequency_divisions=8)
     wheel.add(
-        starting_at(EchoBlock("noise"), start),
-        role="noise",
+        starting_at(EchoNoise("noise"), start),
         data_domain=domain,
         **options,
     )
-    assert wheel.covariance.grid == reference.grid
-    np.testing.assert_array_equal(wheel.covariance.A, variances)
     assert wheel.ledger["noise"].covariance.grid == reference.grid
+    np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, variances)
+    assert wheel.covariance.grid == wheel.home_grid
+    np.testing.assert_allclose(
+        transform(wheel.covariance, domain, **options).A, variances, rtol=1e-12
+    )


 @pytest.mark.parametrize("domain", ["frequency", "wdm"])
@@ -524,11 +419,10 @@
     data = observations()
     _, options = block_reference(data, domain)
     start = BlockState(covariance=white_covariance(data, 1.0))  # on the time grid
-    wheel = Wheel(data)
+    wheel = Wheel(data, num_frequency_divisions=8)
     with pytest.raises(ValueError, match="noise.initial_sample covariance is on grid"):
         wheel.add(
-            starting_at(EchoBlock("noise"), start),
-            role="noise",
+            starting_at(EchoNoise("noise"), start),
             data_domain=domain,
             **options,
         )
@@ -539,11 +433,9 @@
 def test_wheel_refuses_a_covariance_with_reordered_channel_names():
     data = observations()
     reordered = replace(data, channel_names=("E", "A"))
-    wheel = Wheel(data)
+    wheel = Wheel(data, num_frequency_divisions=8)
     with pytest.raises(ValueError, match="channel_names"):
-        wheel.add(
-            FixedNoiseBlock("noise", white_covariance(reordered, 1.0)), role="noise"
-        )
+        wheel.add(FixedNoiseBlock("noise", white_covariance(reordered, 1.0)))
     assert len(wheel.ledger) == 0
     assert wheel.covariance is None

@@ -552,10 +444,10 @@
     data = observations()
     reordered = replace(data, channel_names=("E", "A"))
     start = BlockState(covariance=white_covariance(reordered, 1.0))
-    wheel = Wheel(data, random_seed=3)
+    wheel = Wheel(data, num_frequency_divisions=8, random_seed=3)
     rng_before = wheel._rng.bit_generator.state
     with pytest.raises(ValueError, match="channel_names"):
-        wheel.add(starting_at(EchoBlock("noise"), start), role="noise")
+        wheel.add(starting_at(EchoNoise("noise"), start))
     assert len(wheel.ledger) == 0
     assert wheel._blocks == []
     assert wheel._block_domains == {}
@@ -565,26 +457,23 @@
     assert wheel._rng.bit_generator.state == rng_before


-def test_two_wdm_noise_blocks_add_on_the_wdm_grid():
+def test_noise_blocks_on_several_grids_add_on_the_home_grid():
     data = observations()
     reference, options = block_reference(data, "wdm")
     ones = np.ones(reference.grid.channel_shape)
     floor = Covariance.like(reference, {name: ones for name in data.channel_names})
     excess = Covariance.like(reference, {name: 2 * ones for name in data.channel_names})
-    wheel = Wheel(data)
-    wheel.add(
-        FixedNoiseBlock("floor", floor), role="noise", data_domain="wdm", **options
-    )
-    wheel.add(
-        FixedNoiseBlock("excess", excess), role="noise", data_domain="wdm", **options
-    )
-    assert wheel.covariance.grid == reference.grid
+    wheel = Wheel(data, num_frequency_divisions=8)
+    wheel.add(FixedNoiseBlock("floor", floor), data_domain="wdm", **options)
+    wheel.add(FixedNoiseBlock("excess", excess), data_domain="wdm", **options)
+    assert wheel.covariance.grid == reference.grid == wheel.home_grid
     np.testing.assert_array_equal(wheel.covariance.A, 3.0)
+    # A third noise block on the frequency grid: white noise of variance 1 per
+    # time sample, which is 1 / fs per WDM cell once translated onto the home grid.
     spectral = white_covariance(data.to_frequency(), 1.0)
-    with pytest.raises(ValueError, match="does not add across grids"):
-        wheel.add(
-            FixedNoiseBlock("spectral", spectral), role="noise", data_domain="frequency"
-        )
+    wheel.add(FixedNoiseBlock("spectral", spectral), data_domain="frequency")
+    total = 3.0 + 1.0 / data.fs
+    np.testing.assert_allclose(wheel.covariance.A, total, rtol=1e-12)
     received: list[np.ndarray] = []

     class WdmSignal(EchoBlock):
@@ -595,9 +484,9 @@
                 conditional_residual, covariance, current_block_state, rng=rng
             )

-    wheel.add(WdmSignal("wdm_signal"), role="signal", data_domain="wdm", **options)
+    wheel.add(WdmSignal("wdm_signal"), data_domain="wdm", **options)
     wheel.run(1)
-    np.testing.assert_array_equal(wheel.covariance.A, 3.0)
+    np.testing.assert_allclose(wheel.covariance.A, total, rtol=1e-12)
     assert len(received) == 1
-    np.testing.assert_array_equal(received[0], 3.0)
-    assert list(wheel.ledger) == ["floor", "excess", "wdm_signal"]
+    np.testing.assert_allclose(received[0], total, rtol=1e-12)
+    assert list(wheel.ledger) == ["floor", "excess", "spectral", "wdm_signal"]
```

- [ ] **Step 3: Run the tests, ruff and the grep**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_wheel.py tests/test_wheel_translation.py
.venv/bin/python -m ruff check tests/test_wheel.py tests/test_wheel_translation.py
.venv/bin/python -m ruff format --check tests/test_wheel.py tests/test_wheel_translation.py
grep -n "role=\|does not translate between grids\|does not add across grids\|_covariance_grid" tests/test_wheel.py tests/test_wheel_translation.py
```

Expected: `81 passed`, `All checks passed!`, `2 files already formatted`, and the grep prints nothing. If a test fails after a hand-applied hunk, compare the failing region with the patch text: the patch is the validated end state.

- [ ] **Step 4: Commit**

```bash
git branch --show-current   # wdm-home
git add tests/test_wheel.py tests/test_wheel_translation.py
git commit -m "Tests: migrate the Wheel suites to the home grid and declared roles" -m "Roles are class attributes, campaigns name their home grid, views are compared to rounding, noise publications are read natively in the ledger and translated in the sum; the one-grid-rule tests are gone (tests/test_wheel_home_grid.py covers the hand-over)." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

### Task 4: Remaining test migration

**Goal:** The six remaining core test files (`tests/test_state.py`, `tests/test_block_state.py`, `tests/test_block_state_contract.py`, `tests/test_ledger.py`, `tests/test_translation_review.py`, `tests/test_data.py`) run against the home-grid Wheel: roles as attributes, home grids named at construction, views to rounding, covariances read on the home grid or translated back, the small-record parametrisations moved to even records that a WDM grid can carry, and the overflow-at-registration test on WDM observations.

**Files:**
- Modify: `tests/test_state.py`
- Modify: `tests/test_block_state.py`
- Modify: `tests/test_block_state_contract.py`
- Modify: `tests/test_ledger.py`
- Modify: `tests/test_translation_review.py`
- Modify: `tests/test_data.py`

**Acceptance Criteria:**
- [ ] `grep -n "role=" tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py` prints nothing; every block class carries `role = "signal"` or `role = "noise"`.
- [ ] Every `Wheel(...)` on time or frequency data names its home grid (`num_frequency_divisions=8` on the 64-sample fixtures; `=2` on the 4- and 8-sample records); `test_valid_rfft_signal_preserves_all_allowed_components` runs on records `[4, 8]` and `test_rejected_rfft_signal_preserves_estimates_state_and_rng` on `(4, 0, "DC")`, `(4, -1, "Nyquist")`, `(8, 0, "DC")`, with a comment saying why odd records are gone (a WDM home grid needs two even division counts).
- [ ] `test_residual_overflow_is_rejected_before_registration` builds its 1e308 observations on the WDM grid (home == arrival) and still sees `ValueError` "non-finite" with nothing registered.
- [ ] `test_home_grid_block_receives_the_covariance_without_transform` (renamed from `test_same_grid_block...`) runs on WDM observations and records `2.0 / observed.fs` twice without `transform` being called.
- [ ] `wheel.covariance` is compared on the home grid (`level / observed.fs`) or through `transform(wheel.covariance, <domain>)`; ledger covariances stay exact; views via `assert_allclose(rtol=1e-12[, atol=1e-12])`.
- [ ] The six files pass (160 tests), ruff check and format clean.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py` → `160 passed`; `.venv/bin/python -m ruff check` and `.venv/bin/python -m ruff format --check` on the same six paths → clean.

**Steps:**

- [ ] **Step 1: Confirm the branch and the starting point**

```bash
git branch --show-current          # wdm-home, or stop
git status --short tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py   # prints nothing
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py 2>&1 | tail -3
```

Expected before the migration: failures with `TypeError: block '...' declares no role` and `ValueError` about division counts; the red state Task 2 left on purpose.

- [ ] **Step 2: Save and apply the patch**

Save the patch below to `$TMPDIR/task4-tests.patch` (heredoc with a quoted delimiter) and apply it from the repository root:

```bash
git apply --check "$TMPDIR/task4-tests.patch" && git apply "$TMPDIR/task4-tests.patch"
```

Generated against `wdm-home` at 5ef8d85 (identical to `main` at c669da8 for these files); every hunk is a plain substitution. What it does:

- `tests/test_state.py`: `transform` imported; `Counter.role = "signal"`; `Wheel(observed, num_frequency_divisions=8, ...)` everywhere; `role=` keywords gone; `working_residual` compared with `assert_allclose(rtol=1e-12, atol=1e-12)`; "registered as a signal block" → "declares role 'signal'"; the comment on the `None`/`"time"` registrations says both register on the arrival (time) grid; the covariance test reads `wheel.covariance.A` as `1 / observed.fs` and the ledger entry exactly; `test_residual_overflow_is_rejected_before_registration` builds `data = transform(observed, "wdm", num_frequency_divisions=8)` filled with 1e308 and constructs `Wheel(data)` (WDM observations are the home grid, so nothing is transformed and the overflow happens in the subtraction, as the test means it to).
- `tests/test_block_state.py`: `transform` imported; `role = "signal"` on `Signal`, `SignalOnly`, `Population`, `Baseline`, `role = "noise"` on `NoiseOnly`; `test_valid_rfft_signal_preserves_all_allowed_components` parametrised `[4, 8]` with `Wheel(observed, num_frequency_divisions=2)`, the Nyquist bin set to `2.0` (real on an even record), contribution and residual via `assert_allclose`; `test_rejected_rfft_signal_preserves_estimates_state_and_rng` parametrised `[(4, 0, "DC"), (4, -1, "Nyquist"), (8, 0, "DC")]` with `num_frequency_divisions=2`; the noise-only and population tests compare `transform(wheel.covariance, <domain>).A` with `white_covariance(observed, level).A`; the rejected-noise test's messages become "must return a covariance because 'noise' declares role 'noise'" and "returned a template but 'noise' declares role 'noise'", its covariance checks go through `transform(actual.covariance, "time")`, contributions and residuals via `assert_allclose(rtol=1e-12)`.
- `tests/test_block_state_contract.py`: `StatefulSignal.role = "signal"`; the accepted ledger template is checked for `wheel.home_grid.channel_shape` and its values through `wheel.contribution("signal")[name]` (allclose); the final contribution via `assert_allclose`.
- `tests/test_ledger.py`: `AmplitudeUpdate.role = "signal"`; `num_frequency_divisions=8`; `role=` gone.
- `tests/test_translation_review.py`: `role = "signal"` on `RecordInputs`, `RecordCovariance`, `MutateInputs`; `wheel.covariance.A` compared with `2.0 / observed.fs`; `test_same_grid_block_receives_the_covariance_without_transform` becomes `test_home_grid_block_receives_the_covariance_without_transform` on `home = transform(observed, "wdm", num_frequency_divisions=8)` with `FixedNoiseBlock("noise", white_covariance(home, 2.0))` and the probe on `data_domain="wdm"`, recording `float(covariance.A.flat[0])` and expecting `[2.0 / observed.fs] * 2`; `Wheel(canonical, num_frequency_divisions=8, random_seed=7)` for both the actual and the control campaign; the residual comparison via `assert_allclose(rtol=1e-12)`.
- `tests/test_data.py`: `Wheel(reordered, num_frequency_divisions=2)` (a 4-sample record: 2 × 2).

<!-- patch: task4-tests.patch -->
```diff
--- a/tests/test_state.py
+++ b/tests/test_state.py
@@ -6,10 +6,11 @@
 import pytest

 from conftest import starting_at, white_covariance
-from enchilada import FixedNoiseBlock, Wheel
+from enchilada import FixedNoiseBlock, Wheel, transform


 class Counter:
+    role = "signal"
     name = "counter"

     @dataclass
@@ -29,9 +30,9 @@


 def test_ledger_carries_estimates_and_block_keeps_continuation(observed):
-    wheel = Wheel(observed, random_seed=4)
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=4)
     block = Counter()
-    wheel.add(block, role="signal")
+    wheel.add(block)
     assert block.state.n == 0
     initial = wheel.ledger["counter"].parameters["a"]
     wheel.run(2)
@@ -41,20 +42,22 @@

 def test_pristine_working_and_input_are_independent(observed):
     original = observed.channel_data["A"].copy()
-    wheel = Wheel(observed)
+    wheel = Wheel(observed, num_frequency_divisions=8)
     observed.channel_data["A"][:] = 100
     wheel.observed_data.channel_data["A"][:] = 200
     wheel.working_residual.channel_data["A"][:] = 300
     np.testing.assert_array_equal(wheel.observed_data.channel_data["A"], original)
-    np.testing.assert_array_equal(wheel.working_residual.channel_data["A"], original)
+    np.testing.assert_allclose(
+        wheel.working_residual.channel_data["A"], original, rtol=1e-12, atol=1e-12
+    )


 def test_block_state_parameters_are_copied_on_adoption_and_read(observed):
     initial = observed.zero_block_state(
         parameters={"a": 0.0, "amplitudes": [np.ones(2)]}
     )
-    wheel = Wheel(observed)
-    wheel.add(starting_at(Counter(), initial), role="signal")
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(starting_at(Counter(), initial))
     initial.parameters["amplitudes"][0][:] = 10
     wheel.ledger["counter"].parameters["amplitudes"][0][:] = 20
     np.testing.assert_array_equal(
@@ -81,17 +84,23 @@
                 )
             return result

-    actual, expected = Wheel(observed, random_seed=5), Wheel(observed, random_seed=5)
+    actual, expected = (
+        Wheel(observed, num_frequency_divisions=8, random_seed=5),
+        Wheel(observed, num_frequency_divisions=8, random_seed=5),
+    )
     block = FailsOnce()
-    actual.add(block, role="signal")
-    expected.add(Counter(), role="signal")
+    actual.add(block)
+    expected.add(Counter())
     before = actual.ledger["counter"].parameters.copy()
     with pytest.raises(ValueError, match="non-finite"):
         actual.run(1)
     assert actual.ledger["counter"].parameters == before
     assert block.state.n == 1
-    np.testing.assert_array_equal(
-        actual.working_residual.channel_data["A"], observed.channel_data["A"]
+    np.testing.assert_allclose(
+        actual.working_residual.channel_data["A"],
+        observed.channel_data["A"],
+        rtol=1e-12,
+        atol=1e-12,
     )
     actual.run(1)
     expected.run(1)
@@ -100,9 +109,12 @@

 def test_separate_blocks_keep_independent_campaign_continuation(observed):
     first, second = Counter(), Counter()
-    one, two = Wheel(observed, random_seed=7), Wheel(observed, random_seed=7)
-    one.add(first, role="signal")
-    two.add(second, role="signal")
+    one, two = (
+        Wheel(observed, num_frequency_divisions=8, random_seed=7),
+        Wheel(observed, num_frequency_divisions=8, random_seed=7),
+    )
+    one.add(first)
+    two.add(second)
     one.run(2)
     assert first.state.n == 2
     assert second.state.n == 0
@@ -122,9 +134,12 @@
                 conditional_residual, covariance, current_block_state, rng=rng
             )

-    actual, expected = Wheel(observed, random_seed=11), Wheel(observed, random_seed=11)
-    actual.add(RetainsRng(), role="signal")
-    expected.add(Counter(), role="signal")
+    actual, expected = (
+        Wheel(observed, num_frequency_divisions=8, random_seed=11),
+        Wheel(observed, num_frequency_divisions=8, random_seed=11),
+    )
+    actual.add(RetainsRng())
+    expected.add(Counter())
     actual.run(1)
     expected.run(1)
     retained[0].normal(size=100)
@@ -156,13 +171,16 @@
                 return result.with_covariance(white_covariance(conditional_residual, 1))
             return result

-    actual, expected = Wheel(observed, random_seed=12), Wheel(observed, random_seed=12)
+    actual, expected = (
+        Wheel(observed, num_frequency_divisions=8, random_seed=12),
+        Wheel(observed, num_frequency_divisions=8, random_seed=12),
+    )
     block = Invalid()
-    actual.add(block, role="signal")
-    expected.add(Counter(), role="signal")
+    actual.add(block)
+    expected.add(Counter())
     expected_error, message = {
         "copy": (RuntimeError, "cannot copy parameter"),
-        "covariance": (ValueError, "registered as a signal block"),
+        "covariance": (ValueError, "declares role 'signal'"),
         "exception": (RuntimeError, "sampler failed"),
     }[failure]
     with pytest.raises(expected_error, match=message):
@@ -175,12 +193,12 @@
     assert actual.ledger["counter"].parameters == expected.ledger["counter"].parameters


-# "time" on time observations takes the Wheel's same-grid short-circuit.
+# None and "time" both register the block on the arrival (time) grid.
 @pytest.mark.parametrize("data_domain", [None, "time"])
 def test_covariance_input_and_output_snapshots_are_isolated(observed, data_domain):
     covariance = white_covariance(observed, 1)
-    wheel = Wheel(observed)
-    wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(FixedNoiseBlock("noise", covariance))
     covariance.blocks.setflags(write=True)
     covariance.A[:] *= 2
     inspected = wheel.covariance
@@ -193,17 +211,20 @@
             covariance.A[:] *= 4
             return current_block_state

-    wheel.add(EditsNoise(), role="signal", data_domain=data_domain)
+    wheel.add(EditsNoise(), data_domain=data_domain)
     wheel.run(1)
-    np.testing.assert_array_equal(wheel.covariance.A, 1)
+    np.testing.assert_allclose(wheel.covariance.A, 1 / observed.fs, rtol=1e-12)
     np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, 1)


 def test_residual_overflow_is_rejected_before_registration(observed):
+    # WDM observations are the home grid itself, so nothing is transformed and
+    # the overflow happens in the subtraction, as this test means it to.
+    data = transform(observed, "wdm", num_frequency_divisions=8)
     data = replace(
-        observed,
+        data,
         channel_data={
-            ch: np.full_like(x, 1e308) for ch, x in observed.channel_data.items()
+            ch: np.full_like(x, 1e308) for ch, x in data.channel_data.items()
         },
     )

@@ -214,7 +235,6 @@
                 Counter(),
                 data.block_state({ch: -x for ch, x in data.channel_data.items()}),
             ),
-            role="signal",
         )
     assert len(wheel.ledger) == 0
     np.testing.assert_array_equal(
--- a/tests/test_block_state.py
+++ b/tests/test_block_state.py
@@ -6,7 +6,7 @@
 import pytest

 from conftest import const_block_state, make_observed, starting_at, white_covariance
-from enchilada import BlockState, Wheel
+from enchilada import BlockState, Wheel, transform


 def test_wdm_result_factory_accepts_a_full_grid_and_zero_factory_matches_grid(
@@ -76,7 +76,9 @@
         observed.block_state(template=signal)


-@pytest.mark.parametrize("num_time_samples", [1, 4, 5])
+# A WDM home grid needs two even division counts, so an odd record (no Nyquist
+# bin) cannot carry a Wheel; the frequency-grid factories above cover that case.
+@pytest.mark.parametrize("num_time_samples", [4, 8])
 def test_valid_rfft_signal_preserves_all_allowed_components(observed, num_time_samples):
     observed = replace(
         observed,
@@ -85,27 +87,28 @@
     ).to_frequency()
     signal = {ch: samples.copy() for ch, samples in observed.channel_data.items()}
     signal["A"][0] = 1.0
-    if num_time_samples > 1:
-        signal["A"][-1] = 2.0 if num_time_samples % 2 == 0 else 2.0 + 3j
+    signal["A"][-1] = 2.0  # the Nyquist bin of an even record is real

     class Signal:
+        role = "signal"
         name = "signal"

         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
             return current_block_state

-    wheel = Wheel(observed)
-    wheel.add(
-        starting_at(Signal(), observed.block_state(template=signal)), role="signal"
-    )
+    wheel = Wheel(observed, num_frequency_divisions=2)
+    wheel.add(starting_at(Signal(), observed.block_state(template=signal)))
     wheel.run(1)
-    np.testing.assert_array_equal(wheel.contribution("signal")["A"], signal["A"])
-    np.testing.assert_array_equal(
-        wheel.working_residual.channel_data["A"], -signal["A"]
+    np.testing.assert_allclose(
+        wheel.contribution("signal")["A"], signal["A"], rtol=1e-12, atol=1e-12
     )
+    np.testing.assert_allclose(
+        wheel.working_residual.channel_data["A"], -signal["A"], rtol=1e-12, atol=1e-12
+    )


 class NoiseOnly:
+    role = "noise"
     name = "noise"

     @dataclass
@@ -133,6 +136,7 @@


 class SignalOnly:
+    role = "signal"
     name = "signal"

     @dataclass
@@ -165,9 +169,10 @@
         )


+# Even records only: a WDM home grid needs two even division counts.
 @pytest.mark.parametrize(
     "num_time_samples,index,endpoint",
-    [(1, 0, "DC"), (4, 0, "DC"), (4, -1, "Nyquist"), (5, 0, "DC")],
+    [(4, 0, "DC"), (4, -1, "Nyquist"), (8, 0, "DC")],
 )
 def test_rejected_rfft_signal_preserves_estimates_state_and_rng(
     observed, num_time_samples, index, endpoint
@@ -191,10 +196,13 @@
                 return replace(result, template=signal)
             return result

-    actual, expected = Wheel(observed, random_seed=8), Wheel(observed, random_seed=8)
+    actual, expected = (
+        Wheel(observed, num_frequency_divisions=2, random_seed=8),
+        Wheel(observed, num_frequency_divisions=2, random_seed=8),
+    )
     actual_block, expected_block = InvalidOnce(), SignalOnly()
-    actual.add(actual_block, role="signal")
-    expected.add(expected_block, role="signal")
+    actual.add(actual_block)
+    expected.add(expected_block)
     with pytest.raises(ValueError, match=rf"{endpoint}.*real"):
         actual.run(1)
     assert actual_block.state.steps == 1
@@ -202,8 +210,10 @@
     assert actual.ledger["signal"].parameters == expected.ledger["signal"].parameters
     assert actual.ledger["signal"].parameters == {"amplitude": 0.5}
     assert actual.covariance is None
-    np.testing.assert_array_equal(actual.contribution("signal")["A"], 0.5)
-    np.testing.assert_array_equal(actual.working_residual.channel_data["A"], -0.5)
+    np.testing.assert_allclose(actual.contribution("signal")["A"], 0.5, rtol=1e-12)
+    np.testing.assert_allclose(
+        actual.working_residual.channel_data["A"], -0.5, rtol=1e-12
+    )

     actual.run(1)
     expected.run(1)
@@ -211,16 +221,18 @@
     assert expected_block.state.steps == 1
     assert actual_block.state.draw == expected_block.state.draw
     assert actual.ledger["signal"].parameters == expected.ledger["signal"].parameters
-    np.testing.assert_array_equal(actual.working_residual.channel_data["A"], -1.0)
+    np.testing.assert_allclose(
+        actual.working_residual.channel_data["A"], -1.0, rtol=1e-12
+    )


 @pytest.mark.parametrize("domain", ["time", "frequency"])
 def test_noise_only_result_preserves_data_without_signal_arrays(observed, domain):
     if domain == "frequency":
         observed = observed.to_frequency()
-    wheel = Wheel(observed, random_seed=3)
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)
     block = NoiseOnly()
-    wheel.add(block, role="noise")
+    wheel.add(block)
     wheel.run(2)

     result = wheel.ledger["noise"]
@@ -228,12 +240,19 @@
     assert result.parameters == {"variance": 3}
     assert block.state.steps == 2
     np.testing.assert_array_equal(result.covariance.A, white_covariance(observed, 3).A)
-    np.testing.assert_array_equal(wheel.covariance.A, white_covariance(observed, 3).A)
+    np.testing.assert_allclose(
+        transform(wheel.covariance, domain).A,
+        white_covariance(observed, 3).A,
+        rtol=1e-12,
+    )
     contribution = wheel.contribution("noise")
     assert set(contribution) == set(observed.channel_names)
     for channel, samples in observed.channel_data.items():
-        np.testing.assert_array_equal(
-            wheel.working_residual.channel_data[channel], samples
+        np.testing.assert_allclose(
+            wheel.working_residual.channel_data[channel],
+            samples,
+            rtol=1e-12,
+            atol=1e-12,
         )
         np.testing.assert_array_equal(contribution[channel], np.zeros_like(samples))
         assert contribution[channel].dtype == samples.dtype
@@ -257,6 +276,7 @@
         )

     class Population:
+        role = "signal"
         name = "population"

         @dataclass
@@ -279,10 +299,10 @@
                 return BlockState(parameters={"num_sources": 0})
             return population_start(conditional_residual)

-    wheel = Wheel(observed)
-    wheel.add(NoiseOnly(), role="noise")
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(NoiseOnly())
     population = Population()
-    wheel.add(population, role="signal")
+    wheel.add(population)
     np.testing.assert_allclose(
         wheel.working_residual.channel_data["A"], observed.channel_data["A"] - 0.25
     )
@@ -292,7 +312,11 @@
     np.testing.assert_allclose(
         wheel.working_residual.channel_data["A"], observed.channel_data["A"]
     )
-    np.testing.assert_array_equal(wheel.covariance.A, white_covariance(observed, 2).A)
+    np.testing.assert_allclose(
+        transform(wheel.covariance, observed.data_domain).A,
+        white_covariance(observed, 2).A,
+        rtol=1e-12,
+    )
     wheel.run(1)
     assert wheel.ledger["population"].parameters == {"num_sources": 1}
     assert population.state.steps == 2
@@ -302,6 +326,7 @@


 class Baseline:
+    role = "signal"
     name = "baseline"

     def sample(self, conditional_residual, covariance, current_block_state, *, rng):
@@ -335,19 +360,18 @@
                 return replace(result, covariance=white_covariance(wrong_grid, 2))
             return result

-    actual, expected = Wheel(observed, random_seed=8), Wheel(observed, random_seed=8)
+    actual, expected = (
+        Wheel(observed, num_frequency_divisions=8, random_seed=8),
+        Wheel(observed, num_frequency_divisions=8, random_seed=8),
+    )
     actual_block, expected_block = FailsOnce(), NoiseOnly()
     for wheel, block in ((actual, actual_block), (expected, expected_block)):
         # A constant signal gives the residual a non-zero baseline to preserve.
-        wheel.add(
-            starting_at(Baseline(), const_block_state(observed, 0.5)), role="signal"
-        )
-        wheel.add(block, role="noise")
+        wheel.add(starting_at(Baseline(), const_block_state(observed, 0.5)))
+        wheel.add(block)
     error = {
-        "omitted": (
-            "must return a covariance because 'noise' is registered as a noise block"
-        ),
-        "template": "returned a template but 'noise' is registered as a noise block",
+        "omitted": "must return a covariance because 'noise' declares role 'noise'",
+        "template": "returned a template but 'noise' declares role 'noise'",
         "wrong_grid": "grid",
     }[failure]
     with pytest.raises(ValueError, match=error):
@@ -355,9 +379,13 @@
     assert actual_block.state.steps == 1
     assert expected_block.state.steps == 0
     assert actual.ledger["noise"].parameters == expected.ledger["noise"].parameters
-    np.testing.assert_array_equal(actual.covariance.A, white_covariance(observed, 1).A)
+    np.testing.assert_allclose(
+        transform(actual.covariance, "time").A,
+        white_covariance(observed, 1).A,
+        rtol=1e-12,
+    )
     np.testing.assert_array_equal(actual.contribution("noise")["A"], 0.0)
-    np.testing.assert_array_equal(actual.contribution("baseline")["A"], 0.5)
+    np.testing.assert_allclose(actual.contribution("baseline")["A"], 0.5, rtol=1e-12)
     np.testing.assert_allclose(
         actual.working_residual.channel_data["A"], observed.channel_data["A"] - 0.5
     )
@@ -368,7 +396,11 @@
     assert actual_block.state.draw == expected_block.state.draw
     assert actual.ledger["noise"].parameters == expected.ledger["noise"].parameters
     assert actual.ledger["noise"].template is None
-    np.testing.assert_array_equal(actual.covariance.A, white_covariance(observed, 2).A)
     np.testing.assert_allclose(
+        transform(actual.covariance, "time").A,
+        white_covariance(observed, 2).A,
+        rtol=1e-12,
+    )
+    np.testing.assert_allclose(
         actual.working_residual.channel_data["A"], observed.channel_data["A"] - 0.5
     )
--- a/tests/test_block_state_contract.py
+++ b/tests/test_block_state_contract.py
@@ -13,6 +13,7 @@
             raise AssertionError("Wheel must not copy resources owned by a block")

     class StatefulSignal:
+        role = "signal"
         name = "signal"

         @dataclass
@@ -38,8 +39,8 @@
             )

     block = StatefulSignal()
-    wheel = Wheel(observed, random_seed=42)
-    wheel.add(block, role="signal")
+    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=42)
+    wheel.add(block)
     wheel.run(2)

     accepted = wheel.ledger["signal"]
@@ -48,7 +49,9 @@
     assert isinstance(block.state.metadata["resource"], LocalResource)
     assert accepted.parameters == {"amplitude": 2.0}
     for name in observed.channel_names:
-        np.testing.assert_array_equal(accepted.template[name], 2.0)
+        # The ledger keeps the template on the home grid; the views are on the data's.
+        assert accepted.template[name].shape == wheel.home_grid.channel_shape
+        np.testing.assert_allclose(wheel.contribution("signal")[name], 2.0, rtol=1e-12)
         np.testing.assert_allclose(
             wheel.working_residual.channel_data[name], observed.channel_data[name] - 2.0
         )
@@ -57,4 +60,4 @@
     accepted.parameters["amplitude"] = 100.0
     accepted.template["A"][:] = 100.0
     assert wheel.ledger["signal"].parameters == {"amplitude": 2.0}
-    np.testing.assert_array_equal(wheel.contribution("signal")["A"], 2.0)
+    np.testing.assert_allclose(wheel.contribution("signal")["A"], 2.0, rtol=1e-12)
--- a/tests/test_ledger.py
+++ b/tests/test_ledger.py
@@ -6,6 +6,7 @@


 class AmplitudeUpdate:
+    role = "signal"
     name = "amplitudes"

     def initial_sample(self, conditional_residual, covariance, *, rng):
@@ -17,8 +18,8 @@


 def test_ledger_identity_is_reflexive_and_snapshots_are_stable(observed):
-    wheel = Wheel(observed)
-    wheel.add(AmplitudeUpdate(), role="signal")
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(AmplitudeUpdate())
     assert wheel.ledger == wheel.ledger
     snapshot = wheel.ledger.snapshot()
     entry = snapshot["amplitudes"]
@@ -33,7 +34,7 @@


 def test_empty_and_populated_snapshot_names_follow_registration(observed):
-    wheel = Wheel(observed)
+    wheel = Wheel(observed, num_frequency_divisions=8)
     assert wheel.ledger.snapshot() == {}
-    wheel.add(AmplitudeUpdate(), role="signal")
+    wheel.add(AmplitudeUpdate())
     assert list(wheel.ledger.snapshot()) == list(wheel.ledger) == ["amplitudes"]
--- a/tests/test_translation_review.py
+++ b/tests/test_translation_review.py
@@ -37,6 +37,8 @@
     received = []

     class RecordInputs:
+        role = "signal"
+
         def __init__(self, name):
             self.name = name

@@ -49,21 +51,22 @@
             received.append(("sample", self.name, conditional_residual.data_domain))
             return current_block_state

-    wheel = Wheel(observed)
-    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 2.0)), role="noise")
-    wheel.add(RecordInputs("default"), role="signal")
-    wheel.add(RecordInputs("explicit"), role="signal", data_domain="time")
+    wheel = Wheel(observed, num_frequency_divisions=8)
+    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 2.0)))
+    wheel.add(RecordInputs("default"))
+    wheel.add(RecordInputs("explicit"), data_domain="time")
     assert received == []
     wheel.run(1)
     assert received == [("sample", "default", "time"), ("sample", "explicit", "time")]
-    np.testing.assert_array_equal(wheel.covariance.A, 2.0)
+    np.testing.assert_allclose(wheel.covariance.A, 2.0 / observed.fs, rtol=1e-12)


-def test_same_grid_block_receives_the_covariance_without_transform(
+def test_home_grid_block_receives_the_covariance_without_transform(
     observed, monkeypatch
 ):
-    """An explicit domain equal to the observations' hands the covariance over
-    as a deep copy; nothing goes through transform()."""
+    """A block on the home grid is handed the covariance as a deep copy; nothing
+    goes through transform(). WDM observations make the arrival grid the home
+    grid, so a default registration is on it too."""
     import enchilada.wheel

     delegate = enchilada.wheel.transform
@@ -77,20 +80,20 @@
     received = []

     class RecordCovariance:
+        role = "signal"
         name = "explicit"

         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
             covariance.check_compatible(conditional_residual)
-            received.append(float(covariance.A[0]))
+            received.append(float(covariance.A.flat[0]))
             return current_block_state

-    wheel = Wheel(observed)
-    wheel.add(FixedNoiseBlock("noise", white_covariance(observed, 2.0)), role="noise")
-    wheel.add(
-        starting_at(RecordCovariance(), BlockState()), role="signal", data_domain="time"
-    )
+    home = transform(observed, "wdm", num_frequency_divisions=8)
+    wheel = Wheel(home)
+    wheel.add(FixedNoiseBlock("noise", white_covariance(home, 2.0)))
+    wheel.add(starting_at(RecordCovariance(), BlockState()), data_domain="wdm")
     wheel.run(2)
-    assert received == [2.0, 2.0]
+    assert received == [2.0 / observed.fs, 2.0 / observed.fs]


 @pytest.mark.parametrize("canonical_domain", ["time", "wdm"])
@@ -105,6 +108,7 @@
     fail = [True]

     class MutateInputs:
+        role = "signal"
         name = "mutate"

         def sample(self, conditional_residual, covariance, current_block_state, *, rng):
@@ -122,8 +126,8 @@
                 parameters={"amplitude": amplitude, "coordinates": np.array([3.0])}
             )

-    actual = Wheel(canonical, random_seed=7)
-    control = Wheel(canonical, random_seed=7)
+    actual = Wheel(canonical, num_frequency_divisions=8, random_seed=7)
+    control = Wheel(canonical, num_frequency_divisions=8, random_seed=7)
     for wheel in (actual, control):
         wheel.add(
             starting_at(
@@ -132,7 +136,6 @@
                     parameters={"amplitude": 0.0, "coordinates": np.array([3.0])}
                 ),
             ),
-            role="signal",
             data_domain="frequency",
         )
     previous_amplitude = actual.ledger["mutate"].parameters["amplitude"]
@@ -146,8 +149,10 @@
         np.testing.assert_array_equal(
             actual.observed_data.channel_data[name], canonical.channel_data[name]
         )
-        np.testing.assert_array_equal(
-            actual.residual().channel_data[name], canonical.channel_data[name]
+        np.testing.assert_allclose(
+            actual.residual().channel_data[name],
+            canonical.channel_data[name],
+            rtol=1e-12,
         )
     fail[0] = False
     actual.run(1)
--- a/tests/test_data.py
+++ b/tests/test_data.py
@@ -42,8 +42,8 @@
             channel_data={"A": data.channel_data["A"], "E": data.channel_data["E"]},
         )
         covariance = white_covariance(data, {"A": 1.0, "E": 4.0})
-        wheel = Wheel(reordered)
-        wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
+        wheel = Wheel(reordered, num_frequency_divisions=2)
+        wheel.add(FixedNoiseBlock("noise", covariance))
         assert wheel.observed_data.channel_names == ("E", "A")
         assert reordered.to_frequency().to_time().channel_names == ("E", "A")

```

- [ ] **Step 3: Run the tests, ruff and the grep**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py
.venv/bin/python -m ruff check tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py
.venv/bin/python -m ruff format --check tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py
grep -n "role=" tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py
```

Expected: `160 passed`, `All checks passed!`, `6 files already formatted`, and the grep prints nothing.

- [ ] **Step 4: Commit**

```bash
git branch --show-current   # wdm-home
git add tests/test_state.py tests/test_block_state.py tests/test_block_state_contract.py tests/test_ledger.py tests/test_translation_review.py tests/test_data.py
git commit -m "Tests: migrate the state, ledger, review and data suites to the home grid" -m "Roles as attributes, home grids named at construction, views compared to rounding, covariances read on the home grid or translated back; the odd-record parametrisations move to even records a WDM grid can carry, and the overflow-at-registration test runs on WDM observations." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

### Task 5: Examples, benchmark and their tests

**Goal:** The six example scripts, the benchmark and their four test files use the declared role and a WDM home grid sized to each record (demo 1024 → 32 rows, `toy_fit` 4096 → 64, `noise_floor` 2048 → 32, `domain_translation` 128 → 8, the GB contract tests' 64-sample records → 8, the benchmark's `--rows`, default 128; `mojito_wdm_wheel.py` arrives on WDM and needs no count), their prose describes the home grid and the translated noise sum, and their assertions read covariances natively in the ledger or translated back from the home grid.

**Files:**
- Modify: `examples/demo.py:38-40`
- Modify: `examples/toy_fit.py` (module docstring lines 7–13; `SineBlock` and `WhiteNoiseBlock` gain `role`; `run_toy_fit` lines 200–206)
- Modify: `examples/noise_floor.py` (import line 25; module docstring lines 3–8; `ExcessNoiseBlock` gains `role`; `run_example` lines 183–204, whose progress print reads the covariance on the time grid)
- Modify: `examples/domain_translation.py` (`DomainEcho` gains `role`; `run_example` lines 83–90)
- Modify: `examples/gb_model.py` (`GBBlock` docstring lines 136 and 263–265; `role = "signal"` before `SUPPORTED_CHANNELS`)
- Modify: `examples/mojito_wdm_wheel.py:83-84`
- Replace: `benchmarks/benchmark_wheel.py` (whole file)
- Modify: `tests/test_examples.py:78-81, 102-106`
- Modify: `tests/test_toy_translation.py:30-35, 52-54`
- Modify: `tests/test_gb_example_contract.py:134-166`
- Modify: `tests/test_noise_floor_example.py` (import; lines 40–43, 79–84, 110–112)

**Acceptance Criteria:**
- [ ] `grep -c "role=" examples/demo.py examples/toy_fit.py examples/noise_floor.py examples/domain_translation.py examples/gb_model.py examples/mojito_wdm_wheel.py benchmarks/benchmark_wheel.py tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_noise_floor_example.py` prints `0` for every file; `SineBlock`, `DomainEcho`, `GBBlock`, `FixedSignal` and the contract test's `ContextProbe` carry `role = "signal"`, `WhiteNoiseBlock` and `ExcessNoiseBlock` carry `role = "noise"`.
- [ ] Every `Wheel(...)` and `check_block(...)` on time or frequency data names its home grid with the counts above; `mojito_wdm_wheel.py` constructs `Wheel(observed, random_seed=0)` on WDM observations with a comment saying why no count is needed.
- [ ] `benchmarks/benchmark_wheel.py` takes `--rows` (default 128), reports `"rows"`, and `python benchmarks/benchmark_wheel.py --samples 4096 --rows 16 --blocks 2 --cycles 1 --repeats 1` prints one JSON line.
- [ ] `tests/test_noise_floor_example.py` reads the campaign covariance through `transform(wheel.covariance, "time")` and the ledger entries natively; `tests/test_gb_example_contract.py` checks the ledger publication's grid, `wheel.covariance.grid == wheel.home_grid`, the home-grid template shape and translates the result with a home-grid reference; `tests/test_toy_translation.py` builds both campaigns on one 8 × 512 home grid and checks both ledgers hold the time-grid publication.
- [ ] The four test files pass (with `tests/test_examples.py::test_notebook_code_cells_execute` deselected, it belongs to Task 6); ruff check and format clean on every file of this task.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_noise_floor_example.py --deselect tests/test_examples.py::test_notebook_code_cells_execute` → `0 failed` (44 collected; the GB-stack tests that skip today still skip); `.venv/bin/python -m ruff check examples/demo.py examples/toy_fit.py examples/noise_floor.py examples/domain_translation.py examples/gb_model.py examples/mojito_wdm_wheel.py benchmarks/benchmark_wheel.py tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_noise_floor_example.py` → `All checks passed!`; `.venv/bin/python -m ruff format --check` on the same eleven paths → `11 files already formatted`; `.venv/bin/python benchmarks/benchmark_wheel.py --samples 4096 --rows 16 --blocks 2 --cycles 1 --repeats 1` → one JSON line containing `"rows": 16`.

**Steps:**

- [ ] **Step 1: Confirm the branch and see the red state**

```bash
git branch --show-current          # wdm-home, or stop
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_toy_translation.py tests/test_noise_floor_example.py 2>&1 | tail -3
```

Expected: failures with `unexpected keyword argument 'role'` / `declares no role` and the division-count `ValueError`.

- [ ] **Step 2: Save and run the example edit script**

Save this as `$TMPDIR/examples5.py` and run it from the repository root with `.venv/bin/python "$TMPDIR/examples5.py"`. Every old text must occur exactly once in its file; on an assertion failure make that substitution by hand and delete the call before rerunning.

```python
"""Task 5: the examples, the benchmark and their tests on the home grid."""

import pathlib


def edit(path, old, new, count=1):
    p = pathlib.Path(path)
    text = p.read_text()
    found = text.count(old)
    assert found == count, f"{path}: expected {count} of {old!r}, found {found}"
    p.write_text(text.replace(old, new))


# ------------------------------------------------------------------- demo.py
edit(
    "examples/demo.py",
    "wheel = Wheel(observed, random_seed=0)\n"
    'wheel.add(ucb, role="signal")\n'
    'wheel.add(mbhb, role="signal")\n',
    "# The campaign lives on a WDM home grid: 32 rows x 32 columns for 1024 samples.\n"
    '# EchoBlock declares role = "signal"; Wheel reads the attribute at add.\n'
    "wheel = Wheel(observed, num_frequency_divisions=32, random_seed=0)\n"
    "wheel.add(ucb)\n"
    "wheel.add(mbhb)\n",
)

# ---------------------------------------------------------------- toy_fit.py
TOY = "examples/toy_fit.py"
edit(
    TOY,
    "the blocks. The sine blocks register as signal blocks and the white-noise block\n"
    "as the one noise block, so the sines receive exactly its covariance and the\n"
    "noise block itself receives None because it is alone. These\n"
    "blocks request time-domain inputs; Wheel also runs them on Fourier or WDM\n"
    'observations with data_domain="time", because the noise block publishes its\n'
    "covariance on that time grid. Their likelihood requires time-domain noise with\n"
    "independent channels.\n",
    'the blocks. The sine blocks declare role = "signal" and the white-noise block\n'
    'role = "noise" (the one noise block), so the sines receive its covariance,\n'
    "translated onto their time grid (exact for white noise), and the noise block\n"
    "itself receives None because it is alone. These blocks request time-domain\n"
    "inputs; the campaign lives on a WDM home grid (64 rows x 64 columns for the\n"
    "4096 samples) and Wheel also runs them on Fourier or WDM observations with\n"
    'data_domain="time". Their likelihood requires time-domain noise with\n'
    "independent channels.\n",
)
edit(
    TOY,
    '    from the supplied grid each call.\n    """\n',
    '    from the supplied grid each call.\n    """\n\n    role = "signal"\n',
)
edit(
    TOY,
    '    (register it as the only noise block).\n    """\n',
    '    (register it as the only noise block).\n    """\n\n    role = "noise"\n',
)
edit(
    TOY,
    "    wheel = Wheel(observed, random_seed=int(campaign_seed.generate_state(1)[0]))\n"
    "    # The sine starts need no covariance, so the signal blocks may register first.\n"
    '    for name, frequency in (("slow", 0.004), ("fast", 0.011)):\n'
    "        wheel.add(\n"
    '            SineBlock(name=name, freq=frequency), role="signal", data_domain="time"\n'
    "        )\n"
    '    wheel.add(WhiteNoiseBlock(name="noise"), role="noise", data_domain="time")\n',
    "    wheel = Wheel(\n"
    "        observed,\n"
    "        num_frequency_divisions=64,  # the WDM home grid: 64 rows x 64 columns\n"
    "        random_seed=int(campaign_seed.generate_state(1)[0]),\n"
    "    )\n"
    "    # The sine starts need no covariance, so the signal blocks may register first.\n"
    '    for name, frequency in (("slow", 0.004), ("fast", 0.011)):\n'
    '        wheel.add(SineBlock(name=name, freq=frequency), data_domain="time")\n'
    '    wheel.add(WhiteNoiseBlock(name="noise"), data_domain="time")\n',
)

# ------------------------------------------------------------ noise_floor.py
FLOOR = "examples/noise_floor.py"
edit(
    FLOOR,
    'The campaign registers two blocks with role="noise". FixedNoiseBlock publishes\n'
    "the known instrument floor on every call; ExcessNoiseBlock samples an extra\n"
    "white-noise variance with a Metropolis random walk in log-variance. Wheel hands\n"
    "the excess block the floor as its `covariance` (the sum over the other noise\n"
    "blocks) and keeps the campaign covariance equal to floor + excess, which is what\n"
    "a signal block registered here would receive.\n",
    'The campaign registers two noise blocks, each declaring role = "noise".\n'
    "FixedNoiseBlock publishes the known instrument floor on every call;\n"
    "ExcessNoiseBlock samples an extra white-noise variance with a Metropolis random\n"
    "walk in log-variance. Wheel hands the excess block the floor as its `covariance`\n"
    "(the sum over the other noise blocks, translated onto the block's time grid) and\n"
    "keeps the campaign covariance equal to floor + excess on its WDM home grid,\n"
    "which a signal block registered on any grid would receive translated onto its\n"
    "own.\n",
)
edit(
    FLOOR,
    "    refuses a `None` floor, so register it after another noise block.\n"
    '    """\n',
    "    refuses a `None` floor, so register it after another noise block.\n"
    '    """\n\n    role = "noise"\n',
)
edit(
    FLOOR,
    "    wheel = Wheel(observed, random_seed=int(campaign_seed.generate_state(1)[0]))\n"
    "    floor = Covariance.like(\n",
    "    wheel = Wheel(\n"
    "        observed,\n"
    "        num_frequency_divisions=32,  # the WDM home grid: 32 rows x 64 columns\n"
    "        random_seed=int(campaign_seed.generate_state(1)[0]),\n"
    "    )\n"
    "    floor = Covariance.like(\n",
)
edit(
    FLOOR,
    "    # refuses to start without it; Wheel keeps floor + excess as the campaign\n"
    "    # covariance.\n"
    '    wheel.add(FixedNoiseBlock("floor", floor), role="noise")\n'
    '    wheel.add(ExcessNoiseBlock("excess"), role="noise")\n',
    "    # refuses to start without it; Wheel keeps floor + excess as the campaign\n"
    "    # covariance on its home grid.\n"
    '    wheel.add(FixedNoiseBlock("floor", floor))\n'
    '    wheel.add(ExcessNoiseBlock("excess"))\n',
)
edit(
    FLOOR,
    "from enchilada import BlockState, Covariance, FixedNoiseBlock, L1Data, Wheel\n",
    "from enchilada import BlockState, Covariance, FixedNoiseBlock, L1Data, Wheel, transform\n",
)
edit(
    FLOOR,
    "            print(\n"
    '                f"cycle {cycle + 1:4d}: excess variance draw = {chain[-1]:.4f}  "\n'
    '                f"(campaign covariance A[0] = {w.covariance.A[0]:.4f})"\n'
    "            )\n",
    "            # The campaign covariance is on the home grid; read it on the time grid.\n"
    '            total = transform(w.covariance, "time").A[0]\n'
    "            print(\n"
    '                f"cycle {cycle + 1:4d}: excess variance draw = {chain[-1]:.4f}  "\n'
    '                f"(campaign covariance A[0] = {total:.4f})"\n'
    "            )\n",
)

# ----------------------------------------------------- domain_translation.py
TRANSLATION = "examples/domain_translation.py"
edit(
    TRANSLATION,
    '    """A fixed zero-signal model recording the representation it receives."""\n\n'
    "    name: str\n",
    '    """A fixed zero-signal model recording the representation it receives."""\n\n'
    '    role = "signal"\n\n'
    "    name: str\n",
)
edit(
    TRANSLATION,
    "    # Wheel keeps time as the canonical grid while preparing each block's inputs.\n"
    "    wheel = Wheel(observed, random_seed=3)\n",
    "    # The campaign lives on a WDM home grid (8 rows x 16 columns for 128 samples)\n"
    "    # and serves each block on the grid it asks for.\n"
    "    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)\n",
)
edit(
    TRANSLATION,
    '        wheel.add(block, role="signal", data_domain=domain, **options)\n',
    "        wheel.add(block, data_domain=domain, **options)\n",
)

# --------------------------------------------------------------- gb_model.py
GB = "examples/gb_model.py"
edit(
    GB,
    '    Register with ``wheel.add(block, role="signal", data_domain="frequency")``.\n',
    '    Register with ``wheel.add(block, data_domain="frequency")``; the block\n'
    '    declares ``role = "signal"``.\n',
)
edit(
    GB,
    '    """\n\n    SUPPORTED_CHANNELS = ("A", "E")\n',
    '    """\n\n    role = "signal"\n    SUPPORTED_CHANNELS = ("A", "E")\n',
)
edit(
    GB,
    '        first (for a known PSD, ``FixedNoiseBlock`` with ``role="noise"`` and\n'
    '        ``data_domain="frequency"``). It resets ``self.state``, so a fresh block\n'
    "        instance belongs to each independent campaign.\n",
    "        first (for a known PSD, a ``FixedNoiseBlock``, which declares\n"
    '        ``role = "noise"``, with ``data_domain="frequency"``). It resets\n'
    "        ``self.state``, so a fresh block instance belongs to each independent\n"
    "        campaign.\n",
)

# ------------------------------------------------------ mojito_wdm_wheel.py
edit(
    "examples/mojito_wdm_wheel.py",
    "    wheel = Wheel(observed, random_seed=0)\n"
    '    wheel.add(EchoBlock(name="echo"), role="signal")\n',
    "    # WDM observations: their grid is the campaign's home grid, no count needed.\n"
    "    wheel = Wheel(observed, random_seed=0)\n"
    '    wheel.add(EchoBlock(name="echo"))\n',
)

# ---------------------------------------------------------- test_examples.py
EXAMPLES = "tests/test_examples.py"
edit(
    EXAMPLES,
    "    campaigns = [Wheel(observed, random_seed=seed) for seed in (42, 42, 43)]\n"
    "    for wheel in campaigns:\n"
    '        wheel.add(toy.SineBlock(name="slow", freq=0.004), role="signal")\n'
    '        wheel.add(toy.WhiteNoiseBlock(name="noise"), role="noise")\n',
    "    campaigns = [\n"
    "        Wheel(observed, num_frequency_divisions=64, random_seed=seed)\n"
    "        for seed in (42, 42, 43)\n"
    "    ]\n"
    "    for wheel in campaigns:\n"
    '        wheel.add(toy.SineBlock(name="slow", freq=0.004))\n'
    '        wheel.add(toy.WhiteNoiseBlock(name="noise"))\n',
)
edit(
    EXAMPLES,
    "    wheel = Wheel(observed, random_seed=0)\n"
    '    floor = Covariance.like(observed, {"A": np.full(observed.num_time_samples, 0.2)})\n'
    '    wheel.add(FixedNoiseBlock("floor", floor), role="noise", data_domain="time")\n'
    '    with pytest.raises(ValueError, match="only noise block"):\n'
    '        wheel.add(toy.WhiteNoiseBlock(name="noise"), role="noise", data_domain="time")\n',
    "    wheel = Wheel(observed, num_frequency_divisions=64, random_seed=0)\n"
    '    floor = Covariance.like(observed, {"A": np.full(observed.num_time_samples, 0.2)})\n'
    '    wheel.add(FixedNoiseBlock("floor", floor), data_domain="time")\n'
    '    with pytest.raises(ValueError, match="only noise block"):\n'
    '        wheel.add(toy.WhiteNoiseBlock(name="noise"), data_domain="time")\n',
)

# --------------------------------------------------- test_toy_translation.py
TOY_TEST = "tests/test_toy_translation.py"
edit(
    TOY_TEST,
    "    for observed in (data, translated_data):\n"
    "        wheel = Wheel(observed, random_seed=42)\n"
    '        for name, frequency in (("slow", 0.004), ("fast", 0.011)):\n'
    '            wheel.add(toy.SineBlock(name, frequency), role="signal", data_domain="time")\n'
    '        noise_block = toy.WhiteNoiseBlock("noise")\n'
    '        wheel.add(noise_block, role="noise", data_domain="time")\n',
    "    for observed in (data, translated_data):\n"
    "        # One 8 x 512 home grid for both campaigns (the WDM observations' own).\n"
    "        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=42)\n"
    '        for name, frequency in (("slow", 0.004), ("fast", 0.011)):\n'
    '            wheel.add(toy.SineBlock(name, frequency), data_domain="time")\n'
    '        noise_block = toy.WhiteNoiseBlock("noise")\n'
    '        wheel.add(noise_block, data_domain="time")\n',
)
edit(
    TOY_TEST,
    "    # Both wheels hold the noise block's publication on the time grid it\n"
    "    # samples on, whatever grid the observations are stored on.\n"
    "    assert wheels[1].covariance.grid == wheels[0].covariance.grid\n",
    "    # Both ledgers hold the noise block's publication on the time grid it\n"
    "    # samples on; both sums live on the shared home grid.\n"
    "    for wheel in wheels:\n"
    '        assert wheel.ledger["noise"].covariance.grid == data.grid\n'
    "        assert wheel.covariance.grid == wheel.home_grid\n"
    "    assert wheels[1].covariance.grid == wheels[0].covariance.grid\n",
)

# ----------------------------------------------- test_gb_example_contract.py
GB_TEST = "tests/test_gb_example_contract.py"
edit(
    GB_TEST,
    "    observed = transform(data, storage_domain, **options)\n"
    "    wheel = Wheel(observed, random_seed=0)\n"
    "    # Noise registers first: GBBlock.initial_sample needs the covariance.\n"
    '    wheel.add(FixedNoiseBlock("noise", noise), role="noise", data_domain="frequency")\n',
    "    observed = transform(data, storage_domain, **options)\n"
    "    # An 8 x 8 home grid: the WDM observations' own grid, or the time data's.\n"
    "    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=0)\n"
    "    # Noise registers first: GBBlock.initial_sample needs the covariance.\n"
    '    wheel.add(FixedNoiseBlock("noise", noise), data_domain="frequency")\n',
)
edit(
    GB_TEST,
    "        name = block.name\n\n        def initial_sample(",
    '        name = block.name\n        role = "signal"\n\n        def initial_sample(',
)
edit(
    GB_TEST,
    '    wheel.add(ContextProbe(), role="signal", data_domain="frequency")\n',
    '    wheel.add(ContextProbe(), data_domain="frequency")\n',
)
edit(
    GB_TEST,
    "    assert wheel.covariance.grid == data.grid\n"
    '    result = wheel.ledger["gb"]\n'
    '    assert result.template["A"].shape == observed.channel_data["A"].shape\n'
    '    frequency_result = transform(result, "frequency", reference_data=observed)\n',
    "    # The publication is kept natively; the sum is its home-grid translation.\n"
    '    assert wheel.ledger["noise"].covariance.grid == data.grid\n'
    "    assert wheel.covariance.grid == wheel.home_grid\n"
    '    result = wheel.ledger["gb"]\n'
    '    assert result.template["A"].shape == wheel.home_grid.channel_shape\n'
    '    home = transform(observed, "wdm", num_frequency_divisions=8)\n'
    '    frequency_result = transform(result, "frequency", reference_data=home)\n',
)

# ---------------------------------------------- test_noise_floor_example.py
FLOOR_TEST = "tests/test_noise_floor_example.py"
edit(
    FLOOR_TEST,
    "from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel\n",
    "from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel, transform\n",
)
edit(
    FLOOR_TEST,
    "    np.testing.assert_allclose(\n"
    "        wheel.covariance.A,\n"
    '        noise_floor.TRUTH["floor"] + wheel.ledger["excess"].covariance.A,\n'
    "    )\n",
    "    # The sum lives on the home grid; translated back to time it is floor + excess.\n"
    "    np.testing.assert_allclose(\n"
    '        transform(wheel.covariance, "time").A,\n'
    '        noise_floor.TRUTH["floor"] + wheel.ledger["excess"].covariance.A,\n'
    "    )\n",
)
edit(
    FLOOR_TEST,
    "        observed,\n"
    '        role="noise",\n'
    '        noise_blocks=(FixedNoiseBlock("floor", white_covariance(observed, 0.25)),),\n'
    "    )\n",
    "        observed,\n"
    '        noise_blocks=(FixedNoiseBlock("floor", white_covariance(observed, 0.25)),),\n'
    "        num_frequency_divisions=32,\n"
    "    )\n",
)
edit(
    FLOOR_TEST,
    "    wheel = Wheel(observed, random_seed=1)\n"
    '    wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 0.5)), role="noise")\n'
    '    wheel.add(noise_floor.ExcessNoiseBlock("excess"), role="noise")\n',
    "    wheel = Wheel(observed, num_frequency_divisions=32, random_seed=1)\n"
    '    wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 0.5)))\n'
    '    wheel.add(noise_floor.ExcessNoiseBlock("excess"))\n',
)
print("examples edited")
```

Expected output: `examples edited`.

- [ ] **Step 3: Replace the benchmark**

Overwrite `benchmarks/benchmark_wheel.py` with:

```python
"""Measure complete sampling cycles with fixed signals and covariance.

Run from an installed checkout: python benchmarks/benchmark_wheel.py
Use --samples, --rows and --blocks to explore a different workload. Timings
include defensive snapshots and validation; these are measurements, not test
thresholds. The workload is the fixed signals plus one FixedNoiseBlock, which
republishes its covariance every cycle and is revalidated by Wheel each time, so
timings are not comparable with runs from before the noise sum. The campaign
lives on a WDM home grid of --rows rows; every block here works on the time grid
the data arrive on, so each cycle pays the data transforms and the covariance
translation between that grid and the home grid.
"""

import argparse
import json
import resource
import statistics
import time

import numpy as np

from enchilada import Covariance, FixedNoiseBlock, L1Data, Wheel


class FixedSignal:
    role = "signal"

    def __init__(self, name, level):
        self.name = name
        self.level = level

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return conditional_residual.block_state(
            {
                name: np.full_like(values, self.level)
                for name, values in conditional_residual.channel_data.items()
            }
        )

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return current_block_state


def measure(num_blocks, num_samples, num_rows, num_cycles, repeats):
    observed = L1Data(
        channel_data={name: np.ones(num_samples) for name in ("A", "E", "T")},
        channel_names=("A", "E", "T"),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="strain",
    )
    covariance = Covariance.like(
        observed, {name: np.ones(num_samples) for name in observed.channel_names}
    )
    wheel = Wheel(observed, num_frequency_divisions=num_rows, random_seed=0)
    # The noise goes first so every signal block sees the covariance from
    # registration onward, as before.
    wheel.add(FixedNoiseBlock("noise", covariance))
    for index in range(num_blocks):
        wheel.add(FixedSignal(str(index), 0.5 / num_blocks))
    wheel.run(1)
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        wheel.run(num_cycles)
        elapsed.append((time.perf_counter() - start) / num_cycles)
    np.testing.assert_allclose(wheel.residual().channel_data["A"], 0.5)
    return {
        "blocks": num_blocks,
        "noise_blocks": 1,
        "samples_per_channel": num_samples,
        "rows": num_rows,
        "channels": 3,
        "median_seconds_per_cycle": statistics.median(elapsed),
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument(
        "--rows",
        type=int,
        default=128,
        help="rows of the WDM home grid; --samples / --rows must be even",
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    for num_blocks in arguments.blocks:
        print(
            json.dumps(
                measure(
                    num_blocks,
                    arguments.samples,
                    arguments.rows,
                    arguments.cycles,
                    arguments.repeats,
                )
            ),
            flush=True,
        )
```

(No test or document calls `measure`, so its new positional `num_rows` breaks nothing.)

- [ ] **Step 4: Format, then run the Verify commands and the greps**

```bash
FILES=(examples/demo.py examples/toy_fit.py examples/noise_floor.py examples/domain_translation.py examples/gb_model.py examples/mojito_wdm_wheel.py benchmarks/benchmark_wheel.py tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_noise_floor_example.py)
.venv/bin/python -m ruff format "${FILES[@]}"
.venv/bin/python -m ruff check "${FILES[@]}"
.venv/bin/python -m ruff format --check "${FILES[@]}"
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_noise_floor_example.py --deselect tests/test_examples.py::test_notebook_code_cells_execute
.venv/bin/python benchmarks/benchmark_wheel.py --samples 4096 --rows 16 --blocks 2 --cycles 1 --repeats 1
.venv/bin/python examples/domain_translation.py | head -5
grep -c "role=" "${FILES[@]}"
```

Expected: `All checks passed!`, `11 files already formatted`, `0 failed` (the GB-stack skips stay), one JSON line with `"rows": 16`, the translation example's JSON report, eleven `:0` counts. (`zsh` arrays as above; in bash the same lines work.) If `test_toy_fit_converges_to_truth` or the noise-floor recovery test fails on its tolerance, the cause is a changed random stream, not the grid: the blocks' draws are unchanged by this task, so look for a registration that consumes RNG differently (compare with `git stash` on the example) before loosening anything.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # wdm-home
git add examples/demo.py examples/toy_fit.py examples/noise_floor.py examples/domain_translation.py examples/gb_model.py examples/mojito_wdm_wheel.py benchmarks/benchmark_wheel.py tests/test_examples.py tests/test_toy_translation.py tests/test_gb_example_contract.py tests/test_noise_floor_example.py
git commit -m "Examples: declared roles and a WDM home grid in the scripts, the benchmark and their tests" -m "Each example sizes its home grid to its record (32, 64, 32, 8 rows; the Mojito example arrives on WDM); the benchmark takes --rows; the tests read noise publications natively in the ledger and the sum translated back from the home grid." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

### Task 6: Notebooks

**Goal:** The four example notebooks and the tutorial test speak the new API: blocks declare `role`, campaigns name their WDM home grid, the prose explains the home grid and the covariance translation instead of the one-grid rule, and every assertion on a view allows rounding.

**Files:**
- Modify: `examples/create_block.ipynb` (cells `create-block-03`, `create-block-05`, `create-block-08`, `create-block-09`, `create-block-10`, `create-block-11`, `known-injection-start`, `known-injection-checks`, `create-block-15`, `create-block-18`, `create-block-22`, `create-block-23`, `create-block-24`, `create-block-25`)
- Modify: `examples/domain_translation.ipynb` (cells `translation-05`, `translation-16`, `translation-17`, `translation-18`, `translation-19`)
- Modify: `examples/gb_block_eryn.ipynb` (cells `ff30cc08`, `7261596e`, `ab8d6832`)
- Modify: `examples/demo.ipynb` (raw text substitution: the code cell with `Wheel(observed, random_seed=0)`, the markdown cells mentioning "by its role" and the `check_block` call)
- Modify: `tests/test_block_tutorial.py:147-157`

**Acceptance Criteria:**
- [ ] `grep -c "role=" examples/create_block.ipynb examples/domain_translation.ipynb examples/demo.ipynb examples/gb_block_eryn.ipynb tests/test_block_tutorial.py` prints `0` for every file; `SineAmplitudeBlock`, `DomainProbe` carry `role = "signal"`, `WhiteNoiseVarianceBlock` carries `role = "noise"`.
- [ ] Every `Wheel(...)` and `check_block(...)` in the notebooks names its home grid: `home_rows = 16` in `create_block.ipynb` (256 samples), `num_frequency_divisions=8` in `domain_translation.ipynb` (128 samples) and `demo.ipynb`'s `check_block` text, `32` in `demo.ipynb`'s campaign (1024 samples), `HOME_ROWS = 512` in `gb_block_eryn.ipynb` (2^19 samples); `tests/test_block_tutorial.py` passes `num_frequency_divisions=8` on its 64-sample records.
- [ ] `create_block.ipynb` asserts the known-injection contribution with `assert_allclose(..., atol=1e-12)`, `wheel.covariance.A` against `noise_standard_deviation**2 / sample_rate_hz` and the ledger publication exactly; the joint campaign checks `variance / sample_rate_hz`, and the posterior check subtracts `wheel.contribution("signal")["A"]` (the ledger template is on the 16 × 16 home grid); `domain_translation.ipynb` checks ledger shapes against `wheel.home_grid.channel_shape` and the residual with `assert_allclose(atol=1e-12)`.
- [ ] No notebook or test file says "registered with `role=`", "observation grid" (in the sense of the storage grid), "first noise publication", "refused at `add`" for another grid, "Canonical domain" or "noise sum stays on that grid"; the prose describes the home grid, the native ledger publication and the translated sum.
- [ ] Every notebook JSON round-trips through `json.load`/`json.dump(indent=1, ensure_ascii=False)` byte for byte before editing (`demo.ipynb` is edited as text); every code line stays ≤ 88 columns and `ruff format --check` on the four notebooks is clean.
- [ ] `tests/test_block_tutorial.py`, `tests/test_examples.py::test_notebook_code_cells_execute` and `tests/test_examples.py::test_all_notebooks_contain_valid_python` pass; the code cells of `domain_translation.ipynb` run end to end under `MPLBACKEND=Agg`.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_block_tutorial.py "tests/test_examples.py::test_notebook_code_cells_execute" "tests/test_examples.py::test_all_notebooks_contain_valid_python"` → all passed, 0 failed; `.venv/bin/python -m ruff check examples/create_block.ipynb examples/domain_translation.ipynb examples/demo.ipynb examples/gb_block_eryn.ipynb tests/test_block_tutorial.py` → `All checks passed!`; `.venv/bin/python -m ruff format --check` on the same five paths → `5 files already formatted`; the `domain_translation.ipynb` smoke run in Step 4 prints `Home grid: Grid(data_domain='wdm', ...)` and exits 0.

**Steps:**

- [ ] **Step 1: Confirm the branch and see the red state**

```bash
git branch --show-current          # wdm-home, or stop
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_block_tutorial.py "tests/test_examples.py::test_notebook_code_cells_execute" 2>&1 | tail -3
```

Expected: failures ending in `TypeError: block 'noise' declares no role` / `Wheel.add() got an unexpected keyword argument 'role'` or the division-count `ValueError`.

- [ ] **Step 2: Save and run the notebook edit script**

Save this as `$TMPDIR/notebooks6.py` and run it from the repository root with `.venv/bin/python "$TMPDIR/notebooks6.py"`. It asserts the round trip, asserts every old text occurs exactly once (or the stated count) in its cell, and writes the notebooks back with `indent=1` and a trailing newline. If an assertion fails, the cell text differs from this plan's snapshot: open the cell, make the same substitution by hand, and delete that `edit` call before rerunning.

```python
"""Task 6: migrate the example notebooks to declared roles and the home grid."""

import json
import pathlib


def load(path):
    raw = pathlib.Path(path).read_text()
    nb = json.loads(raw)
    dumped = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    assert dumped == raw, f"{path} does not round-trip through json; edit it as text"
    return nb


def save(path, nb):
    pathlib.Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def edit(nb, cell_id, old, new, count=1):
    (cell,) = [c for c in nb["cells"] if c.get("id", "").startswith(cell_id)]
    source = "".join(cell["source"])
    found = source.count(old)
    assert found == count, f"{cell_id}: expected {count} of {old!r}, found {found}"
    cell["source"] = source.replace(old, new).splitlines(keepends=True)


# ---------------------------------------------------------------- create_block
nb = load("examples/create_block.ipynb")
edit(
    nb,
    "create-block-03",
    "qualifies as a `Block` when it has these three members",
    "qualifies as a `Block` when it has these four members",
)
edit(
    nb,
    "create-block-03",
    "| `name` | Registration and ledger lookup | A unique, nonempty name within this Wheel. |\n",
    "| `name` | Registration and ledger lookup | A unique, nonempty name within this Wheel. |\n"
    '| `role` | Registration | `"signal"` for a block that publishes a template, `"noise"` for one that publishes a covariance. |\n',
)
edit(
    nb,
    "create-block-03",
    'Every block is registered under a role: `wheel.add(block, role="signal")` for a\n'
    'block that publishes a signal template, `wheel.add(block, role="noise")` for a\n'
    "block that publishes a covariance. Wheel sums the signal blocks' templates into\n",
    'Every block declares its role as a class attribute: `role = "signal"` for a\n'
    'block that publishes a signal template, `role = "noise"` for a block that\n'
    "publishes a covariance; `wheel.add(block)` reads it once. Wheel sums the signal\n"
    "blocks' templates into\n",
)
edit(
    nb,
    "create-block-03",
    "never a template.\n\n",
    "never a template.\n\n"
    "A campaign lives on one WDM grid chosen at construction, its *home grid*\n"
    "(`Wheel(observed, num_frequency_divisions=16)` below: 16 rows × 16 columns for\n"
    "256 samples). Every block is served on the grid it asks for, the observations'\n"
    "own grid by default; Wheel transforms data exactly between grids and\n"
    "translates covariances through a diagonal spectral model (see the README's\n"
    "translation section).\n\n",
)
edit(nb, "create-block-03", "wheel.add(block, role=...)", "wheel.add(block)", count=2)
edit(
    nb,
    "create-block-05",
    "noise_standard_deviation = 0.5\n",
    "noise_standard_deviation = 0.5\n"
    "home_rows = 16  # the campaign's WDM home grid: 16 rows x 16 columns\n",
)
edit(
    nb,
    "create-block-08",
    "class SineAmplitudeBlock:\n    name: str\n",
    "class SineAmplitudeBlock:\n"
    '    role = "signal"  # publishes a template; Wheel reads this at add\n\n'
    "    name: str\n",
)
edit(
    nb,
    "create-block-09",
    'Our signal block omits `covariance`: it is registered with `role="signal"`,\n'
    "and a signal block that returned one would be refused.",
    'Our signal block omits `covariance`: it declares `role = "signal"`, and a\n'
    "signal block that returned one would be refused.",
)
edit(
    nb,
    "create-block-10",
    "covariance on every call and samples nothing, and `check_block` registers each\n"
    'of `noise_blocks` with `role="noise"` before the block under test.',
    'covariance on every call and samples nothing (it declares `role = "noise"`),\n'
    "and `check_block` registers each of `noise_blocks` before the block under test\n"
    "on a scratch Wheel whose home grid the division counts set.",
)
edit(nb, "create-block-11", '    num_cycles=3,\n    role="signal",\n', "    num_cycles=3,\n")
edit(
    nb,
    "create-block-11",
    "    random_seed=5,\n)\n",
    "    random_seed=5,\n    num_frequency_divisions=home_rows,\n)\n",
)
edit(
    nb,
    "known-injection-start",
    '`wheel.add(block, role="signal")` calls `initial_sample` once',
    "`wheel.add(block)` calls `initial_sample` once",
)
edit(
    nb,
    "known-injection-start",
    "`initial_sample` receives the residual on the domain and WDM grid requested at\n"
    "registration, or the observation grid if no domain is requested. Here both are\n"
    "time-domain. Wheel validates, copies, and converts the returned state to its\n"
    "observation grid before adopting it. A noise block's start carries its\n"
    "covariance on the block's own grid; Wheel validates and copies it but never\n"
    "converts it, and the noise sum stays on that grid.",
    "`initial_sample` receives the residual on the domain and WDM grid requested at\n"
    "registration, or the observations' own grid if no domain is requested. Here\n"
    "both blocks work on the time grid the data arrived on. Wheel validates and\n"
    "copies the returned state and transforms a template onto its WDM home grid\n"
    "before adopting it. A noise block's start carries its covariance on the block's\n"
    "own grid; the ledger keeps it there, and the noise sum on the home grid\n"
    "receives its translation (exact for white noise).",
)
edit(
    nb,
    "known-injection-checks",
    "wheel = Wheel(observed, random_seed=23)\n",
    "wheel = Wheel(observed, num_frequency_divisions=home_rows, random_seed=23)\n",
)
edit(
    nb,
    "known-injection-checks",
    'wheel.add(FixedNoiseBlock("noise", covariance), role="noise", data_domain="time")\n'
    'wheel.add(signal_block, role="signal", data_domain="time")\n',
    'wheel.add(FixedNoiseBlock("noise", covariance), data_domain="time")\n'
    'wheel.add(signal_block, data_domain="time")\n',
)
edit(
    nb,
    "known-injection-checks",
    "np.testing.assert_array_equal(\n"
    '    wheel.contribution("signal")["A"], true_amplitude * signal_basis\n'
    ")\n",
    "# Views come back from the home grid through one exact inverse transform.\n"
    "np.testing.assert_allclose(\n"
    '    wheel.contribution("signal")["A"], true_amplitude * signal_basis, atol=1e-12\n'
    ")\n",
)
edit(
    nb,
    "known-injection-checks",
    "np.testing.assert_array_equal(wheel.covariance.A, covariance.A)\n",
    "# wheel.covariance is on the home grid: a time variance of sigma^2 per sample is\n"
    "# sigma^2 / sample_rate_hz per WDM cell; the ledger keeps the publication as is.\n"
    "np.testing.assert_allclose(\n"
    "    wheel.covariance.A, noise_standard_deviation**2 / sample_rate_hz, rtol=1e-12\n"
    ")\n"
    'np.testing.assert_array_equal(wheel.ledger["noise"].covariance.A, covariance.A)\n',
)
edit(
    nb,
    "create-block-15",
    "np.testing.assert_allclose(\n"
    '    wheel.working_residual.channel_data["A"],\n'
    '    observed.channel_data["A"] - last_result.template["A"],\n'
    ")\n",
    "# The ledger template is on the home grid; the contribution view is on the data's.\n"
    "np.testing.assert_allclose(\n"
    '    wheel.working_residual.channel_data["A"],\n'
    '    observed.channel_data["A"] - wheel.contribution("signal")["A"],\n'
    ")\n",
)
edit(
    nb,
    "create-block-18",
    "continued = Wheel(observed, random_seed=31)\n"
    "uninterrupted = Wheel(observed, random_seed=31)\n",
    "continued = Wheel(observed, num_frequency_divisions=home_rows, random_seed=31)\n"
    "uninterrupted = Wheel(observed, num_frequency_divisions=home_rows, random_seed=31)\n",
)
edit(
    nb,
    "create-block-18",
    '    campaign.add(FixedNoiseBlock("noise", covariance), role="noise", data_domain="time")\n'
    '    campaign.add(block, role="signal", data_domain="time")\n',
    '    campaign.add(FixedNoiseBlock("noise", covariance), data_domain="time")\n'
    '    campaign.add(block, data_domain="time")\n',
)
edit(
    nb,
    "create-block-22",
    "There is no separate NoiseBlock base class. A noise block is registered with\n"
    '`role="noise"` and returns',
    "There is no separate NoiseBlock base class. A noise block declares\n"
    '`role = "noise"` and returns',
)
edit(
    nb,
    "create-block-23",
    'class WhiteNoiseVarianceBlock:\n    name: str = "noise"\n',
    "class WhiteNoiseVarianceBlock:\n"
    '    role = "noise"  # publishes a covariance; Wheel reads this at add\n\n'
    '    name: str = "noise"\n',
)
edit(
    nb,
    "create-block-24",
    "joint_campaign = Wheel(observed, random_seed=9)\n",
    "joint_campaign = Wheel(observed, num_frequency_divisions=home_rows, random_seed=9)\n",
)
edit(
    nb,
    "create-block-24",
    "    WhiteNoiseVarianceBlock(start_variance=noise_standard_deviation**2),\n"
    '    role="noise",\n',
    "    WhiteNoiseVarianceBlock(start_variance=noise_standard_deviation**2),\n",
)
edit(nb, "create-block-24", '    ),\n    role="signal",\n', "    ),\n")
edit(
    nb,
    "create-block-24",
    "np.testing.assert_allclose(\n"
    "    joint_campaign.covariance.A,\n"
    '    noise_result.parameters["variance"],\n'
    ")\n",
    "# On the home grid, the sampled time variance is variance / sample_rate_hz per cell.\n"
    "np.testing.assert_allclose(\n"
    "    joint_campaign.covariance.A,\n"
    '    noise_result.parameters["variance"] / sample_rate_hz,\n'
    ")\n",
)
edit(
    nb,
    "create-block-25",
    "- Decide whether the block is a signal block or a noise block and register it\n"
    "  with that `role`; a signal block never publishes a covariance and a noise\n"
    "  block publishes one on every call.",
    "- Decide whether the block is a signal block or a noise block and declare it\n"
    '  with `role = "signal"` or `role = "noise"`; a signal block never publishes a\n'
    "  covariance and a noise block publishes one on every call.",
)
edit(
    nb,
    "create-block-25",
    "- Run `check_block` on a separate instance with its `role=` (and `noise_blocks=`\n"
    "  for any covariance it needs), then",
    "- Run `check_block` on a separate instance (with `noise_blocks=` for any\n"
    "  covariance it needs and the division counts of a home grid), then",
)
save("examples/create_block.ipynb", nb)

# ---------------------------------------------------------- domain_translation
nb = load("examples/domain_translation.ipynb")
edit(
    nb,
    "translation-05",
    "and is `None` for time or frequency data. Configure divisions through `transform`\n"
    'or `Wheel.add(block, role=..., data_domain="wdm", ...)`; do not pass a grid to\n'
    "`L1Data`.",
    "and is `None` for time or frequency data. Configure divisions through\n"
    "`transform`, through `Wheel(observed, num_frequency_divisions=...)` for the\n"
    'campaign\'s home grid, or through `Wheel.add(block, data_domain="wdm", ...)` for\n'
    "a block's grid; do not pass a grid to `L1Data`.",
)
edit(
    nb,
    "translation-16",
    'They are registered with `role="signal"` because they publish templates; a\n'
    'block registered with `role="noise"` would publish a covariance into the\n'
    "campaign's noise sum instead.",
    'They declare `role = "signal"` because they publish templates; a block\n'
    'declaring `role = "noise"` would publish a covariance into the campaign\'s noise\n'
    "sum instead.",
)
edit(
    nb,
    "translation-17",
    "class DomainProbe:\n    name: str\n",
    'class DomainProbe:\n    role = "signal"\n\n    name: str\n',
)
edit(
    nb,
    "translation-18",
    "wheel = Wheel(observed, random_seed=3)\n",
    "# The campaign's WDM home grid: 8 rows x 16 columns for the 128 samples.\n"
    "wheel = Wheel(observed, num_frequency_divisions=8, random_seed=3)\n",
)
edit(
    nb,
    "translation-18",
    '    wheel.add(block, role="signal", data_domain=domain, **options)\n',
    "    wheel.add(block, data_domain=domain, **options)\n",
)
edit(
    nb,
    "translation-18",
    '    assert result.template["A"].shape == observed.channel_data["A"].shape\n',
    '    assert result.template["A"].shape == wheel.home_grid.channel_shape\n',
)
edit(
    nb,
    "translation-18",
    "for name in channel_names:\n"
    "    np.testing.assert_array_equal(\n"
    "        wheel.working_residual.channel_data[name], observed.channel_data[name]\n"
    "    )\n"
    'print("Canonical domain:", wheel.working_residual.data_domain)\n',
    "for name in channel_names:\n"
    "    np.testing.assert_allclose(\n"
    "        wheel.working_residual.channel_data[name],\n"
    "        observed.channel_data[name],\n"
    "        atol=1e-12,\n"
    "    )\n"
    'print("Home grid:", wheel.home_grid)\n'
    'print("Residual view:", wheel.working_residual.data_domain)\n',
)
edit(
    nb,
    "translation-19",
    "Wheel stores accepted estimates on the original observation grid. Before each\n"
    "call, it translates the conditional residual and the block's previous signal\n"
    "into the registered representation. Returned estimates are converted back\n"
    "before adoption. Parameters and sampler state retain their block-local meaning.\n"
    "\n"
    "This campaign has no noise block. The noise sum lives on the grid of the first\n"
    "noise publication, and only blocks registered on that grid receive it; a noise\n"
    "block on another grid is refused at `add`, a signal block on another grid at\n"
    "`add` or `run`. Every noise block returns its complete covariance on every\n"
    "call. Omitting `data_domain` in `Wheel.add`\n"
    "preserves the existing input representations rather than selecting a new one.",
    "Wheel stores accepted templates on its WDM home grid. Before each call, it\n"
    "transforms the conditional residual and the block's previous signal onto the\n"
    "registered grid; returned templates are transformed back onto the home grid\n"
    "before adoption, and the public views (`working_residual`, `residual()`,\n"
    "`contribution()`) come back on the observations' own grid through one exact\n"
    "inverse transform. Parameters and sampler state retain their block-local\n"
    "meaning.\n"
    "\n"
    "This campaign has no noise block. A noise block publishes its covariance on the\n"
    "grid it registered on; the ledger keeps it there, the noise sum on the home grid\n"
    "receives its translation, and every block receives the sum translated onto its\n"
    "own grid (white noise translates exactly; the README's translation table lists\n"
    "the smoothings for anything else). Omitting `data_domain` in `Wheel.add` serves\n"
    'the block on the observations\' own grid; `data_domain="wdm"` without division\n'
    "counts is the home grid itself.",
)
save("examples/domain_translation.ipynb", nb)

# ---------------------------------------------------------------- gb_block_eryn
nb = load("examples/gb_block_eryn.ipynb")
edit(
    nb,
    "ff30cc08",
    "N_time = 2**19  # time-domain samples -> pins Tobs, df\n",
    "N_time = 2**19  # time-domain samples -> pins Tobs, df\n"
    "HOME_ROWS = 512  # the Wheel's WDM home grid: 512 rows x 1024 columns\n",
)
edit(
    nb,
    "7261596e",
    "wheel = Wheel(observed, random_seed=0)  # enchilada\n"
    "# The fixed PSD is a noise block: it publishes the covariance into the noise sum\n"
    "# on the frequency grid, and the GB block registered after it receives it there.\n"
    "# Register it first: GBBlock.initial_sample needs the covariance.\n"
    'wheel.add(FixedNoiseBlock("noise", covariance), role="noise", data_domain="frequency")\n'
    "# initial_sample on the GB grid\n"
    'wheel.add(block, role="signal", data_domain="frequency")\n',
    "wheel = Wheel(observed, num_frequency_divisions=HOME_ROWS, random_seed=0)  # enchilada\n"
    "# The fixed PSD is a noise block: it publishes the covariance on the frequency\n"
    "# grid; the ledger keeps it there, the noise sum on the WDM home grid receives\n"
    "# its translation, and the GB block registered after it receives the sum\n"
    "# translated back onto the frequency grid (band-smoothed for a coloured PSD).\n"
    "# Register it first: GBBlock.initial_sample needs the covariance.\n"
    'wheel.add(FixedNoiseBlock("noise", covariance), data_domain="frequency")\n'
    "# initial_sample on the GB grid\n"
    'wheel.add(block, data_domain="frequency")\n',
)
edit(
    nb,
    "ab8d6832",
    'replace the `FixedNoiseBlock` with a block registered with `role="noise"` on the frequency grid, before the GB block,',
    'replace the `FixedNoiseBlock` with a block declaring `role = "noise"`, registered on the frequency grid before the GB block,',
)
save("examples/gb_block_eryn.ipynb", nb)

# ------------------------------------------------- demo (raw text substitution)
demo = pathlib.Path("examples/demo.ipynb")
raw = demo.read_text()


def sub(old, new, count=1):
    global raw
    found = raw.count(old)
    assert found == count, f"demo.ipynb: expected {count} of {old!r}, found {found}"
    raw = raw.replace(old, new)


sub(
    '    "wheel = Wheel(observed, random_seed=0)\\n",\n',
    '    "# The campaign lives on a WDM home grid: 32 rows x 32 columns for 1024 samples.\\n",\n'
    '    "wheel = Wheel(observed, num_frequency_divisions=32, random_seed=0)\\n",\n',
)
sub('wheel.add(ucb, role=\\"signal\\")', "wheel.add(ucb)")
sub('wheel.add(mbhb, role=\\"signal\\")', "wheel.add(mbhb)")
sub("by its role, a `template`", "by its declared `role`, a `template`")
sub(
    'num_cycles=2, role=\\"signal\\", random_seed=0)',
    "num_cycles=2, random_seed=0, num_frequency_divisions=8)",
)
sub("provide `name` and implement", "provide `name` and `role`, and implement")
sub(
    "`role` names the block's role, `noise_blocks` supplies any covariance it needs,",
    "the block's `role` attribute says whether it publishes a template or a covariance, "
    "`noise_blocks` supplies any covariance it needs, `num_frequency_divisions` sets "
    "the scratch campaign's home grid,",
)
demo.write_text(raw)
print("notebooks edited")
```

Expected output: `notebooks edited`. Then `git diff --stat examples/` shows the four notebooks changed and nothing else.

- [ ] **Step 3: Migrate `tests/test_block_tutorial.py`**

In `test_tutorial_noise_block_refuses_another_noise_block` (lines 147–157) replace

```python
    second = Wheel(data, random_seed=0)
    second.add(FixedNoiseBlock("floor", floor), role="noise")
    with pytest.raises(ValueError, match="only noise block"):
        second.add(WhiteNoiseVarianceBlock(), role="noise")

    # Registered first, it starts alone; the floor registered after it reaches
    # it at its first sample.
    first = Wheel(data, random_seed=0)
    first.add(WhiteNoiseVarianceBlock(), role="noise")
    first.add(FixedNoiseBlock("floor", floor), role="noise")
```

with

```python
    second = Wheel(data, num_frequency_divisions=8, random_seed=0)
    second.add(FixedNoiseBlock("floor", floor))
    with pytest.raises(ValueError, match="only noise block"):
        second.add(WhiteNoiseVarianceBlock())

    # Registered first, it starts alone; the floor registered after it reaches
    # it at its first sample (translated onto its time grid, exact for white noise).
    first = Wheel(data, num_frequency_divisions=8, random_seed=0)
    first.add(WhiteNoiseVarianceBlock())
    first.add(FixedNoiseBlock("floor", floor))
```

(The 64-sample record carries an 8 × 8 home grid; `WhiteNoiseVarianceBlock.role` comes from the notebook cell edited in Step 2.) Then `grep -n "role=" tests/test_block_tutorial.py` must print nothing.

- [ ] **Step 4: Run the Verify commands and the translation-notebook smoke run**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_block_tutorial.py "tests/test_examples.py::test_notebook_code_cells_execute" "tests/test_examples.py::test_all_notebooks_contain_valid_python"
.venv/bin/python -m ruff check examples/create_block.ipynb examples/domain_translation.ipynb examples/demo.ipynb examples/gb_block_eryn.ipynb tests/test_block_tutorial.py
.venv/bin/python -m ruff format --check examples/create_block.ipynb examples/domain_translation.ipynb examples/demo.ipynb examples/gb_block_eryn.ipynb tests/test_block_tutorial.py
grep -c "role=" examples/create_block.ipynb examples/domain_translation.ipynb examples/demo.ipynb examples/gb_block_eryn.ipynb tests/test_block_tutorial.py
cd examples && MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl ../.venv/bin/python - <<'PY'
import json, subprocess, sys
nb = json.load(open("domain_translation.ipynb"))
code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
subprocess.run([sys.executable, "-c", code], check=True)
PY
cd ..
```

Expected: all tests pass; `All checks passed!`; `5 files already formatted`; five `:0` counts; the smoke run prints the three `..._block receives ...` lines with `ledger signal shape (8, 16)`, then `Home grid: Grid(data_domain='wdm', num_time_samples=128, sample_rate_hz=0.5, wdm_grid=WDMGrid(num_frequency_divisions=8, num_time_divisions=16))` and `Residual view: time`, and exits 0. If `ruff format --check` flags a notebook, shorten or reflow the offending code line inside the cell (never run `ruff format` on a notebook) and rerun.

- [ ] **Step 5: Commit**

```bash
git branch --show-current   # wdm-home
git add examples/create_block.ipynb examples/domain_translation.ipynb examples/demo.ipynb examples/gb_block_eryn.ipynb tests/test_block_tutorial.py
git commit -m "Notebooks: declared roles and the WDM home grid in the tutorials" -m "Blocks declare role, campaigns name their home grid, the prose describes the native ledger publication and the translated noise sum instead of the one-grid rule; view assertions allow rounding." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

### Task 7: Forge

**Goal:** The forge workbench (`forge/block.py`, its README and the two notebooks) and `tests/test_forge.py` use the declared role and a WDM home grid whose row count comes from `FORGE_ROWS` (default 64, which fits the default day at factor 10 from a 4 Hz file; the tests set 10 for their 300-sample window), and the prose describes blocks on different grids sharing one campaign instead of the one-grid rule.

**Files:**
- Modify: `forge/block.py` (`ForgeBlock` gains `role = "signal"`; the `_check_residual` message loses `role='signal'`)
- Modify: `forge/README.md` (step 3 of the walkthrough, lines 23–27; the "move the block to another grid" paragraph, lines 76–86)
- Modify: `forge/block.ipynb` (cells `forge-block-01`, `forge-block-07`, `forge-block-08`)
- Modify: `forge/wheel.ipynb` (cells `forge-wheel-01`, `forge-wheel-02`, `forge-wheel-05`)
- Modify: `tests/test_forge.py` (constants, the `simulated_file` fixture, `test_forge_block_is_a_block_with_validated_settings`, the `check_block` call at lines 507–514, the campaign at lines 580–582)

**Acceptance Criteria:**
- [ ] `grep -c "role=" forge/block.py forge/README.md forge/block.ipynb forge/wheel.ipynb tests/test_forge.py` prints `0` for every file; `ForgeBlock.role == "signal"` and `tests/test_forge.py` asserts it.
- [ ] Both notebooks read `NUM_FREQUENCY_DIVISIONS = int(os.environ.get("FORGE_ROWS", 64))`, pass it to `Wheel(...)` / `check_block(...)`, and pass division counts to `transform` only when `DOMAIN == "wdm"` (`transform` refuses them for other targets).
- [ ] `tests/test_forge.py` sets `FORGE_ROWS=10` in the `simulated_file` fixture (300 decimated samples = 10 rows × 30 columns) and names `num_frequency_divisions=HOME_ROWS` in its own `check_block` and `Wheel` calls.
- [ ] `forge/README.md`, `forge-block-07`, `forge-wheel-02` and `forge-wheel-05` no longer say that the covariance "lives on one grid", "does not translate between grids" or that every block registers on the covariance's grid; they say the ledger keeps a publication natively, the home-grid sum receives its translation, and each block receives the sum on its own grid.
- [ ] Both notebooks round-trip through `json.load`/`json.dump(indent=1, ensure_ascii=False)` byte for byte before editing; every code line stays ≤ 88 columns.
- [ ] `tests/test_forge.py` passes (its two notebook-run tests included, in both matplotlib modes), ruff check and format clean on `forge` and `tests/test_forge.py`.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_forge.py` → all passed, 0 failed (the real-file test skips without `ENCHILADA_MOJITO_L1`); `.venv/bin/python -m ruff check forge tests/test_forge.py` → `All checks passed!`; `.venv/bin/python -m ruff format --check forge tests/test_forge.py` → all files already formatted.

**Steps:**

- [ ] **Step 1: Confirm the branch and see the red state**

```bash
git branch --show-current          # wdm-home, or stop
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_forge.py 2>&1 | tail -3
```

Expected: failures with `TypeError: ... declares no role` / `unexpected keyword argument 'role'` and the division-count `ValueError` in the notebook runs.

- [ ] **Step 2: Save and run the forge edit script**

Save this as `$TMPDIR/forge7.py` and run it from the repository root with `.venv/bin/python "$TMPDIR/forge7.py"`. Every old text must occur exactly once in its file or cell; if an assertion fails, make that substitution by hand and delete the call before rerunning.

```python
"""Task 7: the forge on a WDM home grid with a declared role."""

import json
import pathlib


def edit_file(path, old, new, count=1):
    p = pathlib.Path(path)
    text = p.read_text()
    found = text.count(old)
    assert found == count, f"{path}: expected {count} of {old!r}, found {found}"
    p.write_text(text.replace(old, new))


def load(path):
    raw = pathlib.Path(path).read_text()
    nb = json.loads(raw)
    dumped = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    assert dumped == raw, f"{path} does not round-trip through json; edit it as text"
    return nb


def save(path, nb):
    pathlib.Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def edit(nb, cell_id, old, new, count=1):
    (cell,) = [c for c in nb["cells"] if c.get("id") == cell_id]
    source = "".join(cell["source"])
    found = source.count(old)
    assert found == count, f"{cell_id}: expected {count} of {old!r}, found {found}"
    cell["source"] = source.replace(old, new).splitlines(keepends=True)


# ------------------------------------------------------------------ block.py
edit_file(
    "forge/block.py",
    '    draw.\n    """\n\n    name: str = "forge"\n',
    '    draw.\n    """\n\n'
    '    role = "signal"  # publishes a template; Wheel reads this at add\n\n'
    '    name: str = "forge"\n',
)
edit_file(
    "forge/block.py",
    "\"Wheel.add(..., role='signal', data_domain='frequency') or pass \"",
    "\"Wheel.add(..., data_domain='frequency') or pass \"",
)

# ----------------------------------------------------------------- README.md
edit_file(
    "forge/README.md",
    "3. **Run `wheel.ipynb`.** A `Wheel` on the same window, the covariance\n"
    '   registered as a `FixedNoiseBlock` (`role="noise"`), the block registered\n'
    '   on its grid with `role="signal"`, a chain collected through\n'
    "   `on_cycle_complete`, the posterior against a truth you type in, and a\n"
    "   trace plot.\n",
    "3. **Run `wheel.ipynb`.** A `Wheel` on the same window with a WDM home grid of\n"
    "   `FORGE_ROWS` rows (default 64), the covariance registered as a\n"
    '   `FixedNoiseBlock` (a block with `role = "noise"`), `ForgeBlock` (which\n'
    '   declares `role = "signal"`) registered on its grid, a chain collected\n'
    "   through `on_cycle_complete`, the posterior against a truth you type in, and\n"
    "   a trace plot.\n",
)
edit_file(
    "forge/README.md",
    "To move the block to another grid, change `DOMAIN` in both notebooks (for\n"
    "WDM also `NUM_FREQUENCY_DIVISIONS`, the row count: it and the decimated\n"
    "window length divided by it must both be even integers, so choose\n"
    '`FORGE_DURATION_S` to fit, or load with `load_window(..., domain="wdm",\n'
    "num_frequency_divisions=..., num_time_divisions=...)`, which derives the\n"
    "duration), render `_template` on that grid, update `_check_residual`, and\n"
    "rewrite the likelihood in `sample` (`inner_product` is the frequency\n"
    "grid's). A signal block receives the sum of the noise blocks' covariances,\n"
    "which lives on one grid, so every block registers on that grid (a\n"
    "covariance does not translate between grids); another grid means moving\n"
    "`DOMAIN`, and with it the covariance, for the whole campaign.\n",
    "To move the block to another grid, change `DOMAIN` in both notebooks, render\n"
    "`_template` on that grid, update `_check_residual`, and rewrite the likelihood\n"
    "in `sample` (`inner_product` is the frequency grid's). The campaign itself\n"
    "lives on a WDM home grid of `NUM_FREQUENCY_DIVISIONS` rows (`FORGE_ROWS`,\n"
    "default 64): that count and the decimated window length divided by it must\n"
    "both be even integers, so choose `FORGE_DURATION_S` to fit, or load with\n"
    '`load_window(..., domain="wdm", num_frequency_divisions=...,\n'
    'num_time_divisions=...)`, which derives the duration. `DOMAIN = "wdm"` serves\n'
    "the block on the home grid itself. A block on any other grid is served by\n"
    "exact data transforms and receives the noise sum translated onto its grid\n"
    "through the diagonal spectral-model rule (exact for the white stand-in PSD,\n"
    "band-smoothed for a coloured one; see the README's translation table), so\n"
    "blocks on different grids can share one campaign.\n",
)

# ------------------------------------------------------------ the notebooks
SETTINGS_OLD = (
    'DOMAIN = "frequency"  # the grid ForgeBlock renders on; see _check_residual\n'
    '# DOMAIN = "wdm" needs the row count: it and the decimated window length divided\n'
    "# by it must both be even integers (64 rows for the default day at factor 10\n"
    "# from a 4 Hz file).\n"
    "NUM_FREQUENCY_DIVISIONS = None\n"
)
SETTINGS_NEW = (
    'DOMAIN = "frequency"  # the grid ForgeBlock renders on; see _check_residual\n'
    "# The campaign's WDM home grid: this row count and the decimated window length\n"
    "# divided by it must both be even integers (64 rows for the default day at\n"
    '# factor 10 from a 4 Hz file). DOMAIN = "wdm" serves the block on this grid.\n'
    'NUM_FREQUENCY_DIVISIONS = int(os.environ.get("FORGE_ROWS", 64))\n'
)
TRANSFORM_OLD = (
    "residual = transform(window, DOMAIN, num_frequency_divisions=NUM_FREQUENCY_DIVISIONS)\n"
)
TRANSFORM_NEW = (
    'HOME = {"num_frequency_divisions": NUM_FREQUENCY_DIVISIONS}\n'
    'residual = transform(window, DOMAIN, **(HOME if DOMAIN == "wdm" else {}))\n'
)

nb = load("forge/block.ipynb")
edit(nb, "forge-block-01", SETTINGS_OLD, SETTINGS_NEW)
edit(nb, "forge-block-01", TRANSFORM_OLD, TRANSFORM_NEW)
edit(
    nb,
    "forge-block-07",
    "`check_block` builds a scratch Wheel, registers the PSD's covariance there as\n"
    "a noise block (`noise_blocks`, here one `FixedNoiseBlock`), then registers the\n"
    "block it is given as a signal block, calling its `initial_sample`, and runs\n",
    "`check_block` builds a scratch Wheel on a WDM home grid of\n"
    "`NUM_FREQUENCY_DIVISIONS` rows, registers the PSD's covariance there as a noise\n"
    "block (`noise_blocks`, here one `FixedNoiseBlock`, which declares\n"
    '`role = "noise"`), then registers the block it is given (`ForgeBlock.role` is\n'
    '`"signal"`), calling its `initial_sample`, and runs\n',
)
edit(nb, "forge-block-08", '    num_cycles=3,\n    role="signal",\n', "    num_cycles=3,\n")
edit(
    nb,
    "forge-block-08",
    "    random_seed=1,\n)\n",
    "    random_seed=1,\n    num_frequency_divisions=NUM_FREQUENCY_DIVISIONS,\n)\n",
)
save("forge/block.ipynb", nb)

nb = load("forge/wheel.ipynb")
edit(nb, "forge-wheel-01", SETTINGS_OLD, SETTINGS_NEW)
edit(nb, "forge-wheel-01", TRANSFORM_OLD, TRANSFORM_NEW)
edit(
    nb,
    "forge-wheel-02",
    "wheel = Wheel(residual, random_seed=0)\n"
    "# The PSD's covariance enters as a noise block, registered first so the signal\n"
    "# blocks' starts receive it.\n"
    'wheel.add(FixedNoiseBlock("psd", covariance), role="noise")\n'
    'block = ForgeBlock(name="forge", frequency_hz=1e-3)  # EDIT: your block\'s configuration\n'
    'wheel.add(block, role="signal", data_domain=DOMAIN)\n'
    "# A second signal block goes here, on the covariance's grid:\n"
    '# wheel.add(OtherBlock(name="other"), role="signal", data_domain=DOMAIN)\n'
    "# A signal block receives the sum of the noise blocks' covariances, which lives\n"
    "# on one grid, so every block registers on that grid (a covariance does not\n"
    "# translate between grids); to work on another grid, move DOMAIN, and with it\n"
    "# the covariance, for the whole campaign.\n",
    "wheel = Wheel(residual, num_frequency_divisions=NUM_FREQUENCY_DIVISIONS, random_seed=0)\n"
    "# The PSD's covariance enters as a noise block, registered first so the signal\n"
    "# blocks' starts receive it. It publishes on the window's grid; the ledger keeps\n"
    "# it there and the noise sum on the WDM home grid receives its translation.\n"
    'wheel.add(FixedNoiseBlock("psd", covariance))\n'
    'block = ForgeBlock(name="forge", frequency_hz=1e-3)  # EDIT: your block\'s configuration\n'
    "wheel.add(block, data_domain=DOMAIN)\n"
    "# A second signal block goes here, on whatever grid it renders on:\n"
    '# wheel.add(OtherBlock(name="other"), data_domain="time")\n'
    "# Every block receives the noise sum translated onto its own grid (exact for\n"
    "# the white stand-in PSD, band-smoothed for a coloured one).\n",
)
edit(
    nb,
    "forge-wheel-05",
    '- Model the noise with a noise block (`role="noise"`), registered before the\n'
    "  signal blocks whose starts need its covariance; blocks start in registration\n"
    "  order. It publishes on the campaign's one grid and its covariance adds to\n"
    "  the noise sum beside the PSD's `FixedNoiseBlock`; drop that block when yours\n"
    "  models the whole noise.\n",
    '- Model the noise with a noise block (a block declaring `role = "noise"`),\n'
    "  registered before the signal blocks whose starts need its covariance; blocks\n"
    "  start in registration order. It publishes on the grid it registered on, the\n"
    "  noise sum on the home grid receives its translation beside the PSD's\n"
    "  `FixedNoiseBlock`, and every block receives the sum on its own grid; drop\n"
    "  that block when yours models the whole noise.\n",
)
save("forge/wheel.ipynb", nb)

# ------------------------------------------------------------- test_forge.py
edit_file(
    "tests/test_forge.py",
    "NF, NT = 8, 16  # 1024 native samples, longer than the 627-tap filter for factor 8\n",
    "NF, NT = 8, 16  # 1024 native samples, longer than the 627-tap filter for factor 8\n"
    "HOME_ROWS = 10  # FORGE_ROWS for the tests: 300 samples = 10 rows x 30 columns\n",
)
edit_file(
    "tests/test_forge.py",
    '    monkeypatch.setenv("FORGE_FACTOR", str(FACTOR))\n',
    '    monkeypatch.setenv("FORGE_FACTOR", str(FACTOR))\n'
    '    monkeypatch.setenv("FORGE_ROWS", str(HOME_ROWS))\n',
)
edit_file(
    "tests/test_forge.py",
    '    assert block.name == "forge"\n',
    '    assert block.name == "forge"\n    assert block.role == "signal"\n',
)
edit_file(
    "tests/test_forge.py",
    '        num_cycles=3,\n        role="signal",\n',
    "        num_cycles=3,\n",
)
edit_file(
    "tests/test_forge.py",
    "        random_seed=1,\n    )\n    assert block.state.num_sample_calls == 3\n",
    "        random_seed=1,\n        num_frequency_divisions=HOME_ROWS,\n    )\n"
    "    assert block.state.num_sample_calls == 3\n",
)
edit_file(
    "tests/test_forge.py",
    "    wheel = Wheel(observed, random_seed=3)\n"
    '    wheel.add(FixedNoiseBlock("noise", noise), role="noise")\n'
    '    wheel.add(block, role="signal", data_domain="frequency")\n',
    "    wheel = Wheel(observed, num_frequency_divisions=HOME_ROWS, random_seed=3)\n"
    '    wheel.add(FixedNoiseBlock("noise", noise))\n'
    '    wheel.add(block, data_domain="frequency")\n',
)
print("forge edited")
```

Expected output: `forge edited`; `git status --short` lists exactly `forge/README.md`, `forge/block.ipynb`, `forge/block.py`, `forge/wheel.ipynb`, `tests/test_forge.py`.

- [ ] **Step 3: Format the Python file, then run the Verify commands and the greps**

```bash
.venv/bin/python -m ruff format tests/test_forge.py forge/block.py
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_forge.py
.venv/bin/python -m ruff check forge tests/test_forge.py
.venv/bin/python -m ruff format --check forge tests/test_forge.py
grep -c "role=" forge/block.py forge/README.md forge/block.ipynb forge/wheel.ipynb tests/test_forge.py
grep -n "one grid\|does not translate" forge/README.md forge/block.ipynb forge/wheel.ipynb
```

Expected: all tests pass (the only skip is the real-file test), `All checks passed!`, every file formatted, five `:0` counts, and the last grep prints nothing. If a notebook line is flagged by `ruff format --check`, reflow it inside the cell by hand (never run `ruff format` on a notebook).

- [ ] **Step 4: Commit**

```bash
git branch --show-current   # wdm-home
git add forge/block.py forge/README.md forge/block.ipynb forge/wheel.ipynb tests/test_forge.py
git commit -m "Forge: a declared role and a FORGE_ROWS home grid" -m "ForgeBlock declares role = 'signal'; both notebooks build the campaign on a WDM home grid of FORGE_ROWS rows (default 64) and pass division counts to transform only for a WDM DOMAIN; the README describes blocks on different grids sharing one campaign through the covariance translation." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

### Task 8: README, CHANGELOG, `docs/wheel.svg`, wording sweep, full gates

**Goal:** The user-facing documents describe the home grid, the covariance translation rule (with its one table of readings and smoothings), the declared role and the new signatures; the one-grid rule and the `role=` keyword are gone from every file in the repository except where a test probes that the keyword is refused or `Covariance.__add__` is described; the full suite with its coverage gate, mypy and ruff pass.

**Files:**
- Modify: `README.md` (intro lines 13–21; the quickstart block, lines 98–100; the block-authoring section, lines 233–258; the noise section, lines 284–285, 311–313, 370–372; the API table, lines 376–387; the `check_block` example, lines 419–431; the translation section, lines 473–513; Known limitations, lines 760–764; Scaling, lines 779–791; Status, lines 795–796)
- Modify: `CHANGELOG.md` (the `[0.4.0]` Added bullets for `Wheel.add(block, *, role=...)` and `check_block`, the Changed bullets for `initial_block_state`, the role keyword, the benchmark and `transform`, the Removed bullets for the covariance overload and `Wheel(initial_covariance=...)`)
- Modify: `docs/wheel.svg` (line 61 gains a "home grid: wdm" label; line 104 `wheel.add(block, role=…)` → `wheel.add(block)`)
- Modify: `src/enchilada/block_state.py:19` (one docstring line)
- Modify: anything the wording sweep in Step 5 finds

**Acceptance Criteria:**
- [ ] The README quickstart (tagged `<!-- runnable -->`) constructs `Wheel(observed_data=observed, num_frequency_divisions=32)` and registers both blocks without `role=`; `tests/test_docs.py` passes (its keyword check reads the real signatures).
- [ ] The README's translation section carries the table of the three spectral-model readings (time, frequency, WDM) and names the three smoothings as the whole approximation, the user's responsibility for it, and the measured 0.20–0.25 s per translation; the Known limitations section states the even-factorisation requirement of the home grid and that translation is diagonal; the API table has the new `Wheel`, `add`, `check_block` and `transform(covariance, ...)` signatures.
- [ ] The CHANGELOG `[0.4.0]` section describes `Block.role`, the home grid constructor, the `transform` covariance overload, the two breaking changes (role keyword gone; divisions at construction) and the benchmark's `--rows`; no bullet still says the role keyword is required, the sum is held on the first publication's grid, or `transform` refuses a `Covariance`.
- [ ] `docs/wheel.svg` reads `wheel.add(block)` and shows "home grid: wdm" under `wheel.py`.
- [ ] The sweep grep in Step 5 reports only the allowed survivors listed there.
- [ ] Full gates: the whole suite passes with coverage ≥ 95 %, `mypy` is clean, `ruff check` and `ruff format --check` are clean on `src tests examples benchmarks forge`.

**Verify:** `MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider --cov=enchilada` → `0 failed`, coverage ≥ 95 % (the skips are the GB stack and the real Mojito file); `.venv/bin/python -m mypy` → `Success: no issues found`; `.venv/bin/python -m ruff check src tests examples benchmarks forge` → `All checks passed!`; `.venv/bin/python -m ruff format --check src tests examples benchmarks forge` → all files already formatted.

**Steps:**

- [ ] **Step 1: Confirm the branch and that Tasks 3–7 are in**

```bash
git branch --show-current          # wdm-home, or stop
git log --oneline -8               # the five migration commits (tests ×2, examples, notebooks, forge) above Task 2's and Task 1's
git status --short                 # nothing staged or modified from this plan
```

- [ ] **Step 2: Save and run the document edit script**

Save this as `$TMPDIR/docs8.py` and run it from the repository root with `.venv/bin/python "$TMPDIR/docs8.py"`. Every old text must occur exactly once in its file; on an assertion failure make that substitution by hand and delete the call before rerunning.

```python
"""Task 8: README, CHANGELOG, wheel.svg and one docstring."""

import pathlib


def edit(path, old, new, count=1):
    p = pathlib.Path(path)
    text = p.read_text()
    found = text.count(old)
    assert found == count, f"{path}: expected {count} of {old!r}, found {found}"
    p.write_text(text.replace(old, new))


README = "README.md"
edit(
    README,
    "parameters. Every block is registered as a signal block or a noise block.\n"
    "Each block owns its sampler continuation state and diagnostics. The Wheel supplies\n",
    "parameters. The campaign lives on one WDM grid chosen at construction, its\n"
    "*home grid*; every block is served on the grid it asks for. Every block declares\n"
    'its role, `"signal"` or `"noise"`, as an attribute.\n'
    "Each block owns its sampler continuation state and diagnostics. The Wheel supplies\n",
)
edit(
    README,
    "wheel = Wheel(observed_data=observed)\n"
    'wheel.add(block_to_register=ucb, role="signal")\n'
    'wheel.add(block_to_register=mbhb, role="signal")\n',
    "# The campaign lives on a WDM home grid: 32 rows x 32 columns for 1024 samples.\n"
    '# EchoBlock declares role = "signal"; Wheel reads the attribute at add.\n'
    "wheel = Wheel(observed_data=observed, num_frequency_divisions=32)\n"
    "wheel.add(block_to_register=ucb)\n"
    "wheel.add(block_to_register=mbhb)\n",
)
edit(
    README,
    'Every block is registered under a role, `role="signal"` or `role="noise"`,\n'
    "and draws or constructs its own start. `Wheel.add` calls the block's\n",
    'Every block declares its role, `role = "signal"` or `role = "noise"`, as a\n'
    "class attribute, and draws or constructs its own start. `Wheel.add` calls the block's\n",
)
edit(
    README,
    'wheel.add(block_to_register=signal_block, role="signal", data_domain="time")\n',
    'wheel.add(block_to_register=signal_block, data_domain="time")\n',
)
edit(
    README,
    "draws from it. The Block protocol requires `name`, `initial_sample` and\n"
    "`sample`. The [Block authoring tutorial](examples/create_block.ipynb) shows a\n"
    "complete example. Return signal arrays on the block's requested domain and\n"
    "WDM grid, or the observation grid if no domain is selected; Wheel converts\n"
    "them to the observation grid for storage. A noise block's start carries its\n"
    "covariance on the block's own grid, like a template, and enters the noise sum\n"
    "there (see\n",
    "draws from it. The Block protocol requires `name`, `role`, `initial_sample`\n"
    "and `sample`. The [Block authoring tutorial](examples/create_block.ipynb) shows\n"
    "a complete example. Return signal arrays on the block's requested domain and\n"
    "WDM grid, or the observations' own grid if no domain is selected; Wheel\n"
    "transforms them onto its home grid for storage. A noise block's start carries\n"
    "its covariance on the block's own grid, like a template; the ledger keeps it\n"
    "there and the noise sum receives its translation (see\n",
)
edit(
    README,
    "wheel = Wheel(observed_data=observed, random_seed=42)\n"
    'wheel.add(FixedNoiseBlock("instrument", noise), role="noise")\n',
    "wheel = Wheel(observed_data=observed, num_frequency_divisions=32, random_seed=42)\n"
    'wheel.add(FixedNoiseBlock("instrument", noise))\n',
)
edit(
    README,
    "Signal blocks and noise blocks implement the same `Block` protocol; the role\n"
    "given at `Wheel.add` says which one a block is. A noise block publishes its\n",
    "Signal blocks and noise blocks implement the same `Block` protocol; the block's\n"
    "`role` attribute says which one it is. A noise block publishes its\n",
)
edit(
    README,
    "`observed.grid == noise.grid`. Wheel validates every covariance a noise block\n"
    "publishes and keeps the noise sum on the grid of the first publication; see\n",
    "`observed.grid == noise.grid`. Wheel validates every covariance a noise block\n"
    "publishes on the block's own grid, keeps it there in the ledger and translates\n"
    "it onto the home grid for the noise sum; see\n",
)
edit(
    README,
    "| Constructor | `Wheel(observed_data, *, random_seed=None)` |\n"
    "| Register a block | `add(block_to_register, *, role, data_domain=None, num_frequency_divisions=None, num_time_divisions=None)` |\n",
    "| Constructor | `Wheel(observed_data, *, num_frequency_divisions=None, num_time_divisions=None, random_seed=None)`; time or frequency observations need one count (the other is derived), WDM observations bring their grid |\n"
    '| Register a block | `add(block_to_register, *, data_domain=None, num_frequency_divisions=None, num_time_divisions=None)`; `None` serves the observations\' own grid, `data_domain="wdm"` without counts the home grid |\n',
)
edit(
    README,
    "| Check a block | `check_block(block_under_test, observed_data, num_cycles=2, *, role, noise_blocks=(), random_seed=0)` |\n"
    '| A known noise model | `FixedNoiseBlock(name, covariance)`, registered with `role="noise"` |\n'
    "| Add two covariances | `noise_a + noise_b` (same grid and channel order) |\n",
    "| Check a block | `check_block(block_under_test, observed_data, num_cycles=2, *, noise_blocks=(), random_seed=0, num_frequency_divisions=None, num_time_divisions=None)` |\n"
    '| A known noise model | `FixedNoiseBlock(name, covariance)`, a block with `role = "noise"` |\n'
    "| Add two covariances | `noise_a + noise_b` (same grid and channel order) |\n"
    "| Translate a covariance | `transform(noise, target_domain, num_frequency_divisions=None, num_time_divisions=None)` (the diagonal spectral-model rule below) |\n",
)
edit(
    README,
    "Inspect `wheel.observed_data`, `wheel.working_residual`, `wheel.covariance`, and\n"
    "`wheel.ledger[name]`; each returns defensive copies of changing data. The orbit is\n",
    "Inspect `wheel.home_grid`, `wheel.observed_data`, `wheel.working_residual`,\n"
    "`wheel.covariance` (on the home grid), and `wheel.ledger[name]`; each returns\n"
    "defensive copies of changing data, and the residual views come back on the\n"
    "observations' own grid through one exact inverse transform. The orbit is\n",
)
edit(
    README,
    "    num_cycles=2,\n"
    '    role="signal",\n'
    '    noise_blocks=(FixedNoiseBlock("noise", toy_noise),),\n'
    "    random_seed=0,\n"
    ")\n",
    "    num_cycles=2,\n"
    '    noise_blocks=(FixedNoiseBlock("noise", toy_noise),),\n'
    "    random_seed=0,\n"
    "    num_frequency_divisions=8,\n"
    ")\n",
)
edit(
    README,
    'This registers each block in `noise_blocks` with `role="noise"`, then the\n'
    "block under test with `role`, which calls its `initial_sample` on the grid of\n",
    "This builds a scratch Wheel on an 8-row home grid, registers each block in\n"
    '`noise_blocks` (each must declare `role = "noise"`), then the block under test,\n'
    "which calls its `initial_sample` on the grid of\n",
)
edit(
    README,
    "Choose division counts\n"
    'through `transform` or `Wheel.add(block, role=..., data_domain="wdm", ...)`,\n'
    "rather than passing `wdm_grid` to `L1Data`.\n",
    "Choose division counts\n"
    "through `transform`, `Wheel(observed, num_frequency_divisions=...)` for the home\n"
    'grid, or `Wheel.add(block, data_domain="wdm", ...)` for a block\'s grid, rather\n'
    "than passing `wdm_grid` to `L1Data`.\n",
)
edit(
    README,
    'wheel.add(frequency_block, role="signal", data_domain="frequency")\n'
    'wheel.add(wdm_block, role="signal", data_domain="wdm", num_frequency_divisions=8)\n',
    'wheel.add(frequency_block, data_domain="frequency")\n'
    'wheel.add(wdm_block, data_domain="wdm", num_frequency_divisions=8)\n',
)
edit(
    README,
    "WDM grid and returns its start there. Wheel converts the conditional residual\n"
    "and previous signal into the selected representation, then converts returned\n"
    "outputs back to the canonical observation grid before atomic adoption. Omitting\n"
    "`data_domain` keeps the default handoff. To convert a standalone state\n",
    "WDM grid and returns its start there. Wheel forms the conditional residual on\n"
    "its home grid, transforms it and the block's previous signal onto the selected\n"
    "grid, and transforms a returned template back onto the home grid before atomic\n"
    "adoption. Omitting `data_domain` serves the block on the grid the observations\n"
    'arrived on; `data_domain="wdm"` without division counts is the home grid itself,\n'
    "the one choice that costs no transform. To convert a standalone state\n",
)
edit(
    README,
    "A covariance does not translate between representations: it carries its grid,\n"
    "`transform` refuses one with `TypeError`, and `noise.copy()` makes an\n"
    "independent copy. A block state's covariance passes through `transform`\n"
    "unchanged. Wheel keeps the noise sum on the grid of the first noise\n"
    "publication and hands it to blocks registered on that grid. A noise block\n"
    "registered on another grid is refused at `add`, before its `initial_sample`\n"
    "runs, because a covariance does not add across grids; a signal block on\n"
    "another grid is refused at `add` or `run`; both refusals are `ValueError`s\n"
    "naming both grids. A covariance a noise block returns on another grid than its\n"
    "own is refused at `add` or `run`. In a campaign that carries covariance,\n"
    "register every block on one grid.\n",
    "A covariance translates between representations through a diagonal spectral\n"
    'model. `transform(noise, "wdm", num_frequency_divisions=8)` reads every stored\n'
    "variance as a piecewise-constant one-sided spectral model S(f, t) and writes\n"
    "the target grid's variances as weighted means of it (`noise.copy()` makes an\n"
    "independent copy; a block state's covariance passes through `transform`\n"
    "unchanged). The readings:\n"
    "\n"
    "| Grid | Stored block | Read as |\n"
    "| --- | --- | --- |\n"
    "| time | σ²(t_j) per sample | S(f, t_j) = 2 dt σ²(t_j), flat in f |\n"
    "| frequency | C_k per bin | S(f_k, t) = 2 C_k / (N dt), flat in t |\n"
    "| WDM | C_mn per cell | S(f, t_n) = 2 C_mn across band m |\n"
    "\n"
    "Into WDM, each cell takes the window-weighted band mean of S at its column\n"
    "(the value `domania.wdm_covariance` computes, here in O(N)); out of WDM, a bin\n"
    "takes the mean over columns and a sample the window-weighted mean over bands;\n"
    "time and frequency translate into each other through the model directly (the\n"
    "record's mean power at every bin, the mean power at every sample). White noise\n"
    "round-trips exactly through every pair. A stationary PSD comes back\n"
    "band-smoothed from a trip through WDM, a time-varying variance column-smoothed,\n"
    "and time ↔ frequency flattens both; those three smoothings are the whole\n"
    "approximation, and whether a diagonal reading of your noise on your grids is\n"
    "good enough is yours to assess: Wheel does not restrict grid choices on its\n"
    "account.\n"
    "\n"
    "Wheel uses the rule for the noise sum. A noise block publishes on the grid it\n"
    "registered on (a covariance on another grid is refused at `add` or `run`,\n"
    "`ValueError` naming both grids); the ledger keeps that publication natively,\n"
    "the noise sum on the home grid receives its translation, and every block\n"
    "receives the sum translated onto its own grid. Measured on 3 channels of 2^20\n"
    "samples, one translation plus the validating construction of its result costs\n"
    "0.20–0.25 s.\n",
)
edit(
    README,
    "- **No covariance delivery across grids yet.** A `Covariance` lives on one\n"
    "  grid and is not translated, so a campaign that carries covariance keeps every\n"
    "  block on the grid of the first noise publication; a noise block on another\n"
    "  grid is refused at `add`, a signal block at `add` or `run`. Delivering a\n"
    "  noise model to blocks on several grids is still open.\n",
    "- **The home grid needs an even factorisation.** `Wheel` stores the campaign on\n"
    "  a WDM grid, so the record length must be `num_frequency_divisions *\n"
    "  num_time_divisions` with both counts even: an odd record, or a length not\n"
    "  divisible by four, cannot carry a campaign; trim or pad before constructing\n"
    "  one.\n"
    "- **Covariance translation is diagonal.** A covariance translated between grids\n"
    "  is read as diagonal on both; the smoothings listed under\n"
    "  [Time, frequency, and WDM translation](#time-frequency-and-wdm-translation)\n"
    "  are the price, and Wheel does not check how good that reading is for your\n"
    "  noise.\n",
)
edit(
    README,
    "blocks, is assembled the same way. Keep unbounded chains in a callback sink,\n"
    "outside `BlockState`. Run [the benchmark](benchmarks/benchmark_wheel.py) for a\n"
    "reproducible workload, cycle timings, and process peak memory usage. Its\n",
    "blocks, is assembled the same way. A block off the home grid pays one data\n"
    "transform each way per cycle (0.08 s per WDM direction, 0.02 s for an rfft, on\n"
    "3 channels of 2^20 samples) and one covariance translation (0.20–0.25 s with the\n"
    "validating construction of the result); a block on the home grid\n"
    '(`data_domain="wdm"` without counts) pays none. Keep unbounded chains in a\n'
    "callback sink, outside `BlockState`. Run\n"
    "[the benchmark](benchmarks/benchmark_wheel.py) for a reproducible workload,\n"
    "cycle timings, and process peak memory usage (`--rows` sets its home grid). Its\n",
)
edit(
    README,
    "0.4.0 — in development, with block-owned samplers and\n"
    "Wheel-owned accepted physical estimates. This is still an alpha; pin a version for\n",
    "0.4.0 — in development, with block-owned samplers, Wheel-owned accepted\n"
    "physical estimates and a WDM home grid. This is still an alpha; pin a version for\n",
)

CHANGELOG = "CHANGELOG.md"
edit(
    CHANGELOG,
    '- `Wheel.add(block, *, role=...)`: every block is registered as `"signal"` or\n'
    "  `\"noise\"`. Wheel sums the noise blocks' covariances the way it sums the\n",
    '- `Block.role`: every block declares `role = "signal"` or `role = "noise"` as\n'
    "  an attribute, read once at `Wheel.add`; a block without one is refused with\n"
    "  `TypeError`, another value with `ValueError`. Wheel sums the noise blocks'\n"
    "  covariances the way it sums the\n",
)
edit(
    CHANGELOG,
    "- `check_block(block_under_test, observed_data, num_cycles=2, *, role,\n"
    "  noise_blocks=(), random_seed=0)`: `role` is required and `noise_blocks` are\n"
    '  registered first with `role="noise"`.\n',
    "- `check_block(block_under_test, observed_data, num_cycles=2, *,\n"
    "  noise_blocks=(), random_seed=0, num_frequency_divisions=None,\n"
    "  num_time_divisions=None)`: `noise_blocks` are registered first (each must\n"
    '  declare `role = "noise"`); the division counts set the scratch Wheel\'s home\n'
    "  grid.\n"
    "- `Wheel(observed_data, *, num_frequency_divisions=None,\n"
    "  num_time_divisions=None, random_seed=None)`: the campaign lives on a WDM\n"
    "  *home grid* chosen at construction (`wheel.home_grid`); time or frequency\n"
    "  observations are transformed onto it once, WDM observations bring their grid.\n"
    "  Templates, the residual and both sums live there; `add(block, *,\n"
    "  data_domain=None, ...)` serves each block on the grid it asks for by exact\n"
    '  data transforms (`None` is the observations\' own grid, `"wdm"` without counts\n'
    "  the home grid), and a returned template is validated on the block's grid and\n"
    "  transformed onto the home grid. `observed_data` is the untouched arrival copy;\n"
    "  `working_residual`, `residual()` and `contribution()` come back on the\n"
    "  observations' own grid through one exact inverse transform.\n"
    "- `transform(covariance, target_domain, *, num_frequency_divisions=None,\n"
    "  num_time_divisions=None)`: translates a diagonal covariance between time,\n"
    "  frequency and WDM grids through a piecewise-constant spectral model (white\n"
    "  noise round-trips exactly; a stationary PSD is band-averaged into WDM with\n"
    "  domania's window, equal to `domania.wdm_covariance`). Wheel uses it for the\n"
    "  noise sum: a noise block's publication is kept natively in the ledger,\n"
    "  translated onto the home grid for the sum, and every block receives the sum\n"
    "  translated onto its own grid.\n",
)
edit(
    CHANGELOG,
    "  start draws from `rng`, and a block registered on another grid than the\n"
    "  current covariance is refused at `add`, before `initial_sample` is called.\n"
    "  Under `check_block`, construct the block with its start.\n",
    "  start draws from `rng`. Under `check_block`, construct the block with its\n"
    "  start.\n",
)
edit(
    CHANGELOG,
    "- **Breaking:** `Wheel.add` and `check_block` require the `role` keyword; the\n"
    "  single covariance owner is gone. Wheel holds the noise sum on the grid of the\n"
    "  first noise publication and hands a deep copy to blocks registered on that\n"
    "  grid. A noise block on another grid is refused at `add`, before its\n"
    '  `initial_sample`, with `ValueError` ("a covariance does not add across\n'
    '  grids"); a signal block on another grid is refused at `add` or `run` as\n'
    "  before; a covariance a noise block returns on another grid than its own is\n"
    "  refused at `add` or `run`. A joint signal-and-noise model becomes two blocks,\n"
    "  one per role.\n",
    "- **Breaking:** the `role` keyword of `Wheel.add` and `check_block` is gone; a\n"
    "  block declares its role. The single covariance owner is gone too: every\n"
    "  noise block adds its covariance to the sum, published on its own grid and\n"
    "  translated onto the home grid, and the sum reaches each block translated\n"
    "  onto its grid; a covariance a noise block returns on another grid than its\n"
    "  own is refused at `add` or `run`. A joint signal-and-noise model becomes two\n"
    "  blocks, one per role.\n"
    "- **Breaking:** `Wheel` stores the campaign on a WDM home grid. Time or\n"
    "  frequency observations need `num_frequency_divisions` or\n"
    "  `num_time_divisions` at construction (`ValueError` otherwise) and the record\n"
    "  length must factor into two even counts; `wheel.covariance` is on the home\n"
    "  grid; ledger templates are on the home grid and noise publications on their\n"
    "  blocks' grids; the residual views equal the arrival-grid values to rounding,\n"
    "  not bit for bit.\n",
)
edit(
    CHANGELOG,
    "- `benchmarks/benchmark_wheel.py` runs one `FixedNoiseBlock` beside the fixed\n"
    '  signals and reports `"noise_blocks": 1`.',
    "- `benchmarks/benchmark_wheel.py` runs one `FixedNoiseBlock` beside the fixed\n"
    '  signals, reports `"noise_blocks": 1` and `"rows"`, and takes `--rows` for\n'
    "  the home grid (default 128).",
)
edit(
    CHANGELOG,
    "- `transform` takes `L1Data` or `BlockState` only; a `BlockState` carries its\n"
    "  covariance through unchanged.\n",
    "- `transform` takes `L1Data`, `BlockState` or `Covariance`; a `BlockState`\n"
    "  carries its covariance through unchanged.\n",
)
edit(
    CHANGELOG,
    "- The covariance overload of `transform` with its `reference_data`, the\n"
    "  covariance translation seam and every `NotImplementedError` in the package.\n",
    "- The covariance translation seam (`transform` of a `Covariance` used to\n"
    "  refuse, and before that took `reference_data`) and every\n"
    "  `NotImplementedError` in the package.\n",
)
edit(
    CHANGELOG,
    "- `Wheel(initial_covariance=...)` and the positional covariance argument: a\n"
    '  known noise model is a `FixedNoiseBlock` registered with `role="noise"`.\n',
    "- `Wheel(initial_covariance=...)` and the positional covariance argument: a\n"
    '  known noise model is a `FixedNoiseBlock` (a block with `role = "noise"`).\n',
)

SVG = "docs/wheel.svg"
edit(
    SVG,
    '<text x="500" y="415" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="11" fill="#6e655d" text-anchor="middle">wheel.py</text>\n',
    '<text x="500" y="415" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="11" fill="#6e655d" text-anchor="middle">wheel.py</text>\n'
    '<text x="500" y="430" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="11" fill="#6e655d" text-anchor="middle">home grid: wdm</text>\n',
)
edit(SVG, "wheel.add(block, role=…)", "wheel.add(block)")

edit(
    "src/enchilada/block_state.py",
    "    Wheel translates them to the observation grid when storing the ledger.\n",
    "    Wheel transforms them onto its home grid when storing the ledger.\n",
)
print("documents edited")
```

Expected output: `documents edited`; `git status --short` lists `CHANGELOG.md`, `README.md`, `docs/wheel.svg`, `src/enchilada/block_state.py`.

- [ ] **Step 3: Check the README's runnable block and the SVG by eye**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_docs.py tests/test_packaging_docs.py
.venv/bin/python -c "import xml.dom.minidom, sys; xml.dom.minidom.parse('docs/wheel.svg'); print('svg parses')"
grep -n "home grid: wdm\|wheel.add(block)" docs/wheel.svg
```

Expected: the doc tests pass (`test_runnable_blocks_actually_run` executes the quickstart with its 32-row home grid; `test_calls_into_enchilada_use_real_keywords` accepts `num_frequency_divisions`), `svg parses`, and the two grep hits at lines 62 and 105.

- [ ] **Step 4: Run the wording sweep and fix what it finds**

```bash
grep -rn "role=\|registered as a signal\|registered as a noise\|does not translate between\|does not add across grids\|on one grid\|first noise publication\|first publication\|canonical grid\|canonical observation grid\|observation grid\|Canonical domain\|covariance's grid" src tests examples forge benchmarks README.md CHANGELOG.md docs/wheel.svg
```

Allowed survivors, and only these:

- `src/enchilada/covariance.py` and `tests/test_covariance.py`: "on one grid" / "a covariance does not add across grids" (`Covariance.__add__` keeps refusing two grids; the header's last refinement).
- `src/enchilada/_sum_tree.py:138` and `tests/test_sum_tree.py:36`: "on one grid" (every leaf of a sum is on the home grid; true).
- `README.md` and `CHANGELOG.md` lines saying `noise_a + noise_b` "adds two covariances on one grid"; the `[0.3.0]` and older CHANGELOG sections (history is not rewritten).
- Tests from Task 2 that probe the removed keyword (`**{"role": "signal"}` or a literal `role=` inside `pytest.raises(TypeError)`), e.g. in `tests/test_noise_sum.py`, `tests/test_testing.py`, `tests/test_typing.py`.
- `examples/gb_block_eryn.ipynb`: "wrapping them in `L1Data` establishes the observation grid" (the data's own grid; true).

Anything else is a miss from Tasks 2–7: fix it in place with the header's wording rules (role attribute; home grid; native ledger publication, translated sum; views to rounding) and add the file to Step 6's `git add`. In particular check `src/enchilada/wheel.py`'s class and method docstrings (Task 2 rewrote them; confirm no "first noise publication" or "observation grid" in the storage sense survives) and `src/enchilada/translation.py:84` (Task 1 rewrote the docstring; confirm it describes the `Covariance` overload).

- [ ] **Step 5: Run the full gates**

```bash
MPLBACKEND=Agg MPLCONFIGDIR=$TMPDIR/mpl .venv/bin/python -m pytest -q -p no:cacheprovider --cov=enchilada 2>&1 | tail -15
.venv/bin/python -m mypy
.venv/bin/python -m ruff check src tests examples benchmarks forge
.venv/bin/python -m ruff format --check src tests examples benchmarks forge
```

Expected: `0 failed` (skips: the GB stack tests and the real Mojito file), `Required test coverage of 95% reached`, `Success: no issues found`, `All checks passed!`, every file already formatted. A failure here is a defect in one of Tasks 1–7; fix it in the file that owns it (the task map says which), rerun, and name the fix in the commit message.

- [ ] **Step 6: Commit**

```bash
git branch --show-current   # wdm-home
git add README.md CHANGELOG.md docs/wheel.svg src/enchilada/block_state.py   # plus any file Step 4 fixed
git commit -m "Docs: the WDM home grid, covariance translation and declared roles" -m "README: home grid, the translation rule with its table of readings and smoothings, declared roles, new signatures, limitations and costs. CHANGELOG: Block.role, the home-grid constructor, the transform overload, the two breaking changes. wheel.svg: wheel.add(block) and the home-grid label. Full suite, coverage, mypy and ruff green." -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P"
```

---

## After the last task (controller)

These steps are the controller's, not an implementer's.

1. **Final review.** Dispatch one reviewer over `git diff c669da8..HEAD` with the spec (`docs/superpowers/specs/2026-11-21-wdm-home-grid-design.md`), this plan's header (user decisions and refinements) and the §5 error table; it checks that every §5 error and every §6 test has a home, that no `role=` keyword or one-grid refusal survives outside the allowed list, and that the README table matches `_spectral.py`'s kernels. Fix findings through implementer subagents, re-review, rerun the Task 8 gates.
2. **Push** (needs the sandbox disabled for the keychain): `git branch --show-current` → `wdm-home`; `git push -u nsgs wdm-home`.
3. **Pull request:** `gh pr create --repo nsgs-analysis-center/enchilada --base main --head wdm-home --title "Wheel: a WDM home grid, covariance translation, block-declared roles" --body-file "$TMPDIR/pr-body.md"`. The body: what changes (home grid at construction; `transform(covariance, ...)` and the spectral-model rule; `Block.role`; the one-grid rule gone), the three user decisions quoted from the header, the refinements a reviewer should know (kernels are enchilada's, domania's window; time weights; one-way reflection; measured costs; `role: str`), the migration note for block authors (add `role = ...`; pass division counts to `Wheel`; views equal to rounding), the red window between Task 2's and Task 8's commits (bisecting across them lands on failing suites by design), the test plan (full suite with coverage, mypy, ruff, the notebook runs, the forge on the simulated file), and the follow-ups below. End with the two attribution lines:

   ```
   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   https://claude.ai/code/session_015KC1Mzehaz1ta7xTeq5t7P
   ```

4. **Follow-ups to list in the PR body** (not done here, by design): an approximation-size helper (how far a given covariance is from its band- or column-smoothed translation); the revalidation cost of a translated sum on every hand-over and caching the translated sum per block grid; the spec's open items from PR #12 (validated-sum cost; `noise_only` test names; `wheel.ROLES` visibility); the GB notebook's coloured PSD now reaches `GBBlock` band-smoothed through the home grid (the user's accepted approximation; a note in the notebook's "Where next" if they want it exact, register the GB block on a WDM home grid of the PSD's resolution).
5. **Memory:** update `enchilada-noise-sum-branch.md`'s successor note and write a `wdm-home` branch memory (branch, commits, decisions, refinements, follow-ups) once the PR exists.
