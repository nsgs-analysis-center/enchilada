# Forge

A workbench for building an enchilada `Block` against the inputs a block
really receives: a conditional residual on a chosen grid and a `Covariance`
on that grid, loaded from a simulated Mojito-layout L1 file. Copy this folder
once per block and fill in the skeleton. The tutorial that explains the
contract is [`examples/create_block.ipynb`](../examples/create_block.ipynb);
this folder assumes you have read it.

## The three steps

1. **Fill in `block.py`.** `ForgeBlock` marks five methods `# EDIT`. Three
   are the model: `_template` renders the signal on the block's grid,
   `initial_sample` draws or renders the start, `sample` advances the state.
   Two are checks, `_check_residual` and `_check_covariance`, which say what
   the model needs from its inputs. The placeholder is a sinusoid of one amplitude on the
   frequency grid with a conjugate Gaussian sampler, so everything runs
   before you change a line.
2. **Run `block.ipynb`.** It loads a window of the simulated file, builds the
   covariance, calls `initial_sample` and `sample` by hand with asserts on
   the returned state, and runs `check_block` with the covariance as a
   `FixedNoiseBlock`. Edits to `block.py` take effect cell by cell.
3. **Run `wheel.ipynb`.** A `Wheel` on the same window with a WDM home grid of
   `FORGE_ROWS` rows (default 64), the covariance registered as a
   `FixedNoiseBlock` (a block with `role = "noise"`), `ForgeBlock` (which
   declares `role = "signal"`) registered on its grid, a chain collected
   through `on_cycle_complete`, the posterior against a truth you type in, and
   a trace plot.

## Setup

From the repository root (`numeric-orbits` brings h5py and, through
lisaorbits, matplotlib; `examples` brings JupyterLab):

```sh
uv sync --extra numeric-orbits --extra examples
ENCHILADA_MOJITO_L1=/path/to/simulated_L1.h5 uv run --no-sync jupyter lab forge/block.ipynb
```

In a running kernel, `%env ENCHILADA_MOJITO_L1=/path/to/simulated_L1.h5` in a
cell sets it before the settings cell runs.

The notebooks read three environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `ENCHILADA_MOJITO_L1` | the Mojito-layout L1 file | required |
| `FORGE_DURATION_S` | native seconds to load | `86400` (one day) |
| `FORGE_FACTOR` | decimation factor | `10` (0.4 Hz from a 4 Hz file such as Mojito's) |

`data.py` is shared by both notebooks. `load_window` goes through
`load_mojito_l1` and `decimate` and returns the window on the time,
frequency or WDM grid (for WDM, pass both division counts instead of
`duration_s`; the native duration is derived so the decimated record fits the
grid). `white_psd` and `analytic_covariance` turn a one-sided PSD, any
callable `psd(f_hz)` or one per channel, into a `Covariance` on that grid.

## What to replace and what to keep

Replace the model (`_template`, `initial_sample`, `sample`) and the checks'
expectations (the grid, what the likelihood needs from the covariance). Keep
the shape of the contract: a `BlockState` with a template on the input grid
and no covariance (a signal block's noise model is a separate noise block),
a complete state after a rejection too, draws from the supplied `rng`,
diagnostics on `block.state` rather than in the ledger, and a fresh instance
per campaign. Collect chains outside the block, as `wheel.ipynb` does. In
the notebooks, change the settings cells and the lines marked `# EDIT` (the
block's constructor, `TRUTH`); the asserts are the contract and name no
parameter of the placeholder. Keep `State.num_sample_calls`: `block.ipynb`
asserts on it. To test recovery, inject a signal into the window the way
`test_forge_block_recovers_an_injected_amplitude` does
(`dataclasses.replace(residual, channel_data=...)`) and put the truth in
`TRUTH`. The notebooks load from the file's start, and the loader refuses a
window with a nonzero `tdi_flags` sample; `load_window(start_offset_s=...)`
moves it.

To move the block to another grid, change `DOMAIN` in both notebooks, render
`_template` on that grid, update `_check_residual`, and rewrite the likelihood
in `sample` (`inner_product` is the frequency grid's). The campaign itself
lives on a WDM home grid of `NUM_FREQUENCY_DIVISIONS` rows (`FORGE_ROWS`,
default 64): that count and the decimated window length divided by it must
both be even integers, so choose `FORGE_DURATION_S` to fit, or load with
`load_window(..., domain="wdm", num_frequency_divisions=...,
num_time_divisions=...)`, which derives the duration. `DOMAIN = "wdm"` serves
the block on the home grid itself. A block on any other grid is served by
exact data transforms and receives the noise sum translated onto its grid
through the diagonal spectral-model rule (exact for the white stand-in PSD,
band-smoothed for a coloured one; see the README's translation table), so
blocks on different grids can share one campaign.

## Caveats

- **Time grid, white noise only.** enchilada's time-grid covariance is one
  variance per sample, so `analytic_covariance` accepts only a white PSD
  there and refuses a coloured one. Coloured noise needs the frequency or
  WDM grid.
- **Edge transient.** `decimate` leaves 40 filter-transient samples at each
  end of the record (`enchilada.resample.edge_samples`); nothing is trimmed,
  and on a WDM grid the transient leaks a few columns inward.
- **Anti-alias roll-off.** `decimate` passes the band fully only up to 0.8 of
  the new Nyquist frequency, so the PSD of the native noise overstates the
  cells above that (conservative) and the white time-grid variance by about
  13 %. A WDM cell's variance is half the PSD averaged across its band
  (`domania.wdm_covariance`), so each cell's variance is right across
  notches and steep walls (the diagonal covariance still drops the
  correlations a steep wall puts between neighbouring cells); row 0
  interleaves DC and Nyquist.
- **White stand-in.** The notebooks' `LEVEL` is a white PSD at the loaded
  window's own variance (for white native noise, about 12 % below the
  in-band PSD, for the reason above), a placeholder for the simulation's noise model, and a loud injected
  signal inflates it. On coloured noise the variance-based level is a band
  average dominated by the loudest band, and the diagonal covariance on an
  untapered window ignores leakage on steep spectra, so posterior widths
  mean little until `PSD` is the simulation's model. The GB example's
  `FixedLISANoise().psd` (copy `examples/gb_model.py` beside the notebooks;
  needs `lisaanalysistools`) is a TDI 1.5 sensitivity, while Mojito's
  channels are TDI 2.0: convert it (equal arms: multiply by
  `4 sin²(4π f L / c)`) and check its levels against the simulation's noise
  model before using it as `PSD`.
