"""One object-level interface for time, Fourier and WDM representations."""

from copy import deepcopy
from dataclasses import replace
from typing import overload

from enchilada._spectral import translate_blocks
from enchilada._transforms import transform_channels
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid, WDMGrid, resolve_wdm_grid


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


@overload
def transform(
    value: Covariance,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> Covariance: ...


def transform(
    value: L1Data | BlockState | Covariance,
    target_domain: str,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
    reference_data: L1Data | None = None,
) -> L1Data | BlockState | Covariance:
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
    frequency and, between time and WDM, its time weights across columns
    (WDM to frequency takes a plain column mean and WDM to WDM reads the
    source column at each target column centre), so the result is a
    validated Covariance with the same channels. This treats every covariance
    as diagonal in the grid index on every grid: white noise round-trips
    exactly, a coloured PSD comes back band-smoothed from WDM, a time-varying
    variance comes back column-smoothed, and time <-> frequency keeps only the
    mean power. A time variance enters WDM exactly only when it varies slowly
    over a column and, the record being read as periodic, its two ends match;
    sharper variations are smoothed, and only the DC and Nyquist rows stay
    exact. The same grid in and out returns ``value.copy()``.
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


def _data_on_grid(data: L1Data, grid: Grid) -> L1Data:
    """`data` on `grid`: itself when already there, else its translation."""
    if data.grid == grid:
        return data
    return transform(
        data,
        grid.data_domain,
        num_frequency_divisions=(
            None if grid.wdm_grid is None else grid.wdm_grid.num_frequency_divisions
        ),
    )


def _state_on_grid(
    state: BlockState, grid: Grid, *, reference_data: L1Data
) -> BlockState:
    """A block state built on `reference_data`'s grid, moved to `grid`.

    The template moves by one exact transform and the covariance is kept, as
    `transform` does; a state already on `grid` is returned as it is.
    """
    if reference_data.grid == grid:
        return state
    return transform(
        state,
        grid.data_domain,
        reference_data=reference_data,
        num_frequency_divisions=(
            None if grid.wdm_grid is None else grid.wdm_grid.num_frequency_divisions
        ),
    )
