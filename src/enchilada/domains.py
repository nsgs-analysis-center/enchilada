"""Representation metadata independent of any optional transform backend."""

import numbers
from dataclasses import dataclass

import numpy as np

DOMAINS = ("time", "frequency", "wdm")
"""Valid values for a data domain."""


def _positive_even_integer(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, bool)
        or value <= 0
        or value % 2
    ):
        raise ValueError(f"{name} must be a positive even integer, got {value!r}")
    return int(value)


@dataclass(frozen=True, kw_only=True)
class WDMGrid:
    """The WDM frequency and time divisions of an unchanged observation.

    Both division counts are positive even integers. The coefficient array has
    ``num_frequency_divisions + 1`` rows, including both edge bands. The two
    edge bands contain coefficients only at even time indices; every interior
    coefficient is active. Sampling interval and epoch belong to ``L1Data``.
    """

    num_frequency_divisions: int
    num_time_divisions: int

    def __post_init__(self) -> None:
        for name in ("num_frequency_divisions", "num_time_divisions"):
            object.__setattr__(
                self, name, _positive_even_integer(getattr(self, name), name)
            )

    @property
    def array_shape(self) -> tuple[int, int]:
        """Frequency rows, including both edges, followed by time columns."""
        return self.num_frequency_divisions + 1, self.num_time_divisions

    @property
    def num_time_samples(self) -> int:
        """Number of samples in the underlying real time series."""
        return self.num_frequency_divisions * self.num_time_divisions

    @property
    def active_mask(self) -> np.ndarray:
        """Return an independent mask of the grid's physical coefficients."""
        active = np.ones(self.array_shape, dtype=bool)
        active[[0, -1], 1::2] = False
        return active


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
                f"wdm_grid applies only to data_domain='wdm', got {self.data_domain!r}"
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


def resolve_wdm_grid(
    num_time_samples: int,
    *,
    num_frequency_divisions: int | None = None,
    num_time_divisions: int | None = None,
) -> WDMGrid:
    """Derive a missing division count without padding or resampling the data.

    Supply at least one count. If both are supplied their product must equal
    ``num_time_samples``. Inferred counts obey the same positive-even contract
    as explicit counts.
    """
    if (
        not isinstance(num_time_samples, (int, np.integer))
        or isinstance(num_time_samples, bool)
        or num_time_samples <= 0
    ):
        raise ValueError("num_time_samples must be a positive integer")
    num_time_samples = int(num_time_samples)
    if num_frequency_divisions is None and num_time_divisions is None:
        raise ValueError(
            "supply num_frequency_divisions or num_time_divisions for the WDM grid"
        )
    if num_frequency_divisions is not None:
        num_frequency_divisions = _positive_even_integer(
            num_frequency_divisions, "num_frequency_divisions"
        )
    if num_time_divisions is not None:
        num_time_divisions = _positive_even_integer(
            num_time_divisions, "num_time_divisions"
        )
    if num_frequency_divisions is None:
        assert num_time_divisions is not None
        if num_time_samples % num_time_divisions:
            raise ValueError("num_time_divisions must divide num_time_samples exactly")
        num_frequency_divisions = num_time_samples // num_time_divisions
    if num_time_divisions is None:
        if num_time_samples % num_frequency_divisions:
            raise ValueError(
                "num_frequency_divisions must divide num_time_samples exactly"
            )
        num_time_divisions = num_time_samples // num_frequency_divisions
    grid = WDMGrid(
        num_frequency_divisions=num_frequency_divisions,
        num_time_divisions=num_time_divisions,
    )
    if grid.num_time_samples != num_time_samples:
        raise ValueError("WDM division product must equal num_time_samples")
    return grid


def check_wdm_coefficients(
    channel_data: dict[str, np.ndarray],
    wdm_grid: WDMGrid,
    context_label: str,
) -> None:
    """Require real coefficient grids with exactly zero inactive entries.

    The WDM inverse discards inactive coefficients. Reject them here rather
    than silently losing caller data. The translator zeros its own numerical
    inactive tails before constructing an ``L1Data``. Active-value finiteness
    remains the responsibility of the Wheel and transform boundaries.
    """
    inactive = ~wdm_grid.active_mask
    for channel_name, coefficients in channel_data.items():
        label = f"{context_label}[{channel_name!r}]"
        if not isinstance(coefficients, np.ndarray):
            raise TypeError(f"{label} must be a numpy array")
        if coefficients.shape != wdm_grid.array_shape:
            raise ValueError(
                f"{label} must have WDM shape {wdm_grid.array_shape}, "
                f"got {coefficients.shape}"
            )
        if not np.issubdtype(coefficients.dtype, np.floating):
            raise TypeError(f"{label} WDM coefficients must be real floating values")
        if np.any(coefficients[inactive] != 0):
            raise ValueError(f"{label} inactive WDM coefficients must be exactly zero")
