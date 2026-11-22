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
