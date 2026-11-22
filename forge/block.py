"""ForgeBlock: the skeleton to fill in.

The methods marked `# EDIT` are the model: `state_from_parameters` renders the
signal on the block's grid from one parameter set, `initial_sample` draws or
renders the start, `sample` advances the state; `_check_residual` and
`_check_covariance` say what the model needs from its inputs. The placeholder
is a sinusoid of one amplitude at `frequency_hz`, rendered natively on the
frequency grid, with a Gaussian prior on the amplitude, so `sample` is an exact
conjugate Gaussian draw: the skeleton runs green on noise-only data with no
tuning and every notebook cell shows real numbers. Replace the model; keep the
shape of the contract. The module-level `rfft_weights` and `inner_product`
carry the Whittle normalisation on the frequency grid; a replacement likelihood
reuses them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from enchilada import BlockState, Covariance, L1Data


def rfft_weights(num_time_samples: int) -> np.ndarray:
    """How often each rfft bin counts in a real signal's inner product: once at
    DC and at Nyquist (even `num_time_samples` only), twice elsewhere."""
    weights = np.full(num_time_samples // 2 + 1, 2.0)
    weights[0] = 1.0
    if num_time_samples % 2 == 0:
        weights[-1] = 1.0
    return weights


def inner_product(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    conditional_residual: L1Data,
    covariance: Covariance,
) -> float:
    """The Whittle inner product of two frequency-grid signals, channels
    independent: `sum over channels and bins of w_k Re(conj(first_k) second_k)
    / C_k`, with `C_k` the per-bin variance `covariance.pair(channel, channel)`
    (`(Tobs / 2) S(f_k)` for a one-sided PSD `S`) and `w_k` from `rfft_weights`.
    `-0.5 * inner_product(r, r, ...)` is a residual `r`'s log-likelihood up to
    a constant; the likelihood alone gives an amplitude precision `<h|h>` and
    mean `<h|d> / <h|h>`, and a Gaussian prior adds `1 / prior_std**2` to the
    precision. Cross terms are ignored, not refused: run `_check_covariance`
    first."""
    weights = rfft_weights(conditional_residual.num_time_samples)
    total = 0.0
    for name in conditional_residual.channel_names:
        variance = covariance.pair(name, name).real
        overlap = np.real(np.conj(first[name]) * second[name])
        total += float(np.sum(weights * overlap / variance))
    return total


@dataclass
class ForgeBlock:
    """A sinusoid of one amplitude on the frequency grid; replace the model.

    `prior_std` is the standard deviation of the zero-mean Gaussian prior on
    the amplitude, in the data's units (fractional frequency). The default is
    wide next to the likelihood on a day of Mojito-level data, so the
    posterior is the data's; tighten it when the data barely constrain the
    amplitude. `start_amplitude` renders a chosen start instead of a prior
    draw.
    """

    # publishes a template; Wheel reads this at add. A class attribute, not a
    # dataclass field: an annotation would make it an __init__ parameter.
    role = "signal"

    name: str = "forge"
    frequency_hz: float = 1e-3
    prior_std: float = 1e-17
    start_amplitude: float | None = None

    @dataclass
    class State:
        """Diagnostics owned by the block; never published to the ledger. Keep
        `num_sample_calls`: `block.ipynb` asserts on it."""

        num_sample_calls: int = 0
        last_posterior_mean: float | None = None
        last_posterior_std: float | None = None

    state: State = field(default_factory=State, init=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        if not np.isfinite(self.prior_std) or self.prior_std <= 0:
            raise ValueError("prior_std must be finite and positive")
        if self.start_amplitude is not None and not np.isfinite(self.start_amplitude):
            raise ValueError("start_amplitude must be finite")

    # ---- checks: keep these honest about what the model needs ----------------

    def _check_residual(self, conditional_residual: L1Data) -> None:
        # EDIT: the grid your template is rendered on, and anything else the
        # model assumes about the data (it also runs inside state_from_parameters:
        # check the grid and conventions, not values).
        if conditional_residual.data_domain != "frequency":
            raise ValueError(
                f"{self.name}: needs frequency-grid data (register it with "
                "Wheel.add(..., data_domain='frequency') or pass "
                f"transform(data, 'frequency')); got "
                f"{conditional_residual.data_domain!r} data"
            )
        if self.frequency_hz >= conditional_residual.fny:
            raise ValueError(
                f"{self.name}: frequency_hz = {self.frequency_hz} is not below the "
                f"Nyquist frequency {conditional_residual.fny} Hz"
            )

    def _check_covariance(
        self, conditional_residual: L1Data, covariance: Covariance | None
    ) -> Covariance:
        # EDIT: what your likelihood needs from the noise model.
        if covariance is None:
            raise ValueError(
                f"{self.name}: the likelihood needs a noise covariance; register a "
                "noise block (a FixedNoiseBlock for a known PSD) before this block"
            )
        covariance.check_compatible(conditional_residual)
        channels = conditional_residual.channel_names
        if any(
            np.any(covariance.pair(first, second))
            for first in channels
            for second in channels
            if first != second
        ):
            raise ValueError(
                f"{self.name}: the likelihood assumes independent channels"
            )
        return covariance

    # ---- the model -------------------------------------------------------------

    def _basis(self, conditional_residual: L1Data) -> np.ndarray:
        """Unit-amplitude sinusoid at `frequency_hz` as `dt * rfft`, the same in
        every channel."""
        times_s = (
            conditional_residual.start_time_gps
            + np.arange(conditional_residual.num_time_samples) / conditional_residual.fs
        )
        return conditional_residual.dt * np.fft.rfft(
            np.sin(2 * np.pi * self.frequency_hz * times_s)
        )

    def state_from_parameters(
        self, parameters: Mapping[str, Any], reference_data: L1Data
    ) -> BlockState:
        # EDIT: render your waveform on the block's grid from one parameter set
        # (the placeholder's lives in _basis). initial_sample and sample publish
        # through this method and check_block and L2Data rebuild with it, so it
        # reads only the parameters, the block's fields and the grid: never
        # channel values, an rng or self.state. Parameters read back from a
        # campaign file arrive as numpy scalars.
        self._check_residual(reference_data)
        amplitude = float(parameters["amplitude"])
        basis = self._basis(reference_data)
        return reference_data.block_state(
            {name: amplitude * basis for name in reference_data.channel_names},
            parameters={"amplitude": amplitude},
        )

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        # EDIT: your start, drawn from your prior or rendered from configuration.
        amplitude = self.start_amplitude
        if amplitude is None:
            amplitude = rng.normal(0.0, self.prior_std)
        return self.state_from_parameters(
            {"amplitude": amplitude}, conditional_residual
        )

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        # EDIT: your sampler. Return a complete state after acceptance and
        # rejection alike. This placeholder ignores current_block_state because
        # it draws the exact conjugate posterior of the amplitude; an MCMC step
        # continues from current_block_state.parameters instead, and commits its
        # diagnostics only once the new state is built.
        self._check_residual(conditional_residual)
        noise = self._check_covariance(conditional_residual, covariance)
        basis = self._basis(conditional_residual)
        unit = {name: basis for name in conditional_residual.channel_names}
        precision = 1.0 / self.prior_std**2 + inner_product(
            unit, unit, conditional_residual, noise
        )
        mean = (
            inner_product(
                unit, conditional_residual.channel_data, conditional_residual, noise
            )
            / precision
        )
        std = precision**-0.5
        moved = self.state_from_parameters(
            {"amplitude": rng.normal(mean, std)}, conditional_residual
        )
        self.state.num_sample_calls += 1
        self.state.last_posterior_mean = mean
        self.state.last_posterior_std = std
        return moved
