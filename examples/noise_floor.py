"""Two noise blocks: a known floor plus a sampled excess variance on top of it.

The campaign registers two noise blocks, each declaring role = "noise".
FixedNoiseBlock publishes the known instrument floor on every call;
ExcessNoiseBlock samples an extra white-noise variance with a Metropolis random
walk in log-variance. Wheel hands the excess block the floor as its `covariance`
(the sum over the other noise blocks, translated onto the block's time grid) and
keeps the campaign covariance equal to floor + excess on its WDM home grid,
which a signal block registered on any grid would receive translated onto its
own.

The likelihood is a Gaussian likelihood of time-domain data with independent
channels: each time sample is an independent real Gaussian whose variance is in
per-sample units, and cross-channel terms are not modelled. On Fourier or WDM
observations both blocks must therefore be registered with data_domain="time",
and the floor built on that time grid. One excess variance is shared across all
channels. ExcessNoiseBlock refuses a None floor, so it must be registered after
another noise block. Run with:

    uv run --no-sync python examples/noise_floor.py
"""

from typing import Any

import numpy as np

from enchilada import BlockState, Covariance, FixedNoiseBlock, L1Data, Wheel, transform

TRUTH = {"floor": 0.25, "excess": 0.75}


class ExcessNoiseBlock:
    """An extra white-noise variance above the covariance this block receives.

    The prior on the variance is log-uniform between `low` and `high`. Each
    `sample` call runs `steps_per_call` Metropolis steps with a Gaussian proposal
    of width `step` in log-variance. The likelihood of the time-domain residual
    uses the total covariance `received + own`, formed with `Covariance.__add__`.

    The likelihood is Gaussian over time-domain data with independent channels:
    each time sample is an independent real Gaussian with its variance in
    per-sample units, and cross-channel terms are not modelled (a floor with
    cross-channel terms is refused). One excess variance is shared across all
    channels. On Fourier or WDM observations register the block with
    `data_domain="time"` and build the floor on that time grid. The block
    refuses a `None` floor, so register it after another noise block.
    """

    role = "noise"

    def __init__(
        self,
        name: str,
        low: float = 1e-3,
        high: float = 1e3,
        step: float = 0.3,
        steps_per_call: int = 10,
    ):
        if not np.isfinite([low, high]).all() or not 0 < low < high:
            raise ValueError("the prior needs finite low and high with 0 < low < high")
        if not np.isfinite(step) or step <= 0:
            raise ValueError("step must be finite and positive")
        if (
            isinstance(steps_per_call, bool)
            or not isinstance(steps_per_call, int)
            or steps_per_call <= 0
        ):
            raise ValueError("steps_per_call must be a positive integer")
        self.name = name
        self.low, self.high = low, high
        self.step, self.steps_per_call = step, steps_per_call

    @staticmethod
    def _own(conditional_residual: L1Data, variance: float) -> Covariance:
        if conditional_residual.data_domain != "time":
            raise ValueError("ExcessNoiseBlock requires time-domain data")
        return Covariance.like(
            conditional_residual,
            {
                ch: np.full(conditional_residual.num_time_samples, variance)
                for ch in conditional_residual.channel_names
            },
        )

    def _render(self, conditional_residual: L1Data, variance: float) -> BlockState:
        return BlockState(
            covariance=self._own(conditional_residual, variance),
            parameters={"variance": float(variance)},
        )

    def _log_likelihood(
        self, conditional_residual: L1Data, floor: Covariance, variance: float
    ) -> float:
        total = floor + self._own(conditional_residual, variance)
        value = 0.0
        for ch, residual in conditional_residual.channel_data.items():
            total_variance = total.pair(ch, ch)
            value -= 0.5 * float(
                np.sum(residual**2 / total_variance) + np.sum(np.log(total_variance))
            )
        return value

    @staticmethod
    def _require_floor(covariance: Covariance | None) -> Covariance:
        if covariance is None:
            raise ValueError(
                "ExcessNoiseBlock adds to another noise block's covariance; "
                "register the floor (for instance a FixedNoiseBlock) first"
            )
        channels = covariance.channel_names
        if any(
            np.any(covariance.pair(first, second) != 0)
            for first in channels
            for second in channels
            if first != second
        ):
            raise ValueError("ExcessNoiseBlock requires uncorrelated channels")
        return covariance

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw the excess variance from its log-uniform prior and publish it."""
        self._require_floor(covariance)
        log_variance = rng.uniform(np.log(self.low), np.log(self.high))
        return self._render(conditional_residual, float(np.exp(log_variance)))

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Advance a Metropolis random walk in log-variance against the residual."""
        # _own repeats this check, but a frequency-domain residual with a None
        # floor should report the domain error before the missing floor.
        if conditional_residual.data_domain != "time":
            raise ValueError("ExcessNoiseBlock requires time-domain data")
        floor = self._require_floor(covariance)
        log_variance = float(np.log(current_block_state.parameters["variance"]))
        current = self._log_likelihood(
            conditional_residual, floor, float(np.exp(log_variance))
        )
        log_low, log_high = np.log(self.low), np.log(self.high)
        for _ in range(self.steps_per_call):
            proposal = log_variance + self.step * rng.normal()
            if not log_low <= proposal <= log_high:
                continue  # outside the prior: rejected without a likelihood call
            proposed = self._log_likelihood(
                conditional_residual, floor, float(np.exp(proposal))
            )
            if rng.random() < np.exp(min(0.0, proposed - current)):
                log_variance, current = proposal, proposed
        return self._render(conditional_residual, float(np.exp(log_variance)))


def make_observed(seed: int = 0) -> L1Data:
    """Pure white noise whose variance is the floor plus the excess."""
    rng = np.random.default_rng(seed)
    n = 2048
    sigma = np.sqrt(TRUTH["floor"] + TRUTH["excess"])
    return L1Data(
        {"A": rng.normal(0.0, sigma, n)},
        sample_rate_hz=0.1,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=0.0,
    )


def run_example(
    n_cycles: int = 400, burn_in: int = 100, seed: int = 0
) -> dict[str, Any]:
    """Run the campaign; return the excess posterior summary and the Wheel."""
    if not 0 <= burn_in < n_cycles:
        raise ValueError("burn_in must be nonnegative and smaller than n_cycles")
    observed = make_observed(seed)
    # Seed the campaign from a child of the observation seed so registration and
    # sampling never replay the observation-noise draws.
    campaign_seed = np.random.SeedSequence(seed).spawn(1)[0]
    wheel = Wheel(
        observed,
        num_frequency_divisions=32,  # the WDM home grid: 32 rows x 64 columns
        random_seed=int(campaign_seed.generate_state(1)[0]),
    )
    floor = Covariance.like(
        observed,
        {
            ch: np.full(observed.num_time_samples, TRUTH["floor"])
            for ch in observed.channel_names
        },
    )
    # Both are noise blocks. The floor must go first because the excess block
    # refuses to start without it; Wheel keeps floor + excess as the campaign
    # covariance on its home grid.
    wheel.add(FixedNoiseBlock("floor", floor))
    wheel.add(ExcessNoiseBlock("excess"))
    chain: list[float] = []

    def collect(cycle: int, w: Wheel) -> None:
        chain.append(w.ledger["excess"].parameters["variance"])
        if (cycle + 1) % 100 == 0:
            # The campaign covariance is on the home grid; read it on the time grid.
            total = transform(w.covariance, "time").A[0]
            print(
                f"cycle {cycle + 1:4d}: excess variance draw = {chain[-1]:.4f}  "
                f"(campaign covariance A[0] = {total:.4f})"
            )

    wheel.run(n_cycles, on_cycle_complete=collect)
    samples = np.asarray(chain[burn_in:])
    return {
        "posterior_mean": float(samples.mean()),
        "posterior_std": float(samples.std()),
        "wheel": wheel,
    }


def main() -> None:
    print(f"truth: floor={TRUTH['floor']} (known), excess={TRUTH['excess']}\n")
    result = run_example()
    print()
    print(
        f"excess posterior: {result['posterior_mean']:.4f} +/- "
        f"{result['posterior_std']:.4f}   (truth {TRUTH['excess']})"
    )


if __name__ == "__main__":
    main()
