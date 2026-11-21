"""Two conjugate Gibbs source blocks plus sampled white noise.

Blocks retain fixed model configuration. Wheel owns the current parameters,
BlockState snapshots, the noise sum, and the sampling random stream. Each block
draws its own start from its prior in initial_sample, with the RNG Wheel passes
at registration; an on_cycle_complete callback collects posterior chains outside
the blocks. The sine blocks register as signal blocks and the white-noise block
as the one noise block, so the sines receive exactly its covariance and the
noise block itself receives None because it is alone. Each block also
implements state_from_parameters, which rebuilds its state from a parameter
set. These
blocks request time-domain inputs; Wheel also runs them on Fourier or WDM
observations with data_domain="time", because the noise block publishes its
covariance on that time grid. Their likelihood requires time-domain noise with
independent channels.
Run with:

    uv run --no-sync python examples/toy_fit.py
"""

from typing import Any

import numpy as np

from enchilada import BlockState, Covariance, L1Data, Wheel


class SineBlock:
    """A known-frequency sinusoid with amplitude prior N(0, prior_std**2).

    The conditional combines this proper Gaussian prior with the likelihood under
    a per-sample noise variance on the data grid. Its waveform basis is derived
    from the supplied grid each call.
    """

    def __init__(self, name: str, freq: float, prior_std: float = 10.0):
        if not np.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("prior_std must be finite and positive")
        self.name, self.freq, self.prior_std = name, freq, prior_std

    def _basis(self, conditional_residual: L1Data) -> np.ndarray:
        if conditional_residual.data_domain != "time":
            raise ValueError("SineBlock requires time-domain data")
        t = (
            conditional_residual.start_time_gps
            + np.arange(conditional_residual.num_time_samples) * conditional_residual.dt
        )
        return np.sin(2.0 * np.pi * self.freq * t)

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        """The sinusoid at the given amplitude on `reference_data`'s time grid."""
        amplitude = float(parameters["amplitude"])
        basis = self._basis(reference_data)
        return reference_data.block_state(
            {ch: amplitude * basis for ch in reference_data.channel_names},
            parameters={"amplitude": amplitude},
        )

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw the amplitude from its prior and render it on the time grid."""
        return self.state_from_parameters(
            {"amplitude": float(rng.normal(0.0, self.prior_std))}, conditional_residual
        )

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        if covariance is None:
            raise ValueError("SineBlock requires a noise covariance")
        try:
            covariance.check_compatible(conditional_residual)
        except ValueError as exc:
            raise ValueError(
                "SineBlock requires time-domain white noise on the data grid"
            ) from exc
        channels = conditional_residual.channel_names
        if any(
            np.any(covariance.pair(first, second))
            for first in channels
            for second in channels
            if first != second
        ):
            raise ValueError("SineBlock requires uncorrelated channels")
        basis = self._basis(conditional_residual)
        # One independent noise variance per time sample and channel.
        precision = 1.0 / self.prior_std**2
        weighted_data = 0.0
        for ch in conditional_residual.channel_names:
            weights = basis / covariance.pair(ch, ch)
            precision += float(basis @ weights)
            weighted_data += float(conditional_residual.channel_data[ch] @ weights)
        amplitude = rng.normal(weighted_data / precision, np.sqrt(1.0 / precision))
        return self.state_from_parameters(
            {"amplitude": float(amplitude)}, conditional_residual
        )


class WhiteNoiseBlock:
    """White-noise variance with a proper inverse-gamma(shape, scale) prior.

    It models the whole noise, so the covariance it receives must be None
    (register it as the only noise block).
    """

    def __init__(self, name: str, shape: float = 2.0, scale: float = 1.0):
        if not np.isfinite([shape, scale]).all() or min(shape, scale) <= 0:
            raise ValueError(
                "inverse-gamma shape and scale must be finite and positive"
            )
        self.name, self.shape, self.scale = name, shape, scale

    def _require_alone(self, covariance: Covariance | None) -> None:
        if covariance is not None:
            raise ValueError(
                "WhiteNoiseBlock models the whole noise; "
                "register it as the only noise block"
            )

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        """White noise of variance sigma**2 on `reference_data`'s time grid."""
        if reference_data.data_domain != "time":
            raise ValueError("WhiteNoiseBlock requires time-domain data")
        sigma = float(parameters["sigma"])
        return BlockState(
            covariance=Covariance.like(
                reference_data,
                {
                    ch: np.full(reference_data.num_time_samples, sigma**2)
                    for ch in reference_data.channel_names
                },
            ),
            parameters={"sigma": sigma},
        )

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw the variance from its inverse-gamma prior and publish it."""
        self._require_alone(covariance)
        return self.state_from_parameters(
            {"sigma": float(np.sqrt(self.scale / rng.gamma(self.shape)))},
            conditional_residual,
        )

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        self._require_alone(covariance)
        if conditional_residual.data_domain != "time":
            raise ValueError("WhiteNoiseBlock requires time-domain data")
        # The noise block sees observations minus every signal contribution.
        n_total = sum(arr.size for arr in conditional_residual.channel_data.values())
        ssr = sum(
            float(arr @ arr) for arr in conditional_residual.channel_data.values()
        )
        shape = self.shape + 0.5 * n_total
        scale = self.scale + 0.5 * ssr
        return self.state_from_parameters(
            {"sigma": float(np.sqrt(scale / rng.gamma(shape)))}, conditional_residual
        )


TRUTH = {"slow": 3.0, "fast": 2.0, "sigma": 0.5}


def make_observed(seed: int = 0) -> L1Data:
    """Two sinusoids in white noise on a single channel."""
    rng = np.random.default_rng(seed)
    fs, n = 0.1, 4096
    t = np.arange(n) / fs
    data = (
        TRUTH["slow"] * np.sin(2.0 * np.pi * 0.004 * t)
        + TRUTH["fast"] * np.sin(2.0 * np.pi * 0.011 * t)
        + rng.normal(0.0, TRUTH["sigma"], n)
    )
    return L1Data(
        {"A": data},
        sample_rate_hz=fs,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
        start_time_gps=0.0,
    )


def run_toy_fit(n_cycles: int = 300, burn_in: int = 100, seed: int = 0):
    """Run the fit; return {name: (posterior_mean, posterior_std)}."""
    if not 0 <= burn_in < n_cycles:
        raise ValueError("burn_in must be nonnegative and smaller than n_cycles")
    observed = make_observed(seed)
    # Seed the campaign from a child of the observation seed so registration and
    # sampling never replay the observation-noise draws.
    campaign_seed = np.random.SeedSequence(seed).spawn(1)[0]
    wheel = Wheel(observed, random_seed=int(campaign_seed.generate_state(1)[0]))
    # The sine starts need no covariance, so the signal blocks may register first.
    for name, frequency in (("slow", 0.004), ("fast", 0.011)):
        wheel.add(
            SineBlock(name=name, freq=frequency), role="signal", data_domain="time"
        )
    wheel.add(WhiteNoiseBlock(name="noise"), role="noise", data_domain="time")
    chains: dict[str, list[float]] = {name: [] for name in ("slow", "fast", "noise")}

    def collect(cycle: int, w: Wheel) -> None:
        for name in chains:
            parameter = "sigma" if name == "noise" else "amplitude"
            chains[name].append(w.ledger[name].parameters[parameter])
        if (cycle + 1) % 100 == 0:
            rms = float(np.sqrt(np.mean(w.residual().channel_data["A"] ** 2)))
            print(
                f"cycle {cycle + 1:4d}: full-residual RMS = {rms:.4f}  "
                f"(sigma draw = {chains['noise'][-1]:.4f})"
            )

    wheel.run(n_cycles, on_cycle_complete=collect)
    results = {}
    for name, samples in chains.items():
        chain = np.asarray(samples[burn_in:])
        results[name] = (float(chain.mean()), float(chain.std()))
    return results


def main() -> None:
    print(
        f"truth: slow={TRUTH['slow']}, fast={TRUTH['fast']}, sigma={TRUTH['sigma']}\n"
    )
    results = run_toy_fit()
    print()
    for name, (mean, std) in results.items():
        key = "sigma" if name == "noise" else name
        print(f"{name:6s} posterior: {mean:.4f} +/- {std:.4f}   (truth {TRUTH[key]})")


if __name__ == "__main__":
    main()
