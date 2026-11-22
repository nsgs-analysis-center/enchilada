"""The time-domain toy preserves its likelihood across Wheel representations."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from enchilada import Wheel, transform


@pytest.fixture
def toy():
    path = Path(__file__).resolve().parent.parent / "examples" / "toy_fit.py"
    spec = importlib.util.spec_from_file_location("toy_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("canonical_domain", ["frequency", "wdm"])
def test_toy_time_blocks_preserve_draws_on_translated_observations(
    toy, canonical_domain
):
    data = toy.make_observed()
    options = {"num_frequency_divisions": 8} if canonical_domain == "wdm" else {}
    translated_data = transform(data, canonical_domain, **options)
    chains = []
    wheels = []
    for observed in (data, translated_data):
        # One 8 x 512 home grid for both campaigns (for the wdm case, the
        # observations' own).
        wheel = Wheel(observed, num_frequency_divisions=8, random_seed=42)
        for name, frequency in (("slow", 0.004), ("fast", 0.011)):
            wheel.add(toy.SineBlock(name, frequency), data_domain="time")
        noise_block = toy.WhiteNoiseBlock("noise")
        wheel.add(noise_block, data_domain="time")
        chain = []

        def collect(cycle, campaign, samples=chain):
            samples.append(
                [
                    campaign.ledger["slow"].parameters["amplitude"],
                    campaign.ledger["fast"].parameters["amplitude"],
                    campaign.ledger["noise"].parameters["sigma"],
                ]
            )

        wheel.run(6, on_cycle_complete=collect)
        chains.append(chain)
        wheels.append(wheel)

    np.testing.assert_allclose(chains[1], chains[0], rtol=1e-12, atol=1e-12)
    # Both ledgers hold the noise block's publication on the time grid it
    # samples on; both sums live on the shared home grid.
    for wheel in wheels:
        assert wheel.ledger["noise"].covariance.grid == data.grid
        assert wheel.covariance.grid == wheel.home_grid
    assert wheels[1].home_grid == wheels[0].home_grid
    np.testing.assert_allclose(
        wheels[1].covariance.A, wheels[0].covariance.A, rtol=1e-12
    )
    restored_residual = transform(wheels[1].working_residual, "time")
    np.testing.assert_allclose(
        restored_residual.channel_data["A"],
        wheels[0].working_residual.channel_data["A"],
        rtol=1e-11,
        atol=1e-11,
    )
