"""The two-noise-block example recovers an excess variance above a known floor."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from conftest import make_observed as small_observed
from conftest import white_covariance
from enchilada import BlockState, Covariance, FixedNoiseBlock, Wheel, transform
from enchilada.testing import check_block

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def noise_floor():
    spec = importlib.util.spec_from_file_location(
        "noise_floor", EXAMPLES / "noise_floor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_excess_variance_is_recovered_above_the_floor(noise_floor):
    result = noise_floor.run_example(n_cycles=200, burn_in=50, seed=0)
    mean, std = result["posterior_mean"], result["posterior_std"]
    truth = noise_floor.TRUTH["excess"]
    assert std > 0.0
    assert mean == pytest.approx(truth, abs=max(5 * std, 0.05)), (
        f"excess: posterior {mean} +/- {std} vs truth {truth}"
    )
    wheel = result["wheel"]
    assert list(wheel.ledger) == ["floor", "excess"]
    np.testing.assert_array_equal(
        wheel.ledger["floor"].covariance.A, noise_floor.TRUTH["floor"]
    )
    # The sum lives on the home grid; translated back to time it is floor + excess.
    np.testing.assert_allclose(
        transform(wheel.covariance, "time").A,
        noise_floor.TRUTH["floor"] + wheel.ledger["excess"].covariance.A,
        rtol=1e-12,
    )
    assert wheel.ledger["excess"].template is None
    assert wheel.ledger["excess"].parameters["variance"] == pytest.approx(
        float(wheel.ledger["excess"].covariance.A[0])
    )


def test_the_excess_block_publishes_only_its_own_covariance_and_refuses_bad_inputs(
    noise_floor,
):
    observed = noise_floor.make_observed(seed=1)
    floor = white_covariance(observed, 0.25)
    block = noise_floor.ExcessNoiseBlock("excess")
    rng = np.random.default_rng(0)
    start = block.initial_sample(observed, floor, rng=rng)
    assert isinstance(start, BlockState)
    assert start.template is None
    np.testing.assert_array_equal(start.covariance.A, start.parameters["variance"])
    later = block.sample(observed, floor, start, rng=rng)
    np.testing.assert_array_equal(later.covariance.A, later.parameters["variance"])
    with pytest.raises(ValueError, match="floor"):
        block.initial_sample(observed, None, rng=rng)
    with pytest.raises(ValueError, match="time-domain"):
        block.sample(observed.to_frequency(), None, start, rng=rng)
    # A floor with a cross-channel term is refused.
    two = small_observed(rng, num_time_samples=16, channel_names=("A", "E"))
    blocks = np.zeros((2, 2, 16))
    blocks[0, 0] = blocks[1, 1] = 1.0
    blocks[0, 1] = blocks[1, 0] = 0.3
    correlated = Covariance(blocks, channel_names=("A", "E"), grid=two.grid)
    with pytest.raises(ValueError, match="uncorrelated"):
        block.initial_sample(two, correlated, rng=rng)


def test_the_excess_block_passes_check_block_beside_a_floor(noise_floor):
    observed = noise_floor.make_observed(seed=0)
    check_block(
        noise_floor.ExcessNoiseBlock("excess"),
        observed,
        noise_blocks=(FixedNoiseBlock("floor", white_covariance(observed, 0.25)),),
        num_frequency_divisions=32,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low": 0.0},
        {"low": 1.0, "high": 1.0},
        {"high": float("inf")},
        {"step": 0.0},
        {"step": float("nan")},
        {"steps_per_call": 0},
        {"steps_per_call": True},
        {"steps_per_call": 2.5},
    ],
)
def test_the_excess_block_validates_its_settings(noise_floor, kwargs):
    with pytest.raises(ValueError):
        noise_floor.ExcessNoiseBlock("excess", **kwargs)


def test_a_different_floor_moves_the_excess_posterior(noise_floor):
    # The data have total variance 1.0 (floor 0.25 + excess 0.75). A floor of
    # 0.5 must leave an excess of 0.5, so the block has to read the floor it is
    # handed rather than assume the example's own 0.25.
    observed = noise_floor.make_observed(seed=0)
    wheel = Wheel(observed, num_frequency_divisions=32, random_seed=1)
    wheel.add(FixedNoiseBlock("floor", white_covariance(observed, 0.5)))
    wheel.add(noise_floor.ExcessNoiseBlock("excess"))
    chain: list[float] = []

    def collect(cycle, w):
        np.testing.assert_array_equal(w.ledger["floor"].covariance.A, 0.5)
        chain.append(w.ledger["excess"].parameters["variance"])

    wheel.run(200, on_cycle_complete=collect)
    samples = np.asarray(chain[50:])
    mean, std = float(samples.mean()), float(samples.std())
    assert mean == pytest.approx(0.5, abs=max(5 * std, 0.05)), (mean, std)
    assert abs(mean - noise_floor.TRUTH["excess"]) >= 0.1, (mean, std)


def test_a_seeded_campaign_is_reproducible(noise_floor):
    first = noise_floor.run_example(n_cycles=60, burn_in=10, seed=0)
    second = noise_floor.run_example(n_cycles=60, burn_in=10, seed=0)
    assert first["posterior_mean"] == second["posterior_mean"]
    assert first["posterior_std"] == second["posterior_std"]
