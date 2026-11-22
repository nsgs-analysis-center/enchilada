"""Measure complete sampling cycles with fixed signals and covariance.

Run from an installed checkout: python benchmarks/benchmark_wheel.py
Use --samples, --rows and --blocks to explore a different workload. Timings
include defensive snapshots and validation; these are measurements, not test
thresholds. The workload is the fixed signals plus one FixedNoiseBlock, which
republishes its covariance every cycle and is revalidated by Wheel each time, so
timings are not comparable with runs from before the noise sum or the home grid
(time-grid blocks pay a transform per call and a covariance translation per
block). The campaign
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
        help="rows of the WDM home grid; --rows must be a positive even integer "
        "and --samples a multiple of 2 * --rows",
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if not (
        arguments.rows > 0
        and arguments.rows % 2 == 0
        and arguments.samples % (2 * arguments.rows) == 0
    ):
        parser.error(
            "--rows must be a positive even integer and --samples a multiple "
            "of 2 * --rows"
        )
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
