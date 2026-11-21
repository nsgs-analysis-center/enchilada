"""Measure complete sampling cycles with fixed signals and covariance.

Run from an installed checkout: python benchmarks/benchmark_wheel.py
Use --samples and --blocks to explore a different workload. Timings include
defensive snapshots and validation; these are measurements, not test thresholds.
The workload is the fixed signals plus one FixedNoiseBlock, which republishes its
covariance every cycle and is revalidated by Wheel each time, so timings are not
comparable with runs from before the noise sum. `--record` measures the same
workload while the Wheel records every cycle to a temporary campaign file.
"""

import argparse
import json
import resource
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from enchilada import Covariance, FixedNoiseBlock, L1Data, Wheel


class FixedSignal:
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def state_from_parameters(self, parameters, reference_data):
        return reference_data.block_state(
            {
                name: np.full_like(values, self.level)
                for name, values in reference_data.channel_data.items()
            }
        )

    def initial_sample(self, conditional_residual, covariance, *, rng):
        return self.state_from_parameters({}, conditional_residual)

    def sample(self, conditional_residual, covariance, current_block_state, *, rng):
        return current_block_state


def measure(num_blocks, num_samples, num_cycles, repeats, *, record=False):
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
    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "benchmark.h5" if record else None
        with Wheel(observed, random_seed=0, output_path=output_path) as wheel:
            # The noise goes first so every signal block sees the covariance from
            # registration onward, as before.
            wheel.add(FixedNoiseBlock("noise", covariance), role="noise")
            for index in range(num_blocks):
                wheel.add(FixedSignal(str(index), 0.5 / num_blocks), role="signal")
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
        "channels": 3,
        "recorded": record,
        "median_seconds_per_cycle": statistics.median(elapsed),
        "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32768)
    parser.add_argument("--blocks", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--record", action="store_true", help="write the campaign file too"
    )
    arguments = parser.parse_args()
    for num_blocks in arguments.blocks:
        print(
            json.dumps(
                measure(
                    num_blocks,
                    arguments.samples,
                    arguments.cycles,
                    arguments.repeats,
                    record=arguments.record,
                )
            ),
            flush=True,
        )
