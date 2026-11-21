"""The block-author tutorial is executable and its sampler targets the posterior."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

TUTORIAL = Path(__file__).resolve().parents[1] / "examples" / "create_block.ipynb"


def test_block_tutorial_runs_and_checks_its_sampler(tmp_path):
    notebook = json.loads(TUTORIAL.read_text())
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    code += (
        "\nif plt is not None:\n"
        "    for number in plt.get_fignums():\n"
        "        plt.figure(number).canvas.draw()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "Known-injection start checks passed" in result.stdout
    assert "Posterior checks passed" in result.stdout
    assert "Continuation and isolation checks passed" in result.stdout
    assert "PSD construction checks passed" in result.stdout
    assert "Noise block checks passed" in result.stdout


@pytest.fixture
def tutorial_definitions():
    """Load the actual taught class without rerunning plots or campaigns."""
    notebook = json.loads(TUTORIAL.read_text())
    module_name = "block_tutorial_for_test"
    import types

    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    try:
        for cell in notebook["cells"]:
            if "definitions" in cell.get("metadata", {}).get("tags", []):
                exec("".join(cell["source"]), module.__dict__)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_tutorial_sampler_refreshes_target_when_other_blocks_change_it(
    tutorial_definitions,
):
    import numpy as np

    from conftest import white_covariance
    from enchilada import L1Data

    module = tutorial_definitions
    times = np.arange(64)
    basis = np.sin(2 * np.pi * 0.125 * times)
    data = L1Data(
        channel_data={"A": 2.0 * basis},
        channel_names=("A",),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    noise = white_covariance(data, 0.2)
    block = module.SineAmplitudeBlock(
        name="signal", frequency_hz=0.125, proposal_std=0.4, steps_per_call=20
    )
    start = block.state_from_parameters({"amplitude": 0.0}, data)
    # Identical continuation state and RNG, but the other blocks change the
    # conditional data and covariance. A cached old target would repeat a draw.
    positive = block.sample(data, noise, start, rng=np.random.default_rng(1))
    changed_data = replace(data, channel_data={"A": -2.0 * basis})
    changed_noise = white_covariance(changed_data, 0.4)
    negative = block.sample(
        changed_data, changed_noise, start, rng=np.random.default_rng(1)
    )
    assert positive.parameters["amplitude"] > 0
    assert negative.parameters["amplitude"] < 0
    assert start.parameters == {"amplitude": 0.0}
    assert block.state.num_proposals == 2 * block.steps_per_call


def test_tutorial_noise_block_passes_check_block(tutorial_definitions):
    import numpy as np

    from enchilada import L1Data
    from enchilada.testing import check_block

    data = L1Data(
        channel_data={"A": np.zeros(64)},
        channel_names=("A",),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    check_block(
        tutorial_definitions.WhiteNoiseVarianceBlock(),
        data,
        num_cycles=3,
        role="noise",
    )


def test_tutorial_rejection_returns_retained_signal_and_complete_state(
    tutorial_definitions,
):
    import numpy as np

    from conftest import white_covariance
    from enchilada import L1Data

    data = L1Data(
        channel_data={"A": np.zeros(64)},
        channel_names=("A",),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    noise = white_covariance(data, 0.2)
    block = tutorial_definitions.SineAmplitudeBlock(
        name="signal", frequency_hz=0.125, proposal_std=1e12, steps_per_call=8
    )
    start = block.state_from_parameters({"amplitude": 0.0}, data)
    result = block.sample(data, noise, start, rng=np.random.default_rng(1))
    assert result.parameters == start.parameters
    np.testing.assert_array_equal(result.template["A"], start.template["A"])
    assert block.state.num_proposals == 8
    assert block.state.num_acceptances == 0
    assert not hasattr(result, "sampler_state")


def test_tutorial_noise_block_refuses_another_noise_block(tutorial_definitions):
    import numpy as np

    from conftest import white_covariance
    from enchilada import FixedNoiseBlock, L1Data, Wheel

    data = L1Data(
        channel_data={"A": np.zeros(64)},
        channel_names=("A",),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    floor = white_covariance(data, 0.2)
    WhiteNoiseVarianceBlock = tutorial_definitions.WhiteNoiseVarianceBlock

    # Registered second, its start already receives the floor's covariance.
    second = Wheel(data, random_seed=0)
    second.add(FixedNoiseBlock("floor", floor), role="noise")
    with pytest.raises(ValueError, match="only noise block"):
        second.add(WhiteNoiseVarianceBlock(), role="noise")

    # Registered first, it starts alone; the floor registered after it reaches
    # it at its first sample.
    first = Wheel(data, random_seed=0)
    first.add(WhiteNoiseVarianceBlock(), role="noise")
    first.add(FixedNoiseBlock("floor", floor), role="noise")
    with pytest.raises(ValueError, match="only noise block"):
        first.run(1)


def test_tutorial_block_instances_keep_independent_diagnostics(tutorial_definitions):
    first = tutorial_definitions.SineAmplitudeBlock("first", 0.125)
    second = tutorial_definitions.SineAmplitudeBlock("second", 0.125)
    first.state.num_proposals = 5
    assert second.state.num_proposals == 0
    assert first.state is not second.state


def test_tutorial_blocks_draw_their_start_from_the_prior_when_unconfigured(
    tutorial_definitions,
):
    import numpy as np

    from enchilada import L1Data

    data = L1Data(
        channel_data={"A": np.zeros(64)},
        channel_names=("A",),
        sample_rate_hz=1.0,
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    signal = tutorial_definitions.SineAmplitudeBlock(name="signal", frequency_hz=0.125)
    noise = tutorial_definitions.WhiteNoiseVarianceBlock()

    def starts(block, key):
        draws = []
        for seed in (0, 0, 1):
            start = block.initial_sample(data, None, rng=np.random.default_rng(seed))
            draws.append(start.parameters[key])
        return draws

    # Without a configured start, the start is a prior draw from the given RNG.
    amplitudes = starts(signal, "amplitude")
    variances = starts(noise, "variance")
    for same_a, same_b, other in (amplitudes, variances):
        assert same_a == same_b
        assert same_a != other
    assert min(variances) > 0
