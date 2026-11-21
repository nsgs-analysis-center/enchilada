"""The shipped examples run, and the toy fit actually converges to truth."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import white_covariance

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_runs_clean():
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "demo.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "residual RMS" in proc.stdout


def test_toy_fit_converges_to_truth():
    toy = load_example("toy_fit")
    results = toy.run_toy_fit(n_cycles=200, burn_in=80, seed=0)
    truth = {
        "slow": toy.TRUTH["slow"],
        "fast": toy.TRUTH["fast"],
        "noise": toy.TRUTH["sigma"],
    }
    for name, (mean, std) in results.items():
        assert std > 0.0
        assert mean == pytest.approx(truth[name], abs=max(5 * std, 0.05)), (
            f"{name}: posterior {mean} +/- {std} vs truth {truth[name]}"
        )


def test_notebook_code_cells_execute():
    nb = json.loads((EXAMPLES / "demo.ipynb").read_text())
    code = "\n\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("notebook_path", sorted(EXAMPLES.glob("*.ipynb")))
def test_all_notebooks_contain_valid_python(notebook_path):
    """Check even optional notebooks when their scientific stack is absent."""
    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells, f"{notebook_path.name} has no executable example"
    for index, cell in enumerate(code_cells, 1):
        compile("".join(cell["source"]), f"{notebook_path}:cell-{index}", "exec")


def test_toy_prior_is_random_and_separate_campaigns_are_reproducible():
    from enchilada import Wheel

    toy = load_example("toy_fit")
    observed = toy.make_observed()
    campaigns = [Wheel(observed, random_seed=seed) for seed in (42, 42, 43)]
    for wheel in campaigns:
        wheel.add(toy.SineBlock(name="slow", freq=0.004), role="signal")
        wheel.add(toy.WhiteNoiseBlock(name="noise"), role="noise")
    priors = [w.ledger["slow"].parameters["amplitude"] for w in campaigns]
    assert priors[0] == priors[1]
    assert priors[0] != priors[2]
    assert priors[0] != 0.0
    campaigns[0].run(3)
    campaigns[1].run(3)
    assert (
        campaigns[0].ledger["slow"].parameters == campaigns[1].ledger["slow"].parameters
    )
    assert (
        campaigns[0].ledger["noise"].parameters
        == campaigns[1].ledger["noise"].parameters
    )


def test_white_noise_block_refuses_to_sit_beside_another_noise_block():
    from enchilada import Covariance, FixedNoiseBlock, Wheel

    toy = load_example("toy_fit")
    observed = toy.make_observed()
    wheel = Wheel(observed, random_seed=0)
    floor = Covariance.like(observed, {"A": np.full(observed.num_time_samples, 0.2)})
    wheel.add(FixedNoiseBlock("floor", floor), role="noise", data_domain="time")
    with pytest.raises(ValueError, match="only noise block"):
        wheel.add(toy.WhiteNoiseBlock(name="noise"), role="noise", data_domain="time")


def test_sine_conditional_includes_declared_amplitude_prior():
    from enchilada import L1Data

    toy = load_example("toy_fit")
    data = L1Data(
        channel_data={"A": np.array([0.0, 4.0, 0.0, -4.0])},
        sample_rate_hz=1.0,
        channel_names=("A",),
        tdi_generation="2.0",
        physical_observable="fractional_frequency",
    )
    noise = white_covariance(data, 2.0)
    signal = toy.SineBlock("signal", 0.25, prior_std=0.5)
    prior = signal.initial_sample(data, None, rng=np.random.default_rng(0))
    draws = np.array(
        [
            signal.sample(
                data, noise, prior, rng=np.random.default_rng(seed)
            ).parameters["amplitude"]
            for seed in range(4000)
        ]
    )
    # <s,s>=2, <s,d>=8: precision=2/2 + 1/0.25=5, mean=(8/2)/5.
    assert draws.mean() == pytest.approx(0.8, abs=0.025)
    assert draws.var() == pytest.approx(0.2, abs=0.015)


def test_sine_rejects_covariance_its_likelihood_cannot_use():
    from dataclasses import replace

    from enchilada import Covariance

    toy = load_example("toy_fit")
    observed = toy.make_observed()
    data = replace(
        observed,
        channel_names=("A", "E"),
        channel_data={"A": observed.channel_data["A"], "E": observed.channel_data["A"]},
    )
    signal = toy.SineBlock("signal", 0.004)
    n = data.num_time_samples
    noise = Covariance.like(
        data, {"A": np.ones(n), "E": np.ones(n), "AE": np.full(n, 0.5)}
    )
    prior = signal.initial_sample(data, None, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="uncorrelated"):
        signal.sample(data, noise, prior, rng=np.random.default_rng(1))


@pytest.fixture
def gb_model_with_test_waveform(monkeypatch):
    # Only the external waveform is replaced with a small deterministic model.
    model = load_example("gb_model")
    monkeypatch.setattr(model, "_new_gb", object)

    def test_waveform(gb, params, angles, Tobs, dt, NB):
        amp, f0, fdot, phi0 = params
        si = 5 + int(f0)
        bins = np.arange(NB)
        h = amp * np.exp(1j * (phi0 + fdot * bins))
        return si, h, 0.5 * h

    monkeypatch.setattr(model, "gb_template", test_waveform)
    return model


@pytest.fixture
def gb_model_with_eryn(gb_model_with_test_waveform):
    # Exercise real Eryn continuation without requiring GBGPU's native backend.
    pytest.importorskip(
        "eryn.ensemble", reason="GB sampler continuation requires optional Eryn"
    )
    return gb_model_with_test_waveform


def gb_test_context():
    from enchilada import Covariance, L1Data

    data = L1Data(
        channel_data={"A": np.ones(33, dtype=complex)},
        sample_rate_hz=1.0,
        num_time_samples=64,
        channel_names=("A",),
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        data_domain="frequency",
    )
    return data, Covariance.like(data, {"A": np.full(33, data.Tobs / 2)})


@pytest.mark.parametrize("num_time_samples", [63, 64])
def test_gb_injection_noise_matches_real_fft_statistics(
    gb_model_with_test_waveform, num_time_samples
):
    from enchilada import L1Data

    class ConstantPSD:
        def psd(self, frequencies):
            return np.full_like(frequencies, 2.0)

    duration = float(num_time_samples)
    draws = []
    for seed in range(2000):
        channel_data, _ = gb_model_with_test_waveform.inject_gb(
            truth=(0.0, 0.0, 0.0, 0.0),
            angles=(0.0, 0.0, 0.0, 0.0),
            Tobs=duration,
            dt=1.0,
            n_samples=num_time_samples,
            channels=("A", "E"),
            noise=ConstantPSD(),
            NB=4,
            seed=seed,
        )
        draws.extend(channel_data.values())
    spectra = np.asarray(draws)
    np.testing.assert_array_equal(spectra[:, 0], 0.0)
    # Interior complex bins split S/(2 df) equally between real and imaginary.
    component_variance = 2.0 * duration / 4.0
    assert spectra[:, 1].real.var() == pytest.approx(component_variance, rel=0.08)
    assert spectra[:, 1].imag.var() == pytest.approx(component_variance, rel=0.08)
    if num_time_samples % 2 == 0:
        # Nyquist has one real degree of freedom with the full S/(2 df) variance.
        np.testing.assert_array_equal(spectra[:, -1].imag, 0.0)
        assert spectra[:, -1].real.var() == pytest.approx(
            2.0 * component_variance, rel=0.08
        )
    else:
        assert spectra[:, -1].real.var() == pytest.approx(component_variance, rel=0.08)
        assert spectra[:, -1].imag.var() == pytest.approx(component_variance, rel=0.08)
    data = L1Data(
        channel_data=channel_data,
        sample_rate_hz=1.0,
        num_time_samples=num_time_samples,
        channel_names=("A", "E"),
        tdi_generation="1.5",
        physical_observable="fractional_frequency",
        data_domain="frequency",
    )
    for channel_name in data.channel_names:
        np.testing.assert_allclose(
            data.to_time().to_frequency().channel_data[channel_name],
            data.channel_data[channel_name],
            atol=1e-12,
        )


def test_gb_eryn_restart_and_global_rng_isolation(gb_model_with_eryn):
    from copy import deepcopy

    gb = gb_model_with_eryn
    data, noise = gb_test_context()
    settings = dict(
        bounds=[[0.1, 2.0], [0.0, 2.0], [0.0, 0.5], [0.0, 6.0]],
        angles=(0.0, 0.0, 0.0, 0.0),
        n_walkers=12,
        band=4,
    )
    short = gb.GBBlock(**settings, steps_per_cycle=2)
    long = gb.GBBlock(**settings, steps_per_cycle=4)
    prior = short.initial_sample(data, noise, rng=np.random.default_rng(2))
    initial_internal_state = deepcopy(short.state)
    prior_copy = deepcopy(prior.template)
    long.state = deepcopy(initial_internal_state)
    global_rng_before = np.random.get_state()
    first = short.sample(data, noise, prior, rng=np.random.default_rng(3))
    # A restart explicitly restores the model's own continuation snapshot.
    restarted_block = gb.GBBlock(**settings, steps_per_cycle=2)
    restarted_block.state = deepcopy(short.state)
    restarted = restarted_block.sample(
        data, noise, first, rng=np.random.default_rng(999)
    )
    uninterrupted = long.sample(data, noise, prior, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(
        restarted_block.state.coords["model_0"],
        long.state.coords["model_0"],
    )
    np.testing.assert_array_equal(prior.template["A"], prior_copy["A"])
    assert restarted.parameters == uninterrupted.parameters
    assert restarted_block.state.samples.shape == (24, 4)
    assert restarted_block.state.updates == 2
    assert initial_internal_state.updates == 0
    assert short.state.updates == 1
    np.testing.assert_array_equal(np.random.get_state()[1], global_rng_before[1])
    assert np.random.get_state()[2:] == global_rng_before[2:]


def test_gb_recomputes_likelihood_for_changed_residual(gb_model_with_eryn):
    from copy import deepcopy
    from dataclasses import replace

    gb = gb_model_with_eryn
    data, noise = gb_test_context()
    block = gb.GBBlock(
        bounds=[[0.1, 2.0], [0.0, 2.0], [0.0, 0.5], [0.0, 6.0]],
        angles=(0.0, 0.0, 0.0, 0.0),
        n_walkers=12,
        steps_per_cycle=3,
        band=4,
    )
    prior = block.initial_sample(data, noise, rng=np.random.default_rng(4))
    first = block.sample(data, noise, prior, rng=np.random.default_rng(5))
    internal_snapshot = deepcopy(block.state)
    changed = replace(data, channel_data={"A": 1e5 * data.channel_data["A"]})
    block.sample(changed, noise, first, rng=np.random.default_rng(6))
    updated_coords = block.state.coords["model_0"].copy()
    block.state = deepcopy(internal_snapshot)
    block.sample(data, noise, first, rng=np.random.default_rng(6))
    assert not np.array_equal(
        updated_coords,
        block.state.coords["model_0"],
    )


def test_sine_requires_time_white_noise_on_the_data_grid():
    toy = load_example("toy_fit")
    data = toy.make_observed()
    block = toy.SineBlock("slow", 0.004)
    prior = block.initial_sample(data, None, rng=np.random.default_rng(0))
    noise = white_covariance(data.to_frequency(), 1.0)
    with pytest.raises(ValueError, match="time-domain white noise on the data grid"):
        block.sample(data, noise, prior, rng=np.random.default_rng(1))


@pytest.mark.parametrize(
    "make_block",
    [
        lambda toy: toy.SineBlock("slow", 0.004),
        lambda toy: toy.WhiteNoiseBlock("noise"),
    ],
    ids=["sine", "white_noise"],
)
def test_toy_starts_require_time_data(make_block):
    toy = load_example("toy_fit")
    block = make_block(toy)
    with pytest.raises(ValueError, match="time-domain"):
        block.initial_sample(
            toy.make_observed().to_frequency(), None, rng=np.random.default_rng(0)
        )


def test_load_mojito_example_reads_a_synthetic_file(tmp_path, capsys):
    from conftest import write_mojito_file

    path = tmp_path / "mojito.h5"
    write_mojito_file(path, n=4096)  # 1024 s at 4 Hz
    example = load_example("load_mojito")
    data = example.main(path, duration_s=1000.0, factor=4)
    assert data.num_time_samples == 1000
    assert data.sample_rate_hz == 1.0
    out = capsys.readouterr().out
    for fragment in ("native grid", "decimated grid", "orbit span"):
        assert fragment in out, out
    for name in data.channel_names:
        assert f"  {name}: rms" in out, out
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "load_mojito.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in os.environ.items() if k != "ENCHILADA_MOJITO_L1"},
    )
    assert proc.returncode != 0
    assert "usage" in (proc.stderr + proc.stdout).lower()


def test_mojito_wdm_wheel_example_builds_a_wdm_wheel(tmp_path, capsys):
    from conftest import MOJITO_T0, write_mojito_file
    from enchilada import WDMGrid

    path = tmp_path / "mojito.h5"
    write_mojito_file(path, n=4096)  # 1024 s at 4 Hz
    example = load_example("mojito_wdm_wheel")
    wheel = example.main(path, num_frequency_divisions=8, factor=4, duration_s=1000.0)
    residual = wheel.residual()
    assert residual.data_domain == "wdm"
    assert residual.wdm_grid == WDMGrid(
        num_frequency_divisions=8, num_time_divisions=124
    )
    assert residual.num_time_samples == 992  # 3968 native samples / 4
    assert residual.sample_rate_hz == 1.0
    assert residual.start_time_gps == MOJITO_T0
    for name in residual.channel_names:
        assert residual.channel_data[name].shape == (8, 124)
    out = capsys.readouterr().out
    assert "window: 992.0 s" in out, out
    assert "WDM grid: Nf = 8, Nt = 124" in out, out
    # 1/(2*8*1 s) and 8*1 s at the decimated 1 Hz
    assert "dF = 6.250e-02 Hz, dT = 8.0 s" in out, out
    for name in residual.channel_names:
        assert f"  {name}: rms" in out, out
    assert out.count("per WDM coefficient") == len(residual.channel_names), out
    assert "residual: domain wdm" in out, out
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / "mojito_wdm_wheel.py")],
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in os.environ.items() if k != "ENCHILADA_MOJITO_L1"},
    )
    assert proc.returncode != 0
    assert "usage" in (proc.stderr + proc.stdout).lower()


@pytest.mark.parametrize(
    ("duration_s", "expected"),
    [
        (1000.0, (992.0, 124)),  # int(1000 / 8) = 125 columns, rounded down to even
        (16.0, (16.0, 2)),  # exactly two 8 s columns
    ],
)
def test_fit_window_rounds_down_to_an_even_number_of_columns(duration_s, expected):
    example = load_example("mojito_wdm_wheel")
    assert (
        example.fit_window(duration_s, dt=0.25, factor=4, num_frequency_divisions=8)
        == expected
    )


def test_fit_window_refuses_a_window_shorter_than_two_columns():
    example = load_example("mojito_wdm_wheel")
    with pytest.raises(ValueError, match="fewer than two WDM time columns"):
        example.fit_window(15.9, dt=0.25, factor=4, num_frequency_divisions=8)


REAL_MOJITO_FILE = os.environ.get("ENCHILADA_MOJITO_L1")


@pytest.mark.skipif(
    not REAL_MOJITO_FILE, reason="set ENCHILADA_MOJITO_L1 to a Mojito L1 file"
)
def test_mojito_wdm_wheel_example_on_the_real_file(capsys):
    """A week at Nf = 512: 2,416,640 native samples per channel, decimated by 10
    to 512 x 472 WDM coefficients. Takes about a second on the Mojito Light file."""
    from enchilada import WDMGrid

    with h5py.File(REAL_MOJITO_FILE, "r") as f:
        t0 = float(f["tdis/sampling"].attrs["t0"])
    example = load_example("mojito_wdm_wheel")
    wheel = example.main(REAL_MOJITO_FILE)
    residual = wheel.residual()
    assert residual.wdm_grid == WDMGrid(
        num_frequency_divisions=512, num_time_divisions=472
    )
    assert residual.start_time_gps == t0
    out = capsys.readouterr().out
    assert "WDM grid: Nf = 512, Nt = 472" in out, out
    assert "dF = 3.906e-04 Hz, dT = 1280.0 s" in out, out
