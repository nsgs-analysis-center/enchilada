"""Regression cases found while independently reviewing object translation."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import BlockState, Covariance, WDMGrid, Wheel
from enchilada.translation import transform


def test_noise_only_wdm_state_translation_carries_its_covariance(observed):
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=16)
    wdm_reference = replace(
        observed,
        channel_data={
            name: np.zeros(grid.array_shape) for name in observed.channel_names
        },
        data_domain="wdm",
    )
    variances = np.where(grid.active_mask, 1.0, 0.0)
    covariance = Covariance.like(
        wdm_reference, {name: variances for name in observed.channel_names}
    )
    original = BlockState(covariance=covariance, parameters={"variance": 1.0})
    converted = transform(original, "wdm", reference_data=wdm_reference)
    assert converted.parameters == original.parameters
    assert converted.template is None
    assert converted.covariance is not covariance
    assert converted.covariance.grid == wdm_reference.grid
    np.testing.assert_array_equal(converted.covariance.A, covariance.A)
    assert not np.shares_memory(converted.covariance.blocks, covariance.blocks)


def test_omitted_domain_and_explicit_same_domain_both_receive_the_covariance(observed):
    received = []

    class RecordInputs:
        def __init__(self, name):
            self.name = name

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            received.append(("sample", self.name, conditional_residual.data_domain))
            return current_block_state

    wheel = Wheel(observed, white_covariance(observed, 2.0))
    wheel.add(RecordInputs("default"), initial_block_state=BlockState())
    wheel.add(
        RecordInputs("explicit"), initial_block_state=BlockState(), data_domain="time"
    )
    assert received == []
    wheel.run(1)
    assert received == [("sample", "default", "time"), ("sample", "explicit", "time")]
    np.testing.assert_array_equal(wheel.covariance.A, 2.0)


def test_same_grid_block_receives_the_covariance_without_transform(
    observed, monkeypatch
):
    """An explicit domain equal to the observations' hands the covariance over
    as a deep copy; nothing goes through transform()."""
    import enchilada.wheel

    delegate = enchilada.wheel.transform

    def transform_without_covariance(value, *args, **kwargs):
        if isinstance(value, Covariance):
            raise AssertionError("a same-grid covariance went through transform()")
        return delegate(value, *args, **kwargs)

    monkeypatch.setattr(enchilada.wheel, "transform", transform_without_covariance)
    received = []

    class RecordCovariance:
        name = "explicit"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            covariance.check_compatible(conditional_residual)
            received.append(float(covariance.A[0]))
            return current_block_state

    wheel = Wheel(observed, white_covariance(observed, 2.0))
    wheel.add(RecordCovariance(), initial_block_state=BlockState(), data_domain="time")
    wheel.run(2)
    assert received == [2.0, 2.0]


@pytest.mark.parametrize(
    "canonical_domain",
    [
        "time",
        pytest.param(
            "wdm",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("wdm") is None,
                reason="local WDM backend is optional",
            ),
        ),
    ],
)
def test_rejected_translated_output_rolls_back_mutated_inputs_and_rng(
    observed, canonical_domain
):
    canonical = (
        observed
        if canonical_domain == "time"
        else transform(observed, "wdm", num_frequency_divisions=8)
    )
    fail = [True]

    class MutateInputs:
        name = "mutate"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            amplitude = rng.normal()
            if fail[0]:
                conditional_residual.channel_data["A"][:] = 1e9
                current_block_state.parameters["coordinates"][0] = -8.0
                signal = {
                    name: np.zeros_like(values)
                    for name, values in conditional_residual.channel_data.items()
                }
                signal["A"][0] = 1j
                return BlockState(template=signal)
            return BlockState(
                parameters={"amplitude": amplitude, "coordinates": np.array([3.0])}
            )

    actual = Wheel(canonical, random_seed=7)
    control = Wheel(canonical, random_seed=7)
    for wheel in (actual, control):
        wheel.add(
            MutateInputs(),
            initial_block_state=BlockState(
                parameters={"amplitude": 0.0, "coordinates": np.array([3.0])}
            ),
            data_domain="frequency",
        )
    previous_amplitude = actual.ledger["mutate"].parameters["amplitude"]
    with pytest.raises(ValueError, match="DC coefficient must be real"):
        actual.run(1)
    assert actual.ledger["mutate"].parameters["amplitude"] == previous_amplitude
    np.testing.assert_array_equal(
        actual.ledger["mutate"].parameters["coordinates"], [3.0]
    )
    for name in canonical.channel_names:
        np.testing.assert_array_equal(
            actual.observed_data.channel_data[name], canonical.channel_data[name]
        )
        np.testing.assert_array_equal(
            actual.residual().channel_data[name], canonical.channel_data[name]
        )
    fail[0] = False
    actual.run(1)
    control.run(1)
    assert (
        actual.ledger["mutate"].parameters["amplitude"]
        == control.ledger["mutate"].parameters["amplitude"]
    )
