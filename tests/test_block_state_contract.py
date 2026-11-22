"""Physical estimates cross the Wheel boundary; model internals stay on blocks."""

from dataclasses import dataclass, field

import numpy as np

from enchilada import BlockState, Wheel


def test_block_local_state_survives_calls_without_entering_the_ledger(observed):
    class LocalResource:
        def __deepcopy__(self, memo):
            raise AssertionError("Wheel must not copy resources owned by a block")

    class StatefulSignal:
        role = "signal"
        name = "signal"

        @dataclass
        class State:
            num_calls: int = 0
            metadata: dict = field(default_factory=dict)

        def __init__(self):
            self.state = self.State(metadata={"resource": LocalResource()})

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return conditional_residual.zero_block_state(parameters={"amplitude": 0.0})

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            self.state.num_calls += 1
            amplitude = current_block_state.parameters["amplitude"] + 1.0
            return conditional_residual.block_state(
                template={
                    name: np.full_like(values, amplitude)
                    for name, values in conditional_residual.channel_data.items()
                },
                parameters={"amplitude": amplitude},
            )

    block = StatefulSignal()
    wheel = Wheel(observed, num_frequency_divisions=8, random_seed=42)
    wheel.add(block)
    wheel.run(2)

    accepted = wheel.ledger["signal"]
    assert isinstance(accepted, BlockState)
    assert block.state.num_calls == 2
    assert isinstance(block.state.metadata["resource"], LocalResource)
    assert accepted.parameters == {"amplitude": 2.0}
    for name in observed.channel_names:
        # The ledger keeps the template on the home grid; the views are on the data's.
        assert accepted.template[name].shape == wheel.home_grid.channel_shape
        np.testing.assert_allclose(wheel.contribution("signal")[name], 2.0, rtol=1e-12)
        np.testing.assert_allclose(
            wheel.working_residual.channel_data[name], observed.channel_data[name] - 2.0
        )
    assert accepted.covariance is None

    accepted.parameters["amplitude"] = 100.0
    accepted.template["A"][:] = 100.0
    assert wheel.ledger["signal"].parameters == {"amplitude": 2.0}
    np.testing.assert_allclose(wheel.contribution("signal")["A"], 2.0, rtol=1e-12)
