"""Helpers for checking the Wheel/Block exchange contract end to end."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from enchilada.block import Block
from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.wheel import Role, Wheel


class EchoBlock:
    """Zero-signal diagnostic block that prints residual RMS and advances a counter.

    Register with ``wheel.add(block, role="signal")``: ``initial_sample``
    publishes an empty state, and every ``sample`` call does the same after
    printing. Inspect ``block.state.num_sample_calls`` for the counter;
    diagnostics belong to ``block.state.metadata``. Use a separate instance per
    campaign.
    """

    @dataclass
    class State:
        """Mutable continuation state and diagnostics owned by this block."""

        num_sample_calls: int = 0
        metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str):
        self.name = name
        self.state = self.State()

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Publish an empty physical state; the counter and the RNG are untouched."""
        return BlockState()

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        """Rebuild the empty state; EchoBlock publishes no signal and no parameters."""
        return BlockState()

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Print the residual RMS, advance the local counter, and publish no signal."""
        first_channel_name = conditional_residual.channel_names[0]
        residual_rms = float(
            np.sqrt(
                np.mean(
                    np.abs(conditional_residual.channel_data[first_channel_name]) ** 2
                )
            )
        )
        num_sample_calls = self.state.num_sample_calls
        print(
            f"[{self.name}] sample {num_sample_calls}: "
            f"residual RMS on {first_channel_name!r} = {residual_rms:.4e}"
        )
        self.state.num_sample_calls += 1
        return BlockState()


def _close_enough(actual: np.ndarray, expected: np.ndarray) -> bool:
    """Relative 1e-9 agreement, scaled by the published array's largest entry."""
    if actual.shape != expected.shape:
        return False
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    return bool(np.allclose(actual, expected, rtol=1e-9, atol=1e-9 * scale))


def _zeroed(data: L1Data) -> L1Data:
    """`data`'s grid with every channel value set to zero.

    `state_from_parameters` may read only grid metadata, so the round trip
    hands it a reference whose values would betray a block that reads them:
    `L2Data` rebuilds states from exactly such a reference.
    """
    return replace(
        data,
        channel_data={
            name: np.zeros_like(values) for name, values in data.channel_data.items()
        },
    )


def _check_round_trip(
    block: Block, entry: BlockState, reference: L1Data, where: str
) -> None:
    """Rebuild `entry` from its parameters on `reference` and compare the two."""
    rebuilt = block.state_from_parameters(entry.parameters, reference)
    if not isinstance(rebuilt, BlockState):
        raise TypeError(
            f"{block.name}.state_from_parameters must return a BlockState, got "
            f"{type(rebuilt).__name__}"
        )
    published, again = entry.template, rebuilt.template
    if (published is None) != (again is None) or (
        published is not None and again is not None and set(published) != set(again)
    ):
        raise ValueError(
            f"{block.name}.state_from_parameters does not reproduce the published "
            f"template at {where} (the channels differ): parameters must "
            "determine the template"
        )
    if published is not None and again is not None:
        for channel in published:
            if not _close_enough(np.asarray(again[channel]), published[channel]):
                raise ValueError(
                    f"{block.name}.state_from_parameters does not reproduce the "
                    f"published template in channel {channel!r} at {where}: "
                    "parameters must determine the template"
                )
    noise, noise_again = entry.covariance, rebuilt.covariance
    if (noise is None) != (noise_again is None) or (
        noise is not None
        and noise_again is not None
        and (
            noise_again.channel_names != noise.channel_names
            or noise_again.grid != noise.grid
            or not _close_enough(noise_again.blocks, noise.blocks)
        )
    ):
        raise ValueError(
            f"{block.name}.state_from_parameters does not reproduce the published "
            f"covariance at {where}: parameters must determine the covariance"
        )


def check_block(
    block_under_test: Block,
    observed_data: L1Data,
    num_cycles: int = 2,
    *,
    role: Role,
    noise_blocks: Sequence[Block] = (),
    random_seed: int | None = 0,
) -> None:
    """Run a block on a scratch Wheel and validate each returned BlockState.

    Registers each block in ``noise_blocks`` with ``role="noise"``, in order,
    then ``block_under_test`` with ``role``, which calls its ``initial_sample``
    on the observation grid, then calls ``sample`` for ``num_cycles``, passing
    accepted state into each call. Wheel validates the start and every sample
    alike: the signal grid, finiteness, the role's rules and any published
    Covariance. After registration (step 0) and after every cycle it also calls
    the block's ``state_from_parameters`` with the ledger entry's parameters and
    a zero-valued copy of the observation (grid metadata only, which is what
    ``L2Data`` supplies) and requires the result to reproduce the published
    template and covariance to a relative 1e-9. After the run it rebuilds the
    step-0 entry once more, so a rebuild that reads the block's own state
    instead of the parameters is caught as well. Otherwise it raises
    ``ValueError`` naming the block, the step and the channel: the published
    parameters must determine the published state.

    Args:
        block_under_test: Configured block whose exchange contract is checked.
            A block that must start at a chosen point is constructed with it.
        observed_data: L1 observations and shared run settings for the scratch run.
        num_cycles: Number of sampling cycles after registration; defaults to 2.
        role: ``"signal"`` or ``"noise"``, the role ``block_under_test`` is
            registered under.
        noise_blocks: Noise blocks registered before ``block_under_test``. A
            signal block that needs a covariance gets it from a
            ``FixedNoiseBlock`` here; a noise block under test receives their
            sum as its ``covariance``. Their covariances must be on
            ``observed_data``'s grid: to test with a frequency-grid covariance,
            pass ``transform(observed_data, "frequency")`` as ``observed_data``.
        random_seed: Seed for the scratch Wheel's NumPy generator, which also
            drives ``initial_sample``. Defaults to 0 for reproducibility; None
            requests nondeterministic initialization.

    This advances the supplied blocks' local state. It checks the exchange
    contract, not statistical correctness or block-local failure recovery.
    Test posterior recovery and independent campaigns separately;
    ``examples/toy_fit.py`` provides a pattern. ``check_block`` does not select
    a separate block domain: ``initial_sample`` and ``sample`` receive
    ``observed_data``'s grid.
    """
    wheel = Wheel(observed_data=observed_data, random_seed=random_seed)
    reference = _zeroed(observed_data)
    for noise_block in noise_blocks:
        wheel.add(block_to_register=noise_block, role="noise")
    wheel.add(block_to_register=block_under_test, role=role)
    first_entry = wheel.ledger[block_under_test.name]
    _check_round_trip(block_under_test, first_entry, reference, "step 0")

    def verify(cycle_index: int, campaign: Wheel) -> None:
        entry = campaign.ledger[block_under_test.name]
        _check_round_trip(block_under_test, entry, reference, f"step {cycle_index + 1}")

    wheel.run(num_cycles=num_cycles, on_cycle_complete=verify)
    _check_round_trip(
        block_under_test,
        first_entry,
        reference,
        f"step 0 after {num_cycles} cycles "
        "(the rebuild must not read the block's own state)",
    )
