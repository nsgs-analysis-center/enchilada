"""The boundary between a model's local state and its published estimates."""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data


@runtime_checkable
class Block(Protocol):
    """A model that publishes physical estimates while owning its sampler state.

    Keep sampler continuation and diagnostics in a block-owned State object.
    BlockState contains only template, covariance, and physical parameters.
    `Wheel.add(block, role=...)` registers a block as a signal block or a noise
    block, calls `initial_sample` once with the block's starting inputs and
    adopts the returned state; `sample` then advances the latest accepted
    `current_block_state` and returns a complete new state. Both receive the
    conditional residual on the block's registered grid, the covariance the
    block's role receives, and the campaign RNG. A block that starts at a known
    point takes that point in its constructor and returns it from
    `initial_sample`; a block that starts from its prior draws from `rng`.
    `state_from_parameters` rebuilds the published state from its parameters
    alone, so a campaign file can store parameters only.

    `conditional_residual` is the observed data minus every OTHER block's current
    signal; this block's signal is still present. The block does not perform
    cross-block subtraction or add-back. `covariance` is a `Covariance` on the
    same grid as `conditional_residual`, or None: for a signal block the sum
    over every noise block, for a noise block the sum over the other noise
    blocks (its own last publication is on `current_block_state.covariance`).

    Signal and noise blocks share this interface; the role given at `Wheel.add`
    says which one a block is. A signal block publishes a template (None means
    zero contribution, replacing any previous signal) and never a covariance.
    A noise block publishes its complete covariance on every call and never a
    template; Wheel adds the noise blocks' covariances, so each block publishes
    only its own component. A joint model is two blocks.

    Each call, `initial_sample` included, receives independent snapshots of the
    conditional residual, covariance, and current state. Use the supplied RNG;
    Wheel commits its progress with the state. Treat the shared orbit as
    immutable.
    Wheel does not copy or roll back block-local state; blocks manage their own
    failure recovery. Use separate block instances for independent campaigns.
    Register a preferred data_domain with Wheel.add to receive matching inputs.
    Validate unsupported conventions explicitly.

    Static type checkers validate signatures; runtime protocol checks only test
    member presence. Neither establishes statistical correctness.
    """

    if TYPE_CHECKING:

        @property
        def name(self) -> str:
            """Stable registration name; immutable configurations are supported."""
            ...
    else:
        # Avoid an inherited property setter blocking dataclass construction
        # when an implementation explicitly subclasses this protocol.
        name: str

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Draw or construct this block's starting physical state on the input grid."""
        ...

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Advance the supplied state and return the complete new block state."""
        ...

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        """Rebuild this block's complete physical state from one parameter set.

        Return the template (signal block) or covariance (noise block) these
        parameters describe, with `parameters` on the result. Compute it from
        `parameters`, the block's constructor configuration and
        `reference_data`'s grid metadata only (domain, sample count, rate,
        start time, channel names, WDM grid and orbit); never from
        `reference_data`'s channel values, a random stream or state that
        `sample` changes. `reference_data` is on the block's registered grid,
        the grid `sample` receives. Read back from a campaign file, a float
        arrives as `np.float64` and a list as an ndarray, so avoid list
        operations and `isinstance(x, float)`. The simplest way to meet
        `check_block`'s 1e-9 round trip is for `initial_sample` and `sample` to
        end by calling this method. `L2Data` uses it to turn recorded
        parameters back into states and residuals; the Wheel never calls it
        during sampling.
        """
        ...
