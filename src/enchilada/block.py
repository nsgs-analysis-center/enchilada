"""The boundary between a model's local state and its published estimates."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data


@runtime_checkable
class Block(Protocol):
    """A model that publishes physical estimates while owning its sampler state.

    Keep sampler continuation and diagnostics in a block-owned State object.
    BlockState contains only template, covariance, and physical parameters.
    A block declares what it publishes in its `role` attribute, "signal" or
    "noise", usually one class-level line (`role = "noise"`); `Wheel.add`
    reads it once at registration. `add` calls `initial_sample` once with the
    block's starting inputs and adopts the returned state; `sample` then
    advances the latest accepted `current_block_state` and returns a complete
    new state. Both receive the conditional residual on the block's registered
    grid, the covariance the block's role receives, and the campaign RNG. A
    block that starts at a known point takes that point in its constructor and
    returns it from `initial_sample`; a block that starts from its prior draws
    from `rng`.

    `conditional_residual` is the observed data minus every OTHER block's current
    signal; this block's signal is still present. The block does not perform
    cross-block subtraction or add-back. `covariance` is a `Covariance` on the
    same grid as `conditional_residual`, or None: for a signal block the sum
    over every noise block, for a noise block the sum over the other noise
    blocks (its own last publication is on `current_block_state.covariance`).
    Wheel keeps the campaign on a WDM home grid and brings these inputs to the
    block's grid: the data by exact transforms, the covariance by
    `transform(covariance, ...)`, which treats every covariance as diagonal in
    its grid index.

    Signal and noise blocks share this interface; `role` says which one a
    block is. A signal block publishes a template (None means zero
    contribution, replacing any previous signal) and never a covariance. A
    noise block publishes its complete covariance on every call, on the grid
    of the data it received, and never a template; Wheel adds the noise
    blocks' covariances, so each block publishes only its own component. A
    joint model is two blocks.

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

        @property
        def role(self) -> str:
            """``"signal"`` or ``"noise"``: what this block publishes."""
            ...
    else:
        # Avoid an inherited property setter blocking dataclass construction
        # when an implementation explicitly subclasses this protocol.
        name: str
        role: str

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
