"""A noise block that publishes one fixed covariance on every call."""

from copy import deepcopy
from typing import Any

import numpy as np

from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data


class FixedNoiseBlock:
    """Publish a known covariance as a noise block; nothing is sampled.

    ``wheel.add(FixedNoiseBlock("floor", covariance), role="noise")`` adds
    ``covariance`` to the campaign's noise sum at registration and republishes
    it every cycle, so signal blocks receive it and other noise blocks add
    their own models on top. The covariance must be on the grid the block is
    registered on (``data_domain`` at ``add``), like any noise publication.
    ``initial_sample`` and ``sample`` ignore their inputs and draw nothing.
    The constructor keeps an independent, revalidated copy.
    """

    def __init__(self, name: str, covariance: Covariance) -> None:
        if not isinstance(covariance, Covariance):
            raise TypeError(
                "FixedNoiseBlock covariance must be a Covariance, got "
                f"{type(covariance).__name__}"
            )
        self.name = name
        # Unbound call: copy without trusting a subclass's overridden methods.
        self._covariance = Covariance.copy(covariance)

    def _publish(self) -> BlockState:
        # deepcopy restores a validated Covariance read-only without revalidating.
        return BlockState(covariance=deepcopy(self._covariance))

    def initial_sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Publish the configured covariance; the inputs and ``rng`` are unused."""
        return self._publish()

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        """Publish the configured covariance; the block has no parameters."""
        return self._publish()

    def sample(
        self,
        conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState,
        *,
        rng: np.random.Generator,
    ) -> BlockState:
        """Publish the configured covariance again; inputs and ``rng`` are unused."""
        return self._publish()
