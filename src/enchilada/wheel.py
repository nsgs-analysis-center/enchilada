"""Orchestrator-owned data, accepted physical estimates, and the two sums."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Literal

import numpy as np

from enchilada._sum_tree import NoiseSum, Signal, SignalSum
from enchilada.block import Block
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid
from enchilada.ledger import Ledger
from enchilada.translation import _data_on_grid, _state_on_grid, _target_grid

Role = Literal["signal", "noise"]
ROLES: tuple[Role, ...] = ("signal", "noise")
_BLOCK_INTERFACE = (
    "a Block needs name, initial_sample(conditional_residual, covariance, *, rng), "
    "sample(conditional_residual, covariance, current_block_state, *, rng) and "
    "state_from_parameters(parameters, reference_data)."
)


class Wheel:
    """Run blocks while owning their accepted physical estimates and shared data.

    The pristine observation is copied at construction. A second L1Data stores
    the full working residual, rebuilt after every accepted update. For block i,
    the input is d - sum(j != i, T_j); the block's own signal remains in its input.
    An absent TDI signal contributes zero, including when a block previously
    returned a signal. Ledger holds complete BlockState snapshots. Public data and
    covariance access returns copies. The ledger property returns a live handle;
    indexing and snapshot() copy its contents. Treat orbit resources as immutable.

    Every block is registered as a signal block or a noise block. Signal blocks
    publish templates, which Wheel sums into the residuals; noise blocks publish
    covariances, which Wheel sums into the campaign covariance. A signal block
    receives the sum over every noise block (None until one has published); a
    noise block receives the sum over the other noise blocks (None when it is
    alone) and its own last publication on current_block_state.covariance. A
    noise block must return a covariance on every call and may not return a
    template; a signal block may not return a covariance.

    Registration calls the block's `initial_sample` with the residual of the
    blocks registered so far on the block's grid, the covariance its role
    receives, and the campaign RNG, then adopts the returned state like a
    `sample` return. Each cycle calls sample with those inputs and the
    current_block_state from the ledger. Sampling receives the supplied RNG and
    independent input snapshots. Failed calls/validation/copying leave the
    ledger, both sums and the Wheel RNG unchanged. Blocks own their sampler
    state and diagnostics; Wheel does not roll back changes to those internals.
    Use separate block instances for separate campaigns. Earlier successful
    updates in a failed cycle remain committed. Exceptions in on_cycle_complete
    occur after that entire cycle has committed.

    Signals are stored on the observation grid. The covariance sum lives on the
    grid of the first noise publication: a noise block registered on another
    grid is refused at add, and a signal block on another grid is refused at
    add or run, each with ValueError naming both grids. A block registered with
    data_domain has its inputs and returned signal translated; a covariance is
    never translated.
    """

    def __init__(
        self,
        observed_data: L1Data,
        *,
        random_seed: int | None = None,
    ) -> None:
        """Initialize a campaign from observations.

        Args:
            observed_data: L1 observations and shared run settings. The Wheel
                makes independent pristine and working copies.
            random_seed: Seed for the campaign's NumPy generator. None requests
                nondeterministic initialization.

        The campaign has no covariance until a noise block publishes one at
        registration; a known noise model is a FixedNoiseBlock.
        """
        if not isinstance(observed_data, L1Data):
            raise TypeError("observed_data must be L1Data")
        for ch in observed_data.channel_names:
            if not np.isfinite(observed_data.channel_data[ch]).all():
                raise ValueError(
                    f"observed_data.channel_data[{ch!r}] has non-finite samples; "
                    "enchilada has no data-quality mask for gaps; "
                    "fill or trim them first"
                )
        self._observed_data = self._copy_data(observed_data)
        self._working_residual = self._copy_data(observed_data)
        self._ledger = Ledger()
        self._blocks: list[tuple[str, Block]] = []
        self._block_indices: dict[str, int] = {}
        self._block_domains: dict[str, Grid] = {}
        self._roles: dict[str, Role] = {}
        self._signal_sum = SignalSum()
        self._noise_sum = NoiseSum()
        self._covariance_grid: Grid | None = None
        self._rng = np.random.default_rng(random_seed)

    @staticmethod
    def _copy_data(data_to_copy: L1Data) -> L1Data:
        """Copy channel arrays, preserving immutable orbit data."""
        if data_to_copy.data_domain == "wdm":
            data_to_copy._validate_channel_data_lengths()
        return replace(
            data_to_copy,
            channel_data={
                ch: arr.copy() for ch, arr in data_to_copy.channel_data.items()
            },
        )

    @property
    def observed_data(self) -> L1Data:
        """A defensive copy of the pristine observation."""
        return self._copy_data(self._observed_data)

    @property
    def working_residual(self) -> L1Data:
        """A defensive copy of the current full residual."""
        return self._copy_data(self._working_residual)

    @property
    def covariance(self) -> Covariance | None:
        """The sum over every noise block's covariance, as an independent copy.

        None until a noise block has published. The sum lives on the grid of
        the first noise publication.
        """
        total = self._noise_sum.total
        if total is None:
            return None
        # Leaves passed Covariance.copy at adoption and sums were built by
        # Covariance.__add__ through the constructor: restore unvalidated.
        return deepcopy(total)

    @property
    def ledger(self) -> Ledger:
        """Live state store; indexing copies entries, snapshot() copies all entries."""
        return self._ledger

    def add(
        self,
        block_to_register: Block,
        *,
        role: Role,
        data_domain: str | None = None,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
    ) -> None:
        """Register a block under a role and adopt the start its initial_sample returns.

        role is "signal" or "noise" and is checked before anything else. A
        signal block publishes a template and never a covariance; a noise
        block publishes a covariance on every call and never a template.
        block_to_register must have a unique name and implement initial_sample,
        sample and state_from_parameters. Failed registration leaves the
        campaign unchanged: an exception from initial_sample, a return that is
        not a BlockState, an off-grid or non-finite template, a field the role
        forbids or lacks, a covariance on another grid than the block's, or a
        state that cannot be deep-copied leaves the ledger, the block list, the
        residual, both sums and the Wheel RNG as they were.

        initial_sample receives the current working residual (the observations
        minus the adopted starts of every block registered so far) on this
        block's grid, the covariance its role receives (the sum over every
        noise block for a signal block, the sum over the other noise blocks for
        a noise block; None while that sum is empty) and a copy of the campaign
        RNG that is committed with the state. Blocks therefore start in
        registration order: register the noise blocks before the blocks whose
        starts need their covariance. The returned state is validated and
        adopted exactly like a sample return: its template must be on this
        block's grid, and Wheel translates it to the observation grid for
        storage. A block that must start at a chosen point takes that point in
        its constructor and returns it from initial_sample.

        data_domain selects time, frequency or wdm for this block's residual,
        current state and published covariance. None keeps the observation grid.
        For WDM, specify either division count; the other follows from the
        observation length. Both counts must be even. An existing observation
        WDM grid is reused if neither count is supplied. The covariance sum
        lives on the grid of the first noise publication; a noise block on
        another grid is refused here, before initial_sample is called, and so
        is a signal block once a covariance has been published.
        """
        if role not in ROLES:
            raise ValueError(f"role must be 'signal' or 'noise', got {role!r}")
        observed = self._observed_data
        if data_domain is None:
            if num_frequency_divisions is not None or num_time_divisions is not None:
                raise ValueError("WDM division counts require data_domain='wdm'")
            block_grid = observed.grid
        else:
            block_grid = Grid(
                data_domain,
                observed.num_time_samples,
                observed.sample_rate_hz,
                _target_grid(
                    data_domain,
                    observed.num_time_samples,
                    observed.wdm_grid,
                    num_frequency_divisions,
                    num_time_divisions,
                ),
            )
        block_name = getattr(block_to_register, "name", None)
        if not isinstance(block_name, str) or not block_name:
            raise ValueError(
                f"block name must be a non-empty string, got {block_name!r}"
            )
        if block_name in self._ledger:
            raise ValueError(f"block name {block_name!r} already registered")
        if not callable(getattr(block_to_register, "sample", None)):
            raise TypeError(
                f"block {block_name!r} does not implement sample; {_BLOCK_INTERFACE}"
            )
        if not callable(getattr(block_to_register, "initial_sample", None)):
            raise TypeError(
                f"block {block_name!r} does not implement initial_sample; "
                f"{_BLOCK_INTERFACE}"
            )
        if not callable(getattr(block_to_register, "state_from_parameters", None)):
            raise TypeError(
                f"block {block_name!r} does not implement state_from_parameters; "
                f"{_BLOCK_INTERFACE}"
            )
        conditional_residual, covariance = self._block_inputs(
            block_name, block_grid, role=role, registered=False
        )
        candidate_rng = deepcopy(self._rng)
        new_block_state: object = block_to_register.initial_sample(
            conditional_residual=conditional_residual,
            covariance=covariance,
            rng=candidate_rng,
        )
        new_block_state = self._canonical_state(new_block_state, conditional_residual)
        self._adopt(
            block_name,
            new_block_state,
            "initial_sample",
            candidate_rng,
            role=role,
            block_to_register=block_to_register,
            block_grid=block_grid,
        )

    def _block_inputs(
        self, block_name: str, block_grid: Grid, *, role: Role, registered: bool
    ) -> tuple[L1Data, Covariance | None]:
        """Conditional residual on the block's grid, and the covariance its role gets.

        A registered block's residual leaves its own signal in; a block being
        registered has no signal yet, so it receives the working residual. A
        signal block receives the sum over every noise block; a noise block
        receives the sum over the other noise blocks, which is the whole sum
        while it is being registered because its own slot does not exist yet.
        """
        if self._covariance_grid is not None and self._covariance_grid != block_grid:
            if role == "noise":
                raise ValueError(
                    f"block {block_name!r} is registered on grid {block_grid} but "
                    f"the campaign covariance is on grid {self._covariance_grid}; "
                    "a covariance does not add across grids"
                )
            raise ValueError(
                f"block {block_name!r} is registered on grid {block_grid} but the "
                f"current covariance is on grid {self._covariance_grid}; a "
                "covariance does not translate between grids. Register the block "
                "on the covariance's grid, or have the noise block publish on the "
                "block's grid."
            )
        conditional_residual = (
            self.residual(exclude_block_name=block_name)
            if registered
            else self.working_residual
        )
        conditional_residual = _data_on_grid(conditional_residual, block_grid)
        if role == "noise" and registered:
            total = self._noise_sum.excluding(self._block_indices[block_name])
        else:
            total = self._noise_sum.total
        # Stored covariances were validated when published or summed: restore
        # an independent read-only copy without revalidating.
        covariance = None if total is None else deepcopy(total)
        return conditional_residual, covariance

    def _canonical_state(self, state: object, source_reference: L1Data) -> object:
        # A non-BlockState passes through; _adopt refuses it.
        return _state_on_grid(
            state, self._observed_data.grid, reference_data=source_reference
        )

    def run(
        self,
        num_cycles: int,
        on_cycle_complete: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Visit blocks in registration order; notify after each complete cycle.

        Cycle indices are local to this invocation. Multiple run calls continue
        the accepted ledger and RNG; run(0) does nothing.

        Args:
            num_cycles: Non-negative number of complete Gibbs cycles to attempt.
            on_cycle_complete: Optional callback receiving (cycle_index, wheel)
                after every block in a cycle has returned an accepted state.
                Indices start at zero for each run call. Partial cycles do not
                invoke the callback.
        """
        if (
            not isinstance(num_cycles, (int, np.integer))
            or isinstance(num_cycles, bool)
            or num_cycles < 0
        ):
            raise ValueError(
                f"num_cycles must be a non-negative integer, got {num_cycles!r}"
            )
        for cycle_index in range(num_cycles):
            for block_name, block in self._blocks:
                candidate_rng = deepcopy(self._rng)
                block_grid = self._block_domains[block_name]
                role = self._roles[block_name]
                conditional_residual, covariance = self._block_inputs(
                    block_name, block_grid, role=role, registered=True
                )
                current_block_state = self._ledger[block_name]
                current_block_state = _state_on_grid(
                    current_block_state, block_grid, reference_data=self._observed_data
                )
                new_block_state: object = block.sample(
                    conditional_residual=conditional_residual,
                    covariance=covariance,
                    current_block_state=current_block_state,
                    rng=candidate_rng,
                )
                new_block_state = self._canonical_state(
                    new_block_state, conditional_residual
                )
                self._adopt(
                    block_name,
                    new_block_state,
                    "sample",
                    candidate_rng,
                    role=role,
                    block_grid=block_grid,
                )
            if on_cycle_complete is not None:
                on_cycle_complete(cycle_index, self)

    def _form_residual(self, signal_sum: Signal) -> L1Data:
        """Subtract an aggregate signal from pristine observations."""
        channel_data = {}
        for ch, observed_channel_values in self._observed_data.channel_data.items():
            if signal_sum is None:
                residual_channel_values = observed_channel_values.copy()
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_channel_values = observed_channel_values - signal_sum[ch]
            if not np.isfinite(residual_channel_values).all():
                raise ValueError(f"residual in channel {ch!r} became non-finite")
            channel_data[ch] = residual_channel_values
        return replace(self._observed_data, channel_data=channel_data)

    def residual(self, exclude_block_name: str | None = None) -> L1Data:
        """Return a copied full residual or a block's conditional residual.

        exclude_block_name identifies a registered block whose signal is left
        in the observations. None subtracts every block's signal.
        """
        if exclude_block_name is None:
            return self.working_residual
        self._require_block_name(exclude_block_name)
        return self._form_residual(
            self._signal_sum.excluding(self._block_indices[exclude_block_name])
        )

    def contribution(self, block_name: str) -> dict[str, np.ndarray]:
        """Copy the registered block's summed TDI signal.

        block_name must identify a registered block. An absent contribution
        returns fresh zero arrays on the observation grid.
        """
        self._require_block_name(block_name)
        signal = self._ledger._states_by_block_name[block_name].template
        if signal is None:
            return {
                ch: np.zeros_like(arr)
                for ch, arr in self._observed_data.channel_data.items()
            }
        return {ch: arr.copy() for ch, arr in signal.items()}

    def _require_block_name(self, block_name: str) -> None:
        if block_name not in self._ledger:
            raise ValueError(
                f"unknown block {block_name!r}; registered: {sorted(self._ledger)}"
            )

    def _adopt(
        self,
        block_name: str,
        new_block_state: object,
        source_method_name: str,
        candidate_rng: np.random.Generator,
        *,
        role: Role,
        block_to_register: Block | None = None,
        block_grid: Grid,
    ) -> None:
        """Validate a candidate state and RNG before committing campaign state.

        source_method_name names the block method whose return is being adopted,
        for diagnostics. role decides which fields the state may and must carry.
        block_to_register is supplied only when adopting a new block's initial state.
        block_grid is the grid the block is registered on; a published covariance
        must be on it.
        """
        if not isinstance(new_block_state, BlockState):
            raise TypeError(
                f"{block_name}.{source_method_name} must return a BlockState, got "
                f"{type(new_block_state).__name__}"
            )
        # Snapshot BEFORE validation so subsequent use only touches owned state.
        candidate_block_state = deepcopy(new_block_state)
        candidate_block_state.__post_init__()
        signal = candidate_block_state.template
        if signal is not None:
            check_on_grid(
                signal,
                channel_names=self._observed_data.channel_names,
                num_time_samples=self._observed_data.num_time_samples,
                data_domain=self._observed_data.data_domain,
                wdm_grid=self._observed_data.wdm_grid,
                context_label=f"{block_name}.{source_method_name} template",
            )
            for ch, arr in signal.items():
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"{block_name}.{source_method_name} returned non-finite "
                        f"samples in channel {ch!r}"
                    )
        block_index = (
            len(self._blocks)
            if block_to_register is not None
            else self._block_indices[block_name]
        )
        candidate_noise_sum = self._noise_sum
        candidate_covariance_grid = self._covariance_grid
        if role == "signal":
            if candidate_block_state.covariance is not None:
                raise ValueError(
                    f"{block_name}.{source_method_name} returned a covariance but "
                    f"{block_name!r} is registered as a signal block"
                )
        else:
            if signal is not None:
                raise ValueError(
                    f"{block_name}.{source_method_name} returned a template but "
                    f"{block_name!r} is registered as a noise block"
                )
            if candidate_block_state.covariance is None:
                raise ValueError(
                    f"{block_name}.{source_method_name} must return a covariance "
                    f"because {block_name!r} is registered as a noise block"
                )
            # Revalidate: callers can edit arrays inside a frozen dataclass.
            # Unbound call: copy without trusting a subclass's overridden methods.
            candidate_covariance = Covariance.copy(candidate_block_state.covariance)
            if candidate_covariance.channel_names != tuple(
                self._observed_data.channel_names
            ):
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance channel_names "
                    f"{candidate_covariance.channel_names} do not match the data's "
                    f"{tuple(self._observed_data.channel_names)}"
                )
            if candidate_covariance.grid != block_grid:
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance is on grid "
                    f"{candidate_covariance.grid} but the block is registered on "
                    f"grid {block_grid}; publish the covariance on the grid of the "
                    "data the block received"
                )
            candidate_block_state = replace(
                candidate_block_state, covariance=candidate_covariance
            )
            candidate_noise_sum = self._noise_sum.with_covariance(
                block_index, candidate_covariance
            )
            candidate_covariance_grid = block_grid
        candidate_block_states = {
            **self._ledger._states_by_block_name,
            block_name: candidate_block_state,
        }
        candidate_signal_sum = self._signal_sum.with_signal(block_index, signal)
        candidate_residual = self._form_residual(candidate_signal_sum.total)
        candidate_block_indices = (
            {**self._block_indices, block_name: block_index}
            if block_to_register is not None
            else self._block_indices
        )
        next_blocks = (
            self._blocks
            if block_to_register is None
            else [*self._blocks, (block_name, block_to_register)]
        )
        candidate_block_domains = (
            self._block_domains
            if block_to_register is None
            else {**self._block_domains, block_name: block_grid}
        )
        candidate_roles = (
            self._roles
            if block_to_register is None
            else {**self._roles, block_name: role}
        )
        next_rng = deepcopy(candidate_rng)
        # All potentially failing work is done.
        self._ledger._states_by_block_name = candidate_block_states
        self._working_residual = candidate_residual
        self._signal_sum = candidate_signal_sum
        self._noise_sum = candidate_noise_sum
        self._covariance_grid = candidate_covariance_grid
        self._block_indices = candidate_block_indices
        self._block_domains = candidate_block_domains
        self._roles = candidate_roles
        self._rng = next_rng
        self._blocks = next_blocks
