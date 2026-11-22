"""Orchestrator-owned data, accepted physical estimates, and the two sums."""

import contextlib
import os
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Self

import numpy as np

from enchilada._sum_tree import NoiseSum, Signal, SignalSum
from enchilada._transforms import transform_channels
from enchilada.block import ROLES as ROLES
from enchilada.block import Block
from enchilada.block import Role as Role
from enchilada.block_state import BlockState, check_on_grid
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid, resolve_wdm_grid
from enchilada.l2 import L2Data, _Encoded, _Recorder
from enchilada.ledger import Ledger
from enchilada.translation import _target_grid, transform

_PROTOCOL = (
    "a Block needs name, role ('signal' or 'noise'), "
    "initial_sample(conditional_residual, covariance, *, rng), "
    "sample(conditional_residual, covariance, current_block_state, *, rng) and "
    "state_from_parameters(parameters, reference_data)."
)


class Wheel:
    """Run blocks while owning their accepted physical estimates and shared data.

    The campaign lives on one WDM grid, the home grid, chosen at construction:
    time or frequency observations are transformed onto it once, exactly, and
    WDM observations bring their own. The pristine observation is kept twice,
    as it arrived (for `observed_data`) and on the home grid (the values every
    residual is built from). A second home-grid L1Data stores the full working
    residual, rebuilt after every accepted update. For block i, the input is
    d - sum(j != i, T_j); the block's own signal remains in its input. An
    absent TDI signal contributes zero, including when a block previously
    returned a signal. Ledger holds complete BlockState snapshots: a signal
    block's template on the home grid, a noise block's covariance as it was
    published. The public data views (`observed_data`, `working_residual`,
    `residual`, `contribution`) come back on the arrival grid through one exact
    inverse transform; `covariance` is on the home grid. Public data and
    covariance access returns copies. The ledger property returns a live
    handle; indexing and snapshot() copy its contents. Treat orbit resources as
    immutable.

    Every block declares its role in the attribute `role`, "signal" or
    "noise", read once at `add`. Signal blocks publish templates, which Wheel
    sums into the residuals; noise blocks publish covariances, which Wheel sums
    into the campaign covariance. A signal block receives the sum over every
    noise block (None until one has published); a noise block receives the sum
    over the other noise blocks (None when it is alone) and its own last
    publication on current_block_state.covariance. A noise block must return a
    covariance on every call and may not return a template; a signal block may
    not return a covariance.

    A block is served on the grid it asks for at `add`: its residual and
    current state by exact data transforms, the covariance it receives by
    `transform(covariance, ...)`, which reads every covariance as a diagonal
    spectral model (see `transform`). A template it returns is transformed
    onto the home grid and a covariance it returns is translated onto the home
    grid before entering the sums; the ledger keeps the covariance as
    published. A block on the home grid pays no transform. The diagonal
    approximation behind the covariance translation is the user's to assess;
    Wheel does not restrict grid choices on its account.

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

    With `output_path` the Wheel records the campaign: every block's parameters
    at registration (the block's entry 0) and after each complete cycle,
    appended to an HDF5 file and flushed before `on_cycle_complete` runs;
    `wheel.l2` reads it live and `L2Data.open` afterwards. The file also holds
    the observation's grid and the home grid, and `L2Data` rebuilds templates,
    the covariance sum and the residual on the home grid, as the Wheel keeps
    them. Parameters must then be recordable (numeric scalars or arrays; key
    set, dtype kind and dimension count fixed at registration; any shape per
    step); a state that is not is refused like an off-grid template. A partial
    cycle records nothing (a block added after one draws its start against
    in-memory states the file never holds, so `residual` at that step is not
    the residual the block saw). If a write fails, the campaign in memory is
    intact and one record ahead, the file is consistent up to its last step,
    and the Wheel refuses further add and run. `close()` or a `with` block
    closes the file.
    """

    def __init__(
        self,
        observed_data: L1Data,
        *,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
        random_seed: int | None = None,
        output_path: str | os.PathLike[str] | None = None,
    ) -> None:
        """Initialize a campaign from observations on a WDM home grid.

        Args:
            observed_data: L1 observations and shared run settings. The Wheel
                keeps an independent arrival-grid copy and a home-grid copy.
            num_frequency_divisions: Row count of the home grid. Time or
                frequency observations need this or num_time_divisions; WDM
                observations supply their own grid, and counts given here
                must agree with it.
            num_time_divisions: Column count of the home grid. The missing
                count follows from the observation length; both counts must
                be positive even integers whose product is the sample count.
            random_seed: Seed for the campaign's NumPy generator. None requests
                nondeterministic initialization.
            output_path: Where to record the campaign: an HDF5 file the Wheel
                creates now (an existing file raises FileExistsError) and
                appends to at every registration and after every complete
                cycle. None records nothing.

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
        arrival = self._copy_data(observed_data)
        counts_given = (
            num_frequency_divisions is not None or num_time_divisions is not None
        )
        if arrival.data_domain == "wdm":
            if counts_given:
                requested = resolve_wdm_grid(
                    arrival.num_time_samples,
                    num_frequency_divisions=num_frequency_divisions,
                    num_time_divisions=num_time_divisions,
                )
                if requested != arrival.wdm_grid:
                    raise ValueError(
                        f"observed_data is on WDM grid {arrival.wdm_grid}, which "
                        f"is the home grid, but {requested} was requested; omit "
                        "the division counts or pass ones that agree"
                    )
            home = self._copy_data(arrival)
        else:
            if not counts_given:
                raise ValueError(
                    f"observed_data is on data_domain={arrival.data_domain!r}; "
                    "the campaign lives on a WDM home grid, so pass "
                    "num_frequency_divisions or num_time_divisions to choose it"
                )
            home = transform(
                arrival,
                "wdm",
                num_frequency_divisions=num_frequency_divisions,
                num_time_divisions=num_time_divisions,
            )
        self._arrival_data = arrival
        self._home_data = home
        self._working_residual = self._copy_data(home)
        self._ledger = Ledger()
        self._blocks: list[tuple[str, Block]] = []
        self._block_indices: dict[str, int] = {}
        self._block_domains: dict[str, Grid] = {}
        self._roles: dict[str, Role] = {}
        self._signal_sum = SignalSum()
        self._noise_sum = NoiseSum()
        self._rng = np.random.default_rng(random_seed)
        self._recorder: _Recorder | None = None
        self._l2: L2Data | None = None
        self._pending: dict[str, _Encoded] = {}
        if output_path is not None:
            recorder = _Recorder.create(output_path, arrival, home_grid=home.grid)
            try:
                self._l2 = L2Data(recorder.file, observed_data=arrival, owns_file=False)
            except BaseException:
                recorder.close()
                with contextlib.suppress(OSError):
                    os.remove(output_path)
                raise
            self._recorder = recorder

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

    @staticmethod
    def _frequency_divisions(grid: Grid) -> int | None:
        if grid.wdm_grid is None:
            return None
        return grid.wdm_grid.num_frequency_divisions

    @property
    def home_grid(self) -> Grid:
        """The WDM grid the campaign lives on."""
        return self._home_data.grid

    @property
    def observed_data(self) -> L1Data:
        """A defensive copy of the pristine observation, as it arrived."""
        return self._copy_data(self._arrival_data)

    @property
    def working_residual(self) -> L1Data:
        """A copy of the current full residual on the arrival grid."""
        return self._on_grid(self._working_residual, self._arrival_data.grid)

    @property
    def covariance(self) -> Covariance | None:
        """The sum over every noise block's covariance, on the home grid.

        An independent copy; None until a noise block has published.
        """
        total = self._noise_sum.total
        if total is None:
            return None
        # Leaves were validated at adoption and sums were built by
        # Covariance.__add__ through the constructor: restore unvalidated.
        return deepcopy(total)

    @property
    def ledger(self) -> Ledger:
        """Live state store; indexing copies entries, snapshot() copies all entries."""
        return self._ledger

    @property
    def l2(self) -> L2Data | None:
        """The live view of the campaign file, or None when no output_path was given."""
        return self._l2

    def close(self) -> None:
        """Close the campaign file. Idempotent; a no-op without an output_path.

        Afterwards `add` and `run` raise ValueError; read the finished file with
        `L2Data.open`.
        """
        if self._recorder is not None:
            self._recorder.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _require_recordable(self) -> None:
        if self._recorder is not None:
            self._recorder.check_appendable(len(self._blocks))

    def _on_grid(self, home_data: L1Data, grid: Grid) -> L1Data:
        """Home-grid data on `grid`: an independent copy or one exact transform."""
        if grid == self._home_data.grid:
            return self._copy_data(home_data)
        return transform(
            home_data,
            grid.data_domain,
            num_frequency_divisions=self._frequency_divisions(grid),
        )

    def add(
        self,
        block_to_register: Block,
        *,
        data_domain: str | None = None,
        num_frequency_divisions: int | None = None,
        num_time_divisions: int | None = None,
    ) -> None:
        """Register a block under its declared role and adopt its initial_sample.

        block_to_register must have a unique name, a `role` attribute that is
        "signal" or "noise", and implement initial_sample, sample and
        state_from_parameters. The role is read here, once. A signal block
        publishes a template and never a covariance; a noise block publishes a
        covariance on every call and never a template. Failed registration
        leaves the ledger, the block list, the residual, both sums, the Wheel
        RNG and the file as they were. It fails on: an exception from
        initial_sample; a return that is not a BlockState; an off-grid or
        non-finite template; a field the role forbids or lacks; a covariance on
        another grid than the block's; a covariance that does not translate
        onto the home grid; a state that cannot be deep-copied; and, when
        recording, parameters that cannot be recorded or a name that cannot
        name an HDF5 group (checked before initial_sample).

        initial_sample receives the current working residual (the observations
        minus the adopted starts of every block registered so far) on this
        block's grid, the covariance its role receives (the sum over every
        noise block for a signal block, the sum over the other noise blocks for
        a noise block; None while that sum is empty; translated to this block's
        grid when that is not the home grid) and a copy of the campaign RNG
        that is committed with the state. Blocks therefore start in
        registration order: register the noise blocks before the blocks whose
        starts need their covariance. The returned state is validated and
        adopted exactly like a sample return: its template must be on this
        block's grid, and Wheel transforms it onto the home grid for storage;
        its covariance must be on this block's grid, and Wheel translates it
        onto the home grid for the sum while the ledger keeps the original. A
        block that must start at a chosen point takes that point in its
        constructor and returns it from initial_sample.

        data_domain selects time, frequency or wdm for this block's residual,
        current state and published covariance. None means the arrival grid of
        the observations. "wdm" without division counts means the home grid,
        which costs no transform; with counts that differ from the home grid's
        it is another WDM grid, and the other count follows from the
        observation length (both must be even).
        """
        self._require_recordable()
        block_grid = self._block_grid(
            data_domain, num_frequency_divisions, num_time_divisions
        )
        block_name = getattr(block_to_register, "name", None)
        if not isinstance(block_name, str) or not block_name:
            raise ValueError(
                f"block name must be a non-empty string, got {block_name!r}"
            )
        if self._recorder is not None:
            self._recorder.check_block_name(block_name)
        if block_name in self._ledger:
            raise ValueError(f"block name {block_name!r} already registered")
        role = self._declared_role(block_to_register, block_name)
        for method_name in ("sample", "initial_sample", "state_from_parameters"):
            if not callable(getattr(block_to_register, method_name, None)):
                raise TypeError(
                    f"block {block_name!r} does not implement {method_name}; "
                    f"{_PROTOCOL}"
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
        self._adopt(
            block_name,
            new_block_state,
            "initial_sample",
            candidate_rng,
            role=role,
            block_to_register=block_to_register,
            block_grid=block_grid,
        )

    def _block_grid(
        self,
        data_domain: str | None,
        num_frequency_divisions: int | None,
        num_time_divisions: int | None,
    ) -> Grid:
        """The grid a block asks for: the arrival grid, or one on the home record."""
        if data_domain is None:
            if num_frequency_divisions is not None or num_time_divisions is not None:
                raise ValueError("WDM division counts require data_domain='wdm'")
            return self._arrival_data.grid
        home = self._home_data.grid
        return Grid(
            data_domain,
            home.num_time_samples,
            home.sample_rate_hz,
            _target_grid(
                data_domain,
                home.num_time_samples,
                home.wdm_grid,
                num_frequency_divisions,
                num_time_divisions,
            ),
        )

    @staticmethod
    def _declared_role(block_to_register: Block, block_name: str) -> Role:
        """The block's `role` attribute, narrowed to a Role."""
        declared = getattr(block_to_register, "role", None)
        if declared is None:
            raise TypeError(f"block {block_name!r} declares no role; {_PROTOCOL}")
        if declared not in ROLES:
            raise ValueError(
                f"block {block_name!r} declares role {declared!r}; "
                "a role is 'signal' or 'noise'"
            )
        return "signal" if declared == "signal" else "noise"

    def _block_inputs(
        self, block_name: str, block_grid: Grid, *, role: Role, registered: bool
    ) -> tuple[L1Data, Covariance | None]:
        """Conditional residual on the block's grid, and the covariance its role gets.

        A registered block's residual leaves its own signal in; a block being
        registered has no signal yet, so it receives the working residual. A
        signal block receives the sum over every noise block; a noise block
        receives the sum over the other noise blocks, which is the whole sum
        while it is being registered because its own slot does not exist yet.
        Both are formed on the home grid and brought to the block's grid once:
        the residual by an exact transform, the covariance by translation.
        """
        residual_on_home = (
            self._form_residual(
                self._signal_sum.excluding(self._block_indices[block_name])
            )
            if registered
            else self._working_residual
        )
        conditional_residual = self._on_grid(residual_on_home, block_grid)
        if role == "noise" and registered:
            total = self._noise_sum.excluding(self._block_indices[block_name])
        else:
            total = self._noise_sum.total
        if total is None:
            covariance = None
        elif block_grid == self._home_data.grid:
            # Stored covariances were validated when published or summed:
            # restore an independent read-only copy without revalidating.
            covariance = deepcopy(total)
        else:
            try:
                covariance = transform(
                    total,
                    block_grid.data_domain,
                    num_frequency_divisions=self._frequency_divisions(block_grid),
                )
            except ValueError as error:
                raise ValueError(
                    f"covariance for block {block_name!r} does not translate onto "
                    f"its grid {block_grid}: {error}"
                ) from error
        return conditional_residual, covariance

    def run(
        self,
        num_cycles: int,
        on_cycle_complete: Callable[[int, "Wheel"], None] | None = None,
    ) -> None:
        """Visit blocks in registration order; notify after each complete cycle.

        Cycle indices are local to this invocation. Multiple run calls continue
        the accepted ledger and RNG; run(0) does nothing (a closed recording
        still refuses).

        Args:
            num_cycles: Non-negative number of complete Gibbs cycles to attempt.
            on_cycle_complete: Optional callback receiving (cycle_index, wheel)
                after every block in a cycle has returned an accepted state.
                Indices start at zero for each run call. Partial cycles do not
                invoke the callback.

        When recording, the cycle's entries are appended to the file before the
        callback runs.
        """
        if (
            not isinstance(num_cycles, (int, np.integer))
            or isinstance(num_cycles, bool)
            or num_cycles < 0
        ):
            raise ValueError(
                f"num_cycles must be a non-negative integer, got {num_cycles!r}"
            )
        self._require_recordable()
        for cycle_index in range(num_cycles):
            self._require_recordable()
            self._pending = {}
            for block_name, block in self._blocks:
                candidate_rng = deepcopy(self._rng)
                block_grid = self._block_domains[block_name]
                role = self._roles[block_name]
                conditional_residual, covariance = self._block_inputs(
                    block_name, block_grid, role=role, registered=True
                )
                current_block_state = self._ledger[block_name]
                if block_grid != self._home_data.grid:
                    current_block_state = transform(
                        current_block_state,
                        block_grid.data_domain,
                        reference_data=self._home_data,
                        num_frequency_divisions=self._frequency_divisions(block_grid),
                    )
                new_block_state: object = block.sample(
                    conditional_residual=conditional_residual,
                    covariance=covariance,
                    current_block_state=current_block_state,
                    rng=candidate_rng,
                )
                self._adopt(
                    block_name,
                    new_block_state,
                    "sample",
                    candidate_rng,
                    role=role,
                    block_grid=block_grid,
                )
            if self._recorder is not None:
                self._recorder.append_step(self._pending)
            if on_cycle_complete is not None:
                on_cycle_complete(cycle_index, self)

    def _form_residual(self, signal_sum: Signal) -> L1Data:
        """Subtract an aggregate signal from the home-grid observations."""
        channel_data = {}
        for ch, observed_channel_values in self._home_data.channel_data.items():
            if signal_sum is None:
                residual_channel_values = observed_channel_values.copy()
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    residual_channel_values = observed_channel_values - signal_sum[ch]
            if not np.isfinite(residual_channel_values).all():
                raise ValueError(f"residual in channel {ch!r} became non-finite")
            channel_data[ch] = residual_channel_values
        return replace(self._home_data, channel_data=channel_data)

    def residual(self, exclude_block_name: str | None = None) -> L1Data:
        """Return a full or conditional residual on the arrival grid, as a copy.

        exclude_block_name identifies a registered block whose signal is left
        in the observations. None subtracts every block's signal.
        """
        if exclude_block_name is None:
            return self.working_residual
        self._require_block_name(exclude_block_name)
        return self._on_grid(
            self._form_residual(
                self._signal_sum.excluding(self._block_indices[exclude_block_name])
            ),
            self._arrival_data.grid,
        )

    def contribution(self, block_name: str) -> dict[str, np.ndarray]:
        """Copy the registered block's summed TDI signal on the arrival grid.

        block_name must identify a registered block. An absent contribution
        returns fresh zero arrays on the arrival grid.
        """
        self._require_block_name(block_name)
        signal = self._ledger._states_by_block_name[block_name].template
        arrival = self._arrival_data
        if signal is None:
            return {ch: np.zeros_like(arr) for ch, arr in arrival.channel_data.items()}
        home = self._home_data
        if arrival.grid == home.grid:
            return {ch: arr.copy() for ch, arr in signal.items()}
        return transform_channels(
            signal,
            source_domain="wdm",
            target_domain=arrival.data_domain,
            num_time_samples=home.num_time_samples,
            sample_rate_hz=home.sample_rate_hz,
            source_wdm_grid=home.wdm_grid,
            target_wdm_grid=arrival.wdm_grid,
        )

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
        block_grid is the grid the block is registered on: a published template
        or covariance must be on it, and is brought onto the home grid here.
        When recording, parameters are encoded before the commit and written
        after it.
        """
        if not isinstance(new_block_state, BlockState):
            raise TypeError(
                f"{block_name}.{source_method_name} must return a BlockState, got "
                f"{type(new_block_state).__name__}"
            )
        # Snapshot BEFORE validation so subsequent use only touches owned state.
        candidate_block_state = deepcopy(new_block_state)
        candidate_block_state.__post_init__()
        home = self._home_data
        signal = candidate_block_state.template
        if signal is not None:
            check_on_grid(
                signal,
                channel_names=home.channel_names,
                num_time_samples=home.num_time_samples,
                data_domain=block_grid.data_domain,
                wdm_grid=block_grid.wdm_grid,
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
        if role == "signal":
            if candidate_block_state.covariance is not None:
                raise ValueError(
                    f"{block_name}.{source_method_name} returned a covariance but "
                    f"{block_name!r} declares role 'signal'"
                )
        else:
            if signal is not None:
                raise ValueError(
                    f"{block_name}.{source_method_name} returned a template but "
                    f"{block_name!r} declares role 'noise'"
                )
            if candidate_block_state.covariance is None:
                raise ValueError(
                    f"{block_name}.{source_method_name} must return a covariance "
                    f"because {block_name!r} declares role 'noise'"
                )
            # Revalidate: callers can edit arrays inside a frozen dataclass.
            # Unbound call: copy without trusting a subclass's overridden methods.
            candidate_covariance = Covariance.copy(candidate_block_state.covariance)
            if candidate_covariance.channel_names != tuple(home.channel_names):
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance channel_names "
                    f"{candidate_covariance.channel_names} do not match the data's "
                    f"{tuple(home.channel_names)}"
                )
            if candidate_covariance.grid != block_grid:
                raise ValueError(
                    f"{block_name}.{source_method_name} covariance is on grid "
                    f"{candidate_covariance.grid} but the block is registered on "
                    f"grid {block_grid}; publish the covariance on the grid of the "
                    "data the block received"
                )
            if block_grid == home.grid:
                home_covariance = candidate_covariance
            else:
                try:
                    home_covariance = transform(
                        candidate_covariance,
                        "wdm",
                        num_frequency_divisions=self._frequency_divisions(home.grid),
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{block_name}.{source_method_name} covariance does not "
                        f"translate onto the home grid {home.grid}: {error}"
                    ) from error
            # The ledger keeps the publication as made; the sum takes it on home.
            candidate_block_state = replace(
                candidate_block_state, covariance=candidate_covariance
            )
            candidate_noise_sum = self._noise_sum.with_covariance(
                block_index, home_covariance
            )
        if signal is not None and block_grid != home.grid:
            signal = transform_channels(
                signal,
                source_domain=block_grid.data_domain,
                target_domain="wdm",
                num_time_samples=home.num_time_samples,
                sample_rate_hz=home.sample_rate_hz,
                source_wdm_grid=block_grid.wdm_grid,
                target_wdm_grid=home.wdm_grid,
            )
            candidate_block_state = replace(candidate_block_state, template=signal)
        encoded = self._encode_for_file(
            block_name,
            source_method_name,
            candidate_block_state.parameters,
            registering=block_to_register is not None,
        )
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
        # Validation is done; commit memory, then write the file. A write error
        # leaves memory one record ahead and latches the recorder.
        self._ledger._states_by_block_name = candidate_block_states
        self._working_residual = candidate_residual
        self._signal_sum = candidate_signal_sum
        self._noise_sum = candidate_noise_sum
        self._block_indices = candidate_block_indices
        self._block_domains = candidate_block_domains
        self._roles = candidate_roles
        self._rng = next_rng
        self._blocks = next_blocks
        self._record_adopted(
            block_name,
            encoded,
            role=role,
            index=block_index,
            grid=block_grid,
            registering=block_to_register is not None,
        )

    def _encode_for_file(
        self,
        block_name: str,
        source_method_name: str,
        parameters: dict[str, object],
        *,
        registering: bool,
    ) -> _Encoded | None:
        """Validate parameters against the recordable rules; None when not recording."""
        if self._recorder is None:
            return None
        return self._recorder.encode(
            block_name, source_method_name, parameters, registering=registering
        )

    def _record_adopted(
        self,
        block_name: str,
        encoded: _Encoded | None,
        *,
        role: Role,
        index: int,
        grid: Grid,
        registering: bool,
    ) -> None:
        """Write a registration now, or queue a cycle's entry for append_step."""
        if encoded is None or self._recorder is None:
            return
        if registering:
            self._recorder.add_block(
                block_name, role=role, index=index, grid=grid, encoded=encoded
            )
        else:
            self._pending[block_name] = encoded
