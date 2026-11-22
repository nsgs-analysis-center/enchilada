"""The py.typed promise: consumer typos are static errors, not silent Any.

enchilada ships py.typed, so its annotations are load-bearing for consumers.
`L1Data.__getattr__` is therefore hidden from type checkers -- any
`__getattr__` would tell a checker that every attribute name exists. These
tests pin that, because the guarantee is documented and easy to regress
(annotating the method `-> Never` looks equivalent and is not: Never is
assignable to everything, so it suppresses the error instead of raising it).
"""

import importlib.util
import subprocess
import sys
import textwrap

import pytest

CONSUMER = """
from dataclasses import dataclass
import numpy as np
from typing import Any, assert_type
from enchilada import (
    Block, L1Data, Covariance, BlockState, Wheel, WDMGrid, Grid,
    FixedNoiseBlock, L2Data, transform,
)
from enchilada.testing import EchoBlock, check_block

r = L1Data(channel_data={"A": np.zeros(8)}, sample_rate_hz=1.0, channel_names=("A",),
              tdi_generation="2.0", physical_observable="strain")
assert_type(r.channel_names, tuple[str, ...])
inferred = L1Data({"E": np.zeros(8), "A": np.zeros(8)}, sample_rate_hz=1.0,
                  tdi_generation="2.0", physical_observable="strain")
assert_type(inferred.channel_names, tuple[str, ...])
explicit_none = L1Data(channel_data={"A": np.zeros(8)}, channel_names=None,
                       sample_rate_hz=1.0, tdi_generation="2.0",
                       physical_observable="strain")
assert_type(explicit_none.channel_names, tuple[str, ...])
assert_type(r.grid, Grid)
print(r.Tobs)      # correct spelling: must NOT error
print(r.Tobbs)     # typo in expression position
y: str = r.T_obs   # typo in assignment position

assert_type(transform(r, "frequency"), L1Data)
assert_type(transform(r, "wdm", num_frequency_divisions=2).wdm_grid, WDMGrid | None)
noise = Covariance.like(r, {"A": np.ones(8)})
assert_type(noise.copy(), Covariance)
assert_type(transform(noise, "wdm", num_frequency_divisions=2), Covariance)
assert_type(transform(BlockState(), "time"), BlockState)
recorded = Wheel(r, num_frequency_divisions=2, output_path="campaign.h5")
live = recorded.l2
assert live is not None
assert_type(live.num_steps, int)
assert_type(live.block_names, tuple[str, ...])
assert_type(live.home_grid, Grid)
with L2Data.open("campaign.h5", observed_data=r) as reader:
    assert_type(reader.parameters("noise"), dict[str, Any])
    assert_type(reader.chain("noise", "x"), np.ndarray)
recorded.close()
wheel = Wheel(r, num_frequency_divisions=2)
wheel.add(FixedNoiseBlock("noise", noise))
wheel.add(EchoBlock("wdm"), data_domain="wdm", num_time_divisions=2)
wheel.add(EchoBlock("plain"))
check_block(EchoBlock("checked"), r, num_frequency_divisions=2)
check_block(EchoBlock("floored"), r, num_frequency_divisions=2,
            noise_blocks=(FixedNoiseBlock("floor", noise),))
check_block(FixedNoiseBlock("probe", noise), r, num_frequency_divisions=2)
from enchilada import NoiseOverwrittenWarning   # removed: static error

class SampleOnly:
    name = "sample_only"
    role = "signal"

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

sample_only: Block = SampleOnly()   # no initial_sample: not a Block

class RoleLess:
    name = "roleless"

    def initial_sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None, *, rng: np.random.Generator,
    ) -> BlockState:
        return BlockState()

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        return BlockState()

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

roleless: Block = RoleLess()   # no role: not a Block

class NoRebuild:
    name = "no_rebuild"
    role = "signal"

    def initial_sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None, *, rng: np.random.Generator,
    ) -> BlockState:
        return BlockState()

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

no_rebuild: Block = NoRebuild()   # no state_from_parameters: not a Block

@dataclass(frozen=True)
class FrozenBlock:
    name: str = "frozen"
    role = "signal"

    def initial_sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None, *, rng: np.random.Generator,
    ) -> BlockState:
        return BlockState()

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        return BlockState()

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

frozen: Block = FrozenBlock()
wheel.add(FrozenBlock())
check_block(FrozenBlock(), r, num_frequency_divisions=2)

@dataclass(frozen=True)
class ExplicitBlock(Block):
    name: str
    role = "signal"

    def initial_sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None, *, rng: np.random.Generator,
    ) -> BlockState:
        return BlockState()

    def state_from_parameters(
        self, parameters: dict[str, Any], reference_data: L1Data
    ) -> BlockState:
        return BlockState()

    def sample(
        self, conditional_residual: L1Data,
        covariance: Covariance | None,
        current_block_state: BlockState, *, rng: np.random.Generator,
    ) -> BlockState:
        return current_block_state

explicit: Block = ExplicitBlock("explicit")
wheel.add(explicit)
"""


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="mypy not installed"
)
def test_consumer_typos_are_static_errors(tmp_path):
    f = tmp_path / "consumer.py"
    f.write_text(textwrap.dedent(CONSUMER))
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(f)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert 'has no attribute "Tobbs"' in out, out
    assert 'has no attribute "T_obs"' in out, out
    # exactly the two typos plus the removed NoiseOverwrittenWarning import --
    # the correct `r.Tobs` on the preceding line must not be flagged (an
    # over-broad ban would break legitimate access)
    assert out.count("[attr-defined]") == 3, out
    # ... and a class without initial_sample is not a Block
    assert (
        'Incompatible types in assignment (expression has type "SampleOnly", '
        'variable has type "Block")'
    ) in out, out
    # ... and neither is a class without a role ...
    assert (
        'Incompatible types in assignment (expression has type "RoleLess", '
        'variable has type "Block")'
    ) in out, out
    # ... or one without state_from_parameters ...
    assert (
        'Incompatible types in assignment (expression has type "NoRebuild", '
        'variable has type "Block")'
    ) in out, out
    # ... and the removed warning class is not importable
    assert 'has no attribute "NoiseOverwrittenWarning"' in out, out
    assert out.count("error:") == 6, out


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="mypy not installed"
)
def test_l1data_positional_settings_are_static_errors(tmp_path):
    consumer = tmp_path / "positional_consumer.py"
    consumer.write_text(
        "import numpy as np\nfrom enchilada import L1Data\n"
        'L1Data({"A": np.zeros(8)}, 1.0, ("A",), "2.0", "strain")\n'
    )
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(consumer)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert 'Too many positional arguments for "L1Data"' in out, out


@pytest.mark.skipif(
    importlib.util.find_spec("mypy") is None, reason="mypy not installed"
)
def test_l1data_wdm_grid_constructor_argument_is_a_static_error(tmp_path):
    consumer = tmp_path / "wdm_grid_consumer.py"
    consumer.write_text(
        "import numpy as np\nfrom enchilada import L1Data, WDMGrid\n"
        'L1Data({"A": np.zeros((3, 2))}, sample_rate_hz=1.0,\n'
        '       tdi_generation="2.0", physical_observable="strain",\n'
        '       data_domain="wdm", wdm_grid=WDMGrid(\n'
        "           num_frequency_divisions=2, num_time_divisions=2))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", str(consumer)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert 'Unexpected keyword argument "wdm_grid" for "L1Data"' in out, out


def test_runtime_attribute_hints_still_work():
    import numpy as np

    from enchilada import L1Data

    r = L1Data(
        channel_data={"A": np.zeros(8)},
        sample_rate_hz=1.0,
        channel_names=("A",),
        tdi_generation="2.0",
        physical_observable="strain",
    )
    assert hasattr(r, "Tobs") and not hasattr(r, "Tobbs")
    with pytest.raises(AttributeError, match="did you mean 'Tobs'"):
        _ = r.T_obs


def test_block_protocol_is_runtime_checkable():
    from enchilada import Block, BlockState
    from enchilada.testing import EchoBlock

    assert isinstance(EchoBlock("e"), Block)

    class NotABlock:
        pass

    assert not isinstance(NotABlock(), Block)

    class SampleOnly:
        name = "s"

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    assert not isinstance(SampleOnly(), Block)

    class RoleLess:
        name = "r"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return BlockState()

        def state_from_parameters(self, parameters, reference_data):
            return BlockState()

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    assert not isinstance(RoleLess(), Block)

    # a class without state_from_parameters is not a Block either
    class NoRebuild:
        name = "n"
        role = "signal"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return None

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    assert not isinstance(NoRebuild(), Block)


def test_frozen_dataclass_can_explicitly_inherit_block():
    from dataclasses import dataclass

    import numpy as np

    from enchilada import Block, BlockState, L1Data, Wheel

    @dataclass(frozen=True)
    class ExplicitBlock(Block):
        name: str
        role = "signal"

        def initial_sample(self, conditional_residual, covariance, *, rng):
            return BlockState(parameters={"value": 1})

        def state_from_parameters(self, parameters, reference_data):
            return BlockState(parameters=dict(parameters))

        def sample(self, conditional_residual, covariance, current_block_state, *, rng):
            return current_block_state

    block = ExplicitBlock("explicit")
    assert isinstance(block, Block)
    observations = L1Data(
        channel_data={"A": np.zeros(4)},
        sample_rate_hz=1.0,
        channel_names=("A",),
        tdi_generation="2.0",
        physical_observable="strain",
    )
    wheel = Wheel(observations, num_frequency_divisions=2)
    wheel.add(block)
    wheel.run(1)
    assert wheel.ledger["explicit"].parameters == {"value": 1}


def test_replace_is_re_exported():
    import dataclasses

    import enchilada

    assert enchilada.replace is dataclasses.replace
    assert "replace" in enchilada.__all__


def test_public_surface_is_pinned():
    """__all__ governs re-export for consumers running mypy --strict, so a
    dropped entry silently becomes a type error downstream while CI stays
    green. Pin it literally."""
    import enchilada

    assert enchilada.__all__ == [
        "Block",
        "Covariance",
        "FixedNoiseBlock",
        "Grid",
        "Ledger",
        "NumericOrbit",
        "Orbit",
        "L1Data",
        "L2Data",
        "BlockState",
        "Wheel",
        "WDMGrid",
        "__version__",
        "decimate",
        "load_mojito_l1",
        "replace",
        "transform",
    ]
    for name in enchilada.__all__:
        assert hasattr(enchilada, name), name
