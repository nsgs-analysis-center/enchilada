from dataclasses import replace  # convenience for frozen data/state snapshots
from importlib.metadata import PackageNotFoundError, version

from enchilada.block import Block
from enchilada.block_state import BlockState
from enchilada.covariance import Covariance
from enchilada.data import L1Data
from enchilada.domains import Grid, WDMGrid
from enchilada.fixed_noise import FixedNoiseBlock
from enchilada.l2 import L2Data
from enchilada.ledger import Ledger
from enchilada.mojito import load_mojito_l1
from enchilada.orbits import NumericOrbit, Orbit
from enchilada.resample import decimate
from enchilada.translation import transform
from enchilada.wheel import Wheel

try:
    __version__ = version("enchilada")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0+unknown"

__all__ = [
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
