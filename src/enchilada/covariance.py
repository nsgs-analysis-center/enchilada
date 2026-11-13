"""Noise covariance: one stacked Hermitian array on a grid, read by channel name."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from enchilada.domains import Grid

if TYPE_CHECKING:
    from enchilada.data import L1Data

AET_FROM_XYZ = np.array(
    [
        [-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
        [1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)],
    ]
)
"""Rows A, E, T from columns X, Y, Z: the orthonormal LDC combination."""

BASES = {"XYZ": ("X", "Y", "Z"), "AET": ("A", "E", "T")}


def split_pair_name(
    name: str, channel_names: tuple[str, ...]
) -> tuple[str, str] | None:
    """Resolve ``"X"`` to ``("X", "X")`` and ``"XY"`` to ``("X", "Y")``.

    Returns None when ``name`` is neither a channel nor a concatenation of two
    channels. Raises ValueError when more than one split works.
    """
    if name in channel_names:
        return (name, name)
    candidates = [
        (name[:cut], name[cut:])
        for cut in range(1, len(name))
        if name[:cut] in channel_names and name[cut:] in channel_names
    ]
    if len(candidates) > 1:
        raise ValueError(
            f"{name!r} is ambiguous between channel pairs {candidates}; "
            "use a (first, second) tuple or pair(first, second)"
        )
    return candidates[0] if candidates else None


def _validated_channel_names(names: object) -> tuple[str, ...]:
    if not isinstance(names, (tuple, list)):
        raise TypeError("channel_names must be a tuple of channel names")
    channels = tuple(names)
    for name in channels:
        if not isinstance(name, str):
            raise TypeError(f"channel names must be strings, got {name!r}")
        if not name:
            raise ValueError("channel names must be non-empty strings")
    if not channels:
        raise ValueError("channel_names must name at least one channel")
    if len(set(channels)) != len(channels):
        raise ValueError(f"channel names must be unique, got {channels}")
    return channels


def _restore(
    blocks: np.ndarray, channel_names: tuple[str, ...], grid: Grid
) -> "Covariance":
    """Rebuild a Covariance from its fields without validating (deepcopy, pickle).

    Callers may pass the original array (``copy.copy``), a fresh copy
    (``deepcopy``) or a read-only buffer (pickle protocol 5); the array is
    marked read-only in every case. Only objects that already passed the
    constructor reach this path, so the batched Cholesky is not repeated.
    """
    restored = object.__new__(Covariance)
    blocks.setflags(write=False)
    object.__setattr__(restored, "blocks", blocks)
    object.__setattr__(restored, "channel_names", channel_names)
    object.__setattr__(restored, "grid", grid)
    return restored


@dataclass(frozen=True, eq=False)
class Covariance:
    """Noise covariance on a grid, as one array of channel-by-channel blocks.

    ``blocks[i, j]`` is ``E[x_i conj(x_j)]`` at every point of ``grid``, so
    ``blocks`` has shape ``(K, K) + grid.channel_shape`` for ``K`` channels and
    ``blocks[j, i] == conj(blocks[i, j])``. There is one value per point; a
    covariance between different points is out of scope.

    Build one from a mapping with :meth:`like`. Read it through channel
    attributes: ``cov.X`` is ``pair("X", "X")``, ``cov.XY`` is
    ``pair("X", "Y")`` and ``cov.YX`` its conjugate. With channels exactly
    X, Y, Z the names A, E, T and their pairs are derived on demand, and the
    reverse; see :meth:`in_basis`. A derivation that is not numerically
    positive definite raises ``ValueError``, so ``hasattr(cov, "X")`` can raise
    ``ValueError`` for such an object; this is deliberate so the numerical
    failure is not hidden.

    Units: ``blocks`` holds variances in the squared units of the data's
    coordinates on its grid (``x_n**2`` in time, ``(dt * rfft(x))**2`` in
    frequency, squared WDM coefficients). Stationary noise with one-sided PSD
    ``S`` has bin variance ``(Tobs / 2) * S(f_k)`` at every rfft bin.

    Construction validates channel names, the grid, dtype, shape, finiteness,
    exact Hermitian symmetry, realness where the domain requires it, positive
    variances at every active point, zero blocks at inactive WDM pixels and
    positive definiteness at every active point. The array is stored as a
    fresh read-only copy. Equality is identity; compare arrays explicitly.
    """

    blocks: np.ndarray
    channel_names: tuple[str, ...] = field(kw_only=True)
    grid: Grid = field(kw_only=True)

    # ---- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        channels = _validated_channel_names(self.channel_names)
        object.__setattr__(self, "channel_names", channels)
        if not isinstance(self.grid, Grid):
            raise TypeError("grid must be a Grid")
        blocks = self._validated_blocks(self.blocks, channels, self.grid)
        object.__setattr__(self, "blocks", blocks)

    @staticmethod
    def _validated_blocks(
        value: object, channels: tuple[str, ...], grid: Grid
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError("blocks must be a numpy array")
        if value.dtype.kind not in "fc":
            raise TypeError(
                f"blocks must have a floating or complex dtype, got {value.dtype}"
            )
        count = len(channels)
        expected = (count, count, *grid.channel_shape)
        if value.shape != expected:
            raise ValueError(
                f"blocks must have shape {expected} for {count} channels on this "
                f"{grid.data_domain} grid, got {value.shape}"
            )
        dtype = np.complex128 if value.dtype.kind == "c" else np.float64
        blocks = np.array(value, dtype=dtype, copy=True)
        if not np.all(np.isfinite(blocks)):
            raise ValueError("blocks must be finite")
        # Exact symmetry also forces every blocks[i, i] to be real.
        if np.any(blocks != np.conj(np.swapaxes(blocks, 0, 1))):
            raise ValueError(
                "blocks must be Hermitian at every point: "
                "blocks[j, i] == conj(blocks[i, j])"
            )
        if blocks.dtype.kind == "c":
            if grid.data_domain != "frequency":
                raise ValueError(
                    f"blocks must be real for data_domain={grid.data_domain!r}"
                )
            real_bins = [0] + ([-1] if grid.num_time_samples % 2 == 0 else [])
            if np.any(blocks[:, :, real_bins].imag != 0):
                raise ValueError(
                    "blocks must be real where both coefficients are real "
                    "(DC and even-length Nyquist)"
                )
        variances = np.real(np.diagonal(blocks, axis1=0, axis2=1))  # (*S, K)
        active = grid.active_mask
        if np.any(variances[active] <= 0):
            raise ValueError(
                "every channel's variance must be positive at every active point"
            )
        if np.any(blocks[:, :, ~active] != 0):
            raise ValueError("blocks must be zero at structurally inactive WDM pixels")
        matrices = np.moveaxis(blocks[:, :, active], (0, 1), (-2, -1))  # (P, K, K)
        try:
            np.linalg.cholesky(matrices)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "covariance must be positive definite at every active point"
            ) from exc
        blocks.setflags(write=False)
        return blocks

    @classmethod
    def like(cls, data: "L1Data", pairs: Mapping[Any, np.ndarray]) -> "Covariance":
        """Build a covariance on ``data``'s grid from per-pair arrays.

        Keys are a channel name (``"A"``), a two-channel concatenation
        (``"AE"``, resolved with :func:`split_pair_name`) or a
        ``(first, second)`` tuple. A key ``(a, b)`` sets ``blocks[a, b]`` to the
        array and ``blocks[b, a]`` to its conjugate; two keys naming one pair in
        either orientation raise ``ValueError``. Every channel's own entry is
        required; an absent cross term is zero. Each value is a floating or
        complex array with ``data.grid.channel_shape``. The result is validated
        by the constructor.
        """
        channels = tuple(data.channel_names)
        grid = data.grid
        if not isinstance(pairs, Mapping):
            raise TypeError(
                "pairs must be a mapping from channel or pair names to arrays"
            )
        origins: dict[tuple[str, str], object] = {}
        entries: dict[tuple[str, str], np.ndarray] = {}
        for key, value in pairs.items():
            first, second = cls._resolved_key(key, channels)
            i, j = channels.index(first), channels.index(second)
            canonical = (channels[min(i, j)], channels[max(i, j)])
            if canonical in entries:
                raise ValueError(
                    f"keys {origins[canonical]!r} and {key!r} both name pair "
                    f"{canonical!r}"
                )
            if not isinstance(value, np.ndarray):
                raise TypeError(f"pairs[{key!r}] must be a numpy array")
            if value.dtype.kind not in "fc":
                raise TypeError(
                    f"pairs[{key!r}] must have a floating or complex dtype, "
                    f"got {value.dtype}"
                )
            if value.shape != grid.channel_shape:
                raise ValueError(
                    f"pairs[{key!r}] must have the channel shape "
                    f"{grid.channel_shape} of this {grid.data_domain} grid, "
                    f"got {value.shape}"
                )
            entries[canonical] = value if i <= j else np.conj(value)
            origins[canonical] = key
        missing = [ch for ch in channels if (ch, ch) not in entries]
        if missing:
            raise ValueError(
                f"pairs is missing the diagonal entry for channels {missing}"
            )
        dtype = (
            np.complex128
            if any(np.iscomplexobj(array) for array in entries.values())
            else np.float64
        )
        blocks = np.zeros((len(channels), len(channels), *grid.channel_shape), dtype)
        for (first, second), array in entries.items():
            i, j = channels.index(first), channels.index(second)
            blocks[i, j] = array
            blocks[j, i] = np.conj(array)
        return cls(blocks, channel_names=channels, grid=grid)

    @staticmethod
    def _resolved_key(key: object, channels: tuple[str, ...]) -> tuple[str, str]:
        if isinstance(key, str):
            resolved = split_pair_name(key, channels)
            if resolved is None:
                raise ValueError(
                    f"key {key!r} is neither a channel nor a pair of channels "
                    f"from {channels}"
                )
            return resolved
        if isinstance(key, tuple) and len(key) == 2:
            first, second = key
            if isinstance(first, str) and isinstance(second, str):
                for name in (first, second):
                    if name not in channels:
                        raise ValueError(
                            f"unknown channel {name!r} in key {key!r}; "
                            f"channels are {channels}"
                        )
                return first, second
        raise TypeError(
            "pair keys must be a channel name, a two-channel name or a "
            f"(first, second) tuple, got {key!r}"
        )

    # ---- access -----------------------------------------------------------

    def pair(self, first: str, second: str) -> np.ndarray:
        """Covariance between two channels at every point: a read-only view."""
        for name in (first, second):
            if name not in self.channel_names:
                raise ValueError(
                    f"unknown channel {name!r}; channels are {self.channel_names}"
                )
        return self.blocks[
            self.channel_names.index(first), self.channel_names.index(second)
        ]

    def __getattr__(self, name: str) -> np.ndarray:
        # Only reached when normal lookup fails, so fields and methods win.
        if name.startswith("_") or name in _FIELD_NAMES:
            raise AttributeError(name)
        channels = self.channel_names
        try:
            resolved = split_pair_name(name, channels)
        except ValueError as exc:
            raise AttributeError(str(exc)) from None
        if resolved is not None:
            return self.pair(*resolved)
        derived_basis = self._other_basis()
        if derived_basis is not None:
            if split_pair_name(name, BASES[derived_basis]) is not None:
                return getattr(self.in_basis(derived_basis), name)
        valid = [*channels, *(a + b for a in channels for b in channels if a != b)]
        if derived_basis is not None:
            derived_channels = BASES[derived_basis]
            valid += [
                *derived_channels,
                *(a + b for a in derived_channels for b in derived_channels if a != b),
            ]
        raise AttributeError(
            f"Covariance has no attribute {name!r}; channel attributes are {valid}"
        )

    # ---- derived basis ----------------------------------------------------

    def _basis_name(self) -> str | None:
        current = frozenset(self.channel_names)
        for basis, names in BASES.items():
            if current == frozenset(names):
                return basis
        return None

    def _other_basis(self) -> str | None:
        basis = self._basis_name()
        if basis is None:
            return None
        return "AET" if basis == "XYZ" else "XYZ"

    def in_basis(self, basis: str) -> "Covariance":
        """Return this covariance in the ``"XYZ"`` or ``"AET"`` basis.

        Available only when the channels are exactly X, Y, Z or exactly A, E, T.
        The current basis returns ``self``; otherwise a new object with channel
        order ``("A", "E", "T")`` or ``("X", "Y", "Z")`` on the same grid is
        computed on every call (no cache). The rotation is applied per point and
        symmetrised exactly.

        Derivation needs the rotated covariance to be numerically positive
        definite; when one channel's variance is below float64 roundoff of the
        others (T << A at the lowest LISA frequencies) this raises
        ``ValueError`` with that explanation, chaining the constructor's error.
        """
        if basis not in BASES:
            raise ValueError(f"basis must be one of {tuple(BASES)}, got {basis!r}")
        current = self._basis_name()
        if current is None:
            raise ValueError(
                "deriving a basis needs channels exactly X, Y, Z or A, E, T; "
                f"got {self.channel_names}"
            )
        if basis == current:
            return self
        source, target = BASES[current], BASES[basis]
        order = [self.channel_names.index(name) for name in source]
        ordered = self.blocks[np.ix_(order, order)]
        matrix = AET_FROM_XYZ if basis == "AET" else AET_FROM_XYZ.T
        rotated = np.einsum("ai,bj,ij...->ab...", matrix, matrix, ordered)
        rotated = 0.5 * (rotated + np.conj(np.swapaxes(rotated, 0, 1)))
        for index in range(len(target)):
            rotated[index, index] = rotated[index, index].real
        try:
            return Covariance(rotated, channel_names=target, grid=self.grid)
        except ValueError as exc:
            raise ValueError(
                f"cannot derive the {basis} basis from {current}: the rotated "
                "covariance is not numerically positive definite (one channel's "
                "variance is below the roundoff of the others, e.g. T << A at "
                f"low frequency): {exc}"
            ) from exc

    # ---- the accompanying data --------------------------------------------

    def check_compatible(self, data: "L1Data") -> None:
        """Require this covariance to describe ``data``: same channels, same grid."""
        if tuple(data.channel_names) != self.channel_names:
            raise ValueError(
                f"covariance channel_names {self.channel_names} do not match the "
                f"data's {tuple(data.channel_names)}"
            )
        if data.grid != self.grid:
            raise ValueError(
                f"covariance grid {self.grid} does not match the data's grid "
                f"{data.grid}"
            )

    # ---- copies -----------------------------------------------------------

    def copy(self) -> "Covariance":
        """An independent, revalidated plain ``Covariance``.

        The trust boundary: the Wheel calls ``Covariance.copy(x)`` unbound on
        every covariance it receives, so a subclass cannot skip validation and
        an array whose write flag a caller flipped is caught.
        """
        return Covariance(self.blocks, channel_names=self.channel_names, grid=self.grid)

    def __reduce__(
        self,
    ) -> tuple[Callable[..., "Covariance"], tuple[np.ndarray, tuple[str, ...], Grid]]:
        # deepcopy and pickle restore without validating; a subclass instance
        # comes back as a plain Covariance, like copy().
        return _restore, (self.blocks, self.channel_names, self.grid)


_FIELD_NAMES = frozenset(f.name for f in fields(Covariance))
