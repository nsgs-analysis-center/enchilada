"""Covariance: one stacked Hermitian array per grid, read through channel names."""

import pickle
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from conftest import white_covariance
from enchilada import Covariance, Grid, L1Data, WDMGrid, transform
from enchilada.covariance import AET_FROM_XYZ

XYZ = ("X", "Y", "Z")
AET = ("A", "E", "T")


def time_data(num_time_samples=8, channel_names=("A", "E"), sample_rate_hz=2.0):
    rng = np.random.default_rng(0)
    return L1Data(
        {ch: rng.normal(size=num_time_samples) for ch in channel_names},
        sample_rate_hz=sample_rate_hz,
        channel_names=channel_names,
        tdi_generation="2.0",
        physical_observable="strain",
    )


def frequency_data(num_time_samples=8, channel_names=("A", "E")):
    return transform(time_data(num_time_samples, channel_names), "frequency")


def wdm_data(channel_names=("A", "E")):
    # Built by hand on the folded (Nf, Nt) layout; no transform needed.
    grid = WDMGrid(num_frequency_divisions=4, num_time_divisions=4)
    return L1Data(
        {ch: np.zeros(grid.array_shape) for ch in channel_names},
        sample_rate_hz=2.0,
        channel_names=channel_names,
        tdi_generation="2.0",
        physical_observable="strain",
        data_domain="wdm",
    )


def stacked(shape, variances, cross=0.0):
    """(K, K, *shape) blocks with constant variances and one constant cross term."""
    count = len(variances)
    blocks = np.full((count, count, *shape), cross, dtype=float)
    for index, variance in enumerate(variances):
        blocks[index, index] = variance
    return blocks


class TestConstruction:
    def test_stores_a_fresh_read_only_float_copy(self):
        data = time_data()
        blocks = stacked((8,), (1.0, 2.0), 0.5).astype(np.float32)
        cov = Covariance(blocks, channel_names=data.channel_names, grid=data.grid)
        assert cov.blocks.dtype == np.float64
        assert not cov.blocks.flags.writeable
        assert not np.shares_memory(cov.blocks, blocks)
        blocks[0, 0] = 100.0
        np.testing.assert_array_equal(cov.A, 1.0)
        assert cov.channel_names == ("A", "E") and cov.grid == data.grid

    def test_channel_names_may_be_a_list_and_are_stored_as_a_tuple(self):
        data = time_data()
        cov = Covariance(
            stacked((8,), (1.0, 2.0)), channel_names=["A", "E"], grid=data.grid
        )
        assert cov.channel_names == ("A", "E")

    @pytest.mark.parametrize(
        ("names", "error", "match"),
        [
            ("AE", TypeError, "tuple"),
            (("A", 1), TypeError, "strings"),
            ((), ValueError, "at least one"),
            (("A", "A"), ValueError, "unique"),
            (("A", ""), ValueError, "non-empty"),
        ],
    )
    def test_channel_names_validation(self, names, error, match):
        grid = Grid("time", 8, 2.0)
        with pytest.raises(error, match=match):
            Covariance(stacked((8,), (1.0, 2.0)), channel_names=names, grid=grid)

    def test_grid_must_be_a_grid(self):
        with pytest.raises(TypeError, match="grid must be a Grid"):
            Covariance(
                stacked((8,), (1.0, 2.0)),
                channel_names=("A", "E"),
                grid=("time", 8, 2.0),
            )

    def test_blocks_must_be_a_floating_array(self):
        grid = Grid("time", 8, 2.0)
        with pytest.raises(TypeError, match="numpy array"):
            Covariance(
                stacked((8,), (1.0, 2.0)).tolist(), channel_names=("A", "E"), grid=grid
            )
        with pytest.raises(TypeError, match="dtype"):
            Covariance(
                np.ones((2, 2, 8), dtype=int), channel_names=("A", "E"), grid=grid
            )

    @pytest.mark.parametrize("shape", [(2, 2, 7), (3, 3, 8), (2, 8), (2, 2, 8, 1)])
    def test_blocks_shape_must_match_channels_and_grid(self, shape):
        grid = Grid("time", 8, 2.0)
        with pytest.raises(ValueError, match=r"shape \(2, 2, 8\)"):
            Covariance(np.ones(shape), channel_names=("A", "E"), grid=grid)

    def test_blocks_must_be_finite(self):
        blocks = stacked((8,), (1.0, 2.0))
        blocks[1, 1, 3] = np.nan
        with pytest.raises(ValueError, match="finite"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    def test_blocks_must_be_exactly_hermitian(self):
        blocks = stacked((8,), (1.0, 2.0), 0.5)
        blocks[1, 0, 2] = 0.5 + 1e-15
        with pytest.raises(ValueError, match="Hermitian"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    @pytest.mark.parametrize("data", [time_data(), wdm_data()])
    def test_time_and_wdm_blocks_must_be_real(self, data):
        blocks = np.zeros((2, 2, *data.grid.channel_shape), dtype=complex)
        blocks[0, 0] = blocks[1, 1] = 1.0
        with pytest.raises(
            ValueError, match=f"real for data_domain={data.data_domain!r}"
        ):
            Covariance(blocks, channel_names=data.channel_names, grid=data.grid)

    @pytest.mark.parametrize("n", [8, 9])
    def test_frequency_blocks_must_be_real_on_the_diagonal_and_at_real_bins(self, n):
        data = frequency_data(n)
        bins = n // 2 + 1
        blocks = stacked((bins,), (1.0, 2.0)).astype(complex)
        bad_diagonal = blocks.copy()
        bad_diagonal[0, 0, 2] = 1.0 + 1e-3j  # fails exact Hermitian symmetry
        with pytest.raises(ValueError, match="Hermitian"):
            Covariance(bad_diagonal, channel_names=("A", "E"), grid=data.grid)
        bad_dc = blocks.copy()
        bad_dc[0, 1, 0], bad_dc[1, 0, 0] = 0.1j, -0.1j
        with pytest.raises(ValueError, match="DC and even-length Nyquist"):
            Covariance(bad_dc, channel_names=("A", "E"), grid=data.grid)
        nyquist = blocks.copy()
        nyquist[0, 1, -1], nyquist[1, 0, -1] = 0.1j, -0.1j
        if n % 2 == 0:
            with pytest.raises(ValueError, match="DC and even-length Nyquist"):
                Covariance(nyquist, channel_names=("A", "E"), grid=data.grid)
        else:
            Covariance(nyquist, channel_names=("A", "E"), grid=data.grid)

    def test_frequency_blocks_may_have_complex_hermitian_cross_terms(self):
        data = frequency_data(8)
        blocks = stacked((5,), (1.0, 2.0)).astype(complex)
        blocks[0, 1, 1:-1] = 0.3 + 0.1j
        blocks[1, 0, 1:-1] = 0.3 - 0.1j
        cov = Covariance(blocks, channel_names=("A", "E"), grid=data.grid)
        assert cov.blocks.dtype == np.complex128
        np.testing.assert_array_equal(cov.EA, np.conj(cov.AE))
        assert cov.A.dtype == np.complex128 and np.all(cov.A.imag == 0)

    def test_variances_must_be_positive_at_every_point(self):
        blocks = stacked((8,), (1.0, 2.0))
        blocks[1, 1, 5] = 0.0
        with pytest.raises(ValueError, match="variance must be positive"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    def test_wdm_variances_are_required_at_every_cell(self):
        data = wdm_data()
        blocks = stacked(data.grid.channel_shape, (1.0, 2.0))
        cov = Covariance(blocks, channel_names=data.channel_names, grid=data.grid)
        np.testing.assert_array_equal(cov.A, 1.0)
        blocks[0, 0, 0, 1] = 0.0  # row 0, odd slot: a Nyquist cell, not padding
        with pytest.raises(
            ValueError, match="variance must be positive at every point"
        ):
            Covariance(blocks, channel_names=data.channel_names, grid=data.grid)

    def test_channel_matrices_must_be_positive_definite(self):
        blocks = stacked((8,), (1.0, 1.0), 0.5)
        blocks[0, 1, 4] = blocks[1, 0, 4] = 1.5
        with pytest.raises(ValueError, match="positive definite at every point"):
            Covariance(blocks, channel_names=("A", "E"), grid=Grid("time", 8, 2.0))

    def test_equality_is_identity(self):
        data = time_data()
        a = Covariance(
            stacked((8,), (1.0, 2.0)), channel_names=("A", "E"), grid=data.grid
        )
        b = Covariance(
            stacked((8,), (1.0, 2.0)), channel_names=("A", "E"), grid=data.grid
        )
        assert a == a and a != b


class TestLike:
    def test_adopts_the_data_grid_and_channel_order(self):
        data = L1Data(
            {"E": np.ones(8), "A": np.ones(8)},
            sample_rate_hz=2.0,
            tdi_generation="2.0",
            physical_observable="strain",
        )
        cov = Covariance.like(data, {"A": np.full(8, 1.0), "E": np.full(8, 4.0)})
        assert cov.channel_names == ("E", "A") and cov.grid == data.grid
        np.testing.assert_array_equal(cov.blocks[0, 0], 4.0)
        np.testing.assert_array_equal(cov.A, 1.0)
        np.testing.assert_array_equal(cov.AE, 0.0)

    def test_string_and_tuple_keys_in_either_orientation(self):
        data = frequency_data(8)
        cross = np.array([0.0, 0.1j, 0.2j, 0.3j, 0.0])
        by_string = Covariance.like(
            data, {"A": np.ones(5), "E": np.ones(5), "EA": cross}
        )
        by_tuple = Covariance.like(
            data, {"A": np.ones(5), "E": np.ones(5), ("E", "A"): cross}
        )
        for cov in (by_string, by_tuple):
            np.testing.assert_array_equal(cov.pair("E", "A"), cross)
            np.testing.assert_array_equal(cov.pair("A", "E"), np.conj(cross))

    def test_unknown_and_unresolvable_keys(self):
        data = time_data()
        ones = np.ones(8)
        with pytest.raises(ValueError, match="neither a channel nor a pair"):
            Covariance.like(data, {"A": ones, "E": ones, "Q": ones})
        with pytest.raises(ValueError, match="unknown channel 'Q'"):
            Covariance.like(data, {"A": ones, "E": ones, ("A", "Q"): ones})
        with pytest.raises(TypeError, match="pair keys"):
            Covariance.like(data, {"A": ones, "E": ones, 1: ones})
        with pytest.raises(TypeError, match="mapping"):
            Covariance.like(data, [("A", ones)])

    def test_two_keys_naming_one_pair_are_rejected(self):
        data = time_data()
        ones, cross = np.ones(8), np.full(8, 0.1)
        with pytest.raises(ValueError, match="both name pair"):
            Covariance.like(data, {"A": ones, "E": ones, "AE": cross, "EA": cross})
        with pytest.raises(ValueError, match="both name pair"):
            Covariance.like(
                data, {"A": ones, "E": ones, "AE": cross, ("A", "E"): cross}
            )

    def test_every_diagonal_is_required(self):
        with pytest.raises(
            ValueError, match=r"missing the diagonal entry for channels \['E'\]"
        ):
            Covariance.like(time_data(), {"A": np.ones(8)})

    def test_values_must_be_floating_arrays_with_the_channel_shape(self):
        data = time_data()
        with pytest.raises(TypeError, match="numpy array"):
            Covariance.like(data, {"A": [1.0] * 8, "E": np.ones(8)})
        with pytest.raises(TypeError, match="dtype"):
            Covariance.like(data, {"A": np.ones(8, dtype=int), "E": np.ones(8)})
        with pytest.raises(ValueError, match=r"channel shape \(8,\)"):
            Covariance.like(data, {"A": np.ones(7), "E": np.ones(8)})
        with pytest.raises(ValueError, match=r"channel shape \(8,\)"):
            Covariance.like(data, {"A": np.eye(8), "E": np.eye(8)})

    def test_ambiguous_string_keys_are_errors(self):
        data = time_data(channel_names=("A", "B", "AB", "BA"))
        pairs = {name: np.ones(8) for name in data.channel_names}
        with pytest.raises(ValueError, match="ambiguous"):
            Covariance.like(data, {**pairs, "ABA": np.full(8, 0.1)})
        cov = Covariance.like(data, {**pairs, ("AB", "A"): np.full(8, 0.1)})
        np.testing.assert_array_equal(cov.pair("AB", "A"), 0.1)

    def test_result_is_validated(self):
        data = time_data()
        with pytest.raises(ValueError, match="positive definite"):
            Covariance.like(
                data, {"A": np.ones(8), "E": np.ones(8), "AE": np.full(8, 2.0)}
            )

    def test_wdm_variance_at_every_cell(self):
        data = wdm_data()
        variances = np.full(data.grid.channel_shape, 3.0)
        cov = Covariance.like(data, {"A": variances, "E": variances})
        np.testing.assert_array_equal(cov.A, 3.0)
        assert cov.grid.data_domain == "wdm"


class TestAccess:
    def test_pair_returns_a_read_only_view_into_blocks(self):
        cov = white_covariance(time_data(), {"A": 1.0, "E": 4.0})
        view = cov.pair("A", "E")
        assert np.shares_memory(view, cov.blocks)
        assert not view.flags.writeable
        np.testing.assert_array_equal(cov.pair("E", "E"), 4.0)
        with pytest.raises(ValueError, match="unknown channel 'Q'"):
            cov.pair("A", "Q")

    def test_attribute_sugar(self):
        data = frequency_data(8)
        cross = np.array([0.0, 0.1j, 0.2j, 0.3j, 0.0])
        cov = Covariance.like(data, {"A": np.ones(5), "E": np.ones(5), "AE": cross})
        np.testing.assert_array_equal(cov.A, 1.0)
        np.testing.assert_array_equal(cov.AE, cross)
        np.testing.assert_array_equal(cov.EA, np.conj(cross))
        assert np.shares_memory(cov.AE, cov.blocks)

    def test_ambiguous_attribute_names_need_pair(self):
        data = time_data(channel_names=("A", "B", "AB", "BA"))
        cov = white_covariance(data, 1.0)
        with pytest.raises(AttributeError, match="ambiguous"):
            _ = cov.ABA
        np.testing.assert_array_equal(cov.pair("AB", "A"), 0.0)

    def test_unknown_attribute_lists_valid_names(self):
        cov = white_covariance(time_data(), 1.0)
        with pytest.raises(AttributeError, match="channel attributes are .*'AE'"):
            _ = cov.Q
        assert not hasattr(cov, "form")

    def test_uninitialised_instance_has_no_channel_attributes(self):
        empty = object.__new__(Covariance)
        assert not hasattr(empty, "blocks")
        assert not hasattr(empty, "A")
        assert not hasattr(empty, "__deepcopy__")


def xyz_covariance(data, cross=0.2):
    shape = data.grid.channel_shape
    pairs = {
        "X": np.full(shape, 1.0),
        "Y": np.full(shape, 2.0),
        "Z": np.full(shape, 3.0),
        "XY": np.full(shape, cross),
        "YZ": np.full(shape, -cross),
    }
    return Covariance.like(data, pairs)


class TestInBasis:
    def test_aet_matches_the_explicit_matrix_product(self):
        cov = xyz_covariance(time_data(channel_names=XYZ))
        aet = cov.in_basis("AET")
        assert aet.channel_names == AET and aet.grid == cov.grid
        expected = np.einsum("ai,bj,ijn->abn", AET_FROM_XYZ, AET_FROM_XYZ, cov.blocks)
        np.testing.assert_allclose(aet.blocks, expected, atol=1e-14)
        assert not aet.blocks.flags.writeable

    def test_round_trip_without_a_cache(self):
        cov = xyz_covariance(time_data(channel_names=XYZ))
        aet = cov.in_basis("AET")
        np.testing.assert_allclose(aet.in_basis("XYZ").blocks, cov.blocks, atol=1e-14)
        assert cov.in_basis("XYZ") is cov
        assert cov.in_basis("AET") is not aet

    def test_white_equal_noise_stays_white(self):
        cov = white_covariance(time_data(channel_names=XYZ), 2.0)
        aet = cov.in_basis("AET")
        for name in AET:
            np.testing.assert_allclose(aet.pair(name, name), 2.0)
        np.testing.assert_allclose(aet.AE, 0.0, atol=1e-14)

    def test_derived_attributes_read_through_the_other_basis(self):
        cov = xyz_covariance(time_data(channel_names=XYZ))
        np.testing.assert_array_equal(cov.A, cov.in_basis("AET").A)
        np.testing.assert_array_equal(cov.AE, cov.in_basis("AET").AE)
        aet = cov.in_basis("AET")
        np.testing.assert_allclose(aet.X, cov.X, atol=1e-14)
        np.testing.assert_allclose(aet.YZ, cov.YZ, atol=1e-14)

    def test_any_channel_order_works(self):
        ordered = xyz_covariance(time_data(channel_names=XYZ))
        shuffled = Covariance.like(
            time_data(channel_names=("Z", "X", "Y")),
            {
                "X": ordered.X,
                "Y": ordered.Y,
                "Z": ordered.Z,
                "XY": ordered.XY,
                "YZ": ordered.YZ,
            },
        )
        np.testing.assert_allclose(
            shuffled.in_basis("AET").blocks, ordered.in_basis("AET").blocks, atol=1e-14
        )

    def test_basis_needs_the_full_triple_and_a_known_name(self):
        cov = white_covariance(time_data(), 1.0)
        with pytest.raises(ValueError, match="exactly X, Y, Z or A, E, T"):
            cov.in_basis("AET")
        with pytest.raises(ValueError, match="basis must be one of"):
            xyz_covariance(time_data(channel_names=XYZ)).in_basis("ABC")
        with pytest.raises(AttributeError):
            _ = cov.X

    def test_derivation_below_float64_precision_explains_itself(self):
        data = time_data(num_time_samples=64, channel_names=AET)
        cov = Covariance.like(
            data,
            {
                "A": np.linspace(1, 3, 64),
                "E": np.linspace(2, 5, 64),
                "T": np.full(64, 1e-20),
            },
        )
        with pytest.raises(ValueError, match="cannot derive the XYZ basis") as info:
            cov.in_basis("XYZ")
        assert isinstance(info.value.__cause__, ValueError)
        with pytest.raises(ValueError, match="cannot derive the XYZ basis"):
            _ = cov.X

    def test_frequency_complex_blocks_rotate_hermitian(self):
        data = frequency_data(8, channel_names=XYZ)
        pairs = {
            "X": np.ones(5),
            "Y": np.full(5, 2.0),
            "Z": np.full(5, 3.0),
            "XY": np.array([0.0, 0.1j, 0.2j, 0.1j, 0.0]),
        }
        aet = Covariance.like(data, pairs).in_basis("AET")
        assert aet.blocks.dtype == np.complex128
        np.testing.assert_array_equal(
            aet.blocks, np.conj(np.swapaxes(aet.blocks, 0, 1))
        )
        assert np.all(aet.A.imag == 0)

    def test_wdm_derivation_matches_the_time_domain_derivation_cell_by_cell(self):
        wdm = wdm_data(channel_names=XYZ)
        time = time_data(16, channel_names=XYZ)
        cells = 1.0 + np.arange(16, dtype=float)  # a different level at every cell
        levels = dict(zip(XYZ, (1.0, 2.0, 3.0), strict=True))
        aet_wdm = Covariance.like(
            wdm,
            {
                name: level * cells.reshape(wdm.grid.channel_shape)
                for name, level in levels.items()
            },
        ).in_basis("AET")
        aet_time = Covariance.like(
            time, {name: level * cells for name, level in levels.items()}
        ).in_basis("AET")
        assert aet_wdm.grid == wdm.grid
        np.testing.assert_allclose(aet_wdm.blocks.reshape(3, 3, 16), aet_time.blocks)


class TestCopies:
    def test_copy_is_independent_and_revalidated(self):
        cov = white_covariance(time_data(), 1.0)
        copied = cov.copy()
        assert type(copied) is Covariance and copied is not cov
        np.testing.assert_array_equal(copied.blocks, cov.blocks)
        assert not np.shares_memory(copied.blocks, cov.blocks)
        assert not copied.blocks.flags.writeable
        cov.blocks.setflags(write=True)
        cov.A[:] = -1.0
        with pytest.raises(ValueError, match="variance must be positive"):
            cov.copy()

    def test_unbound_copy_returns_a_plain_covariance(self):
        class Unchecked(Covariance):
            def copy(self):
                return self

        cov = white_covariance(time_data(), 1.0)
        unchecked = Unchecked(
            cov.blocks, channel_names=cov.channel_names, grid=cov.grid
        )
        assert Covariance.copy(unchecked) is not unchecked
        assert type(Covariance.copy(unchecked)) is Covariance

    def test_deepcopy_and_pickle_restore_read_only_without_validation(self):
        cov = white_covariance(frequency_data(8), 1.0)
        for restored in (deepcopy(cov), pickle.loads(pickle.dumps(cov))):  # ruff: ignore[S301]
            assert type(restored) is Covariance
            np.testing.assert_array_equal(restored.blocks, cov.blocks)
            assert not np.shares_memory(restored.blocks, cov.blocks)
            assert not restored.blocks.flags.writeable
            assert restored.channel_names == cov.channel_names
            assert restored.grid == cov.grid
            np.testing.assert_array_equal(restored.A, cov.A)
        cov.blocks.setflags(write=True)
        cov.A[:] = -1.0  # invalid, yet the restore paths do not validate
        np.testing.assert_array_equal(deepcopy(cov).A, -1.0)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(cov)).A, -1.0)  # ruff: ignore[S301]


class TestCheckCompatible:
    def test_accepts_the_data_it_was_built_from(self):
        data = time_data()
        white_covariance(data, 1.0).check_compatible(data)
        white_covariance(data, 1.0).check_compatible(replace(data, start_time_gps=9.0))

    def test_rejects_other_channel_order_and_other_grids(self):
        data = time_data()
        cov = white_covariance(data, 1.0)
        reordered = L1Data(
            {"E": data.channel_data["E"], "A": data.channel_data["A"]},
            sample_rate_hz=2.0,
            tdi_generation="2.0",
            physical_observable="strain",
        )
        with pytest.raises(ValueError, match="channel_names"):
            cov.check_compatible(reordered)
        with pytest.raises(ValueError, match="grid"):
            cov.check_compatible(transform(data, "frequency"))
        with pytest.raises(ValueError, match="grid"):
            cov.check_compatible(time_data(num_time_samples=16))
        with pytest.raises(ValueError, match="grid"):
            cov.check_compatible(time_data(sample_rate_hz=1.0))


class TestAdd:
    def test_sums_variances_and_cross_terms_on_one_grid(self):
        data = time_data()
        left = Covariance.like(
            data, {"A": np.full(8, 1.0), "E": np.full(8, 2.0), "AE": np.full(8, 0.5)}
        )
        right = Covariance.like(data, {"A": np.full(8, 3.0), "E": np.full(8, 4.0)})
        total = left + right
        assert type(total) is Covariance
        assert total.grid == data.grid
        assert total.channel_names == ("A", "E")
        np.testing.assert_array_equal(total.A, 4.0)
        np.testing.assert_array_equal(total.E, 6.0)
        np.testing.assert_array_equal(total.AE, 0.5)
        np.testing.assert_array_equal(total.EA, 0.5)
        assert not total.blocks.flags.writeable
        assert not np.shares_memory(total.blocks, left.blocks)
        assert not np.shares_memory(total.blocks, right.blocks)
        np.testing.assert_array_equal(left.A, 1.0)  # the operands are untouched
        np.testing.assert_array_equal(right.AE, 0.0)

    def test_addition_is_commutative_and_promotes_to_complex(self):
        data = frequency_data()  # 8 samples: 5 bins, DC and Nyquist real
        real = Covariance.like(data, {"A": np.full(5, 1.0), "E": np.full(5, 1.0)})
        cross = np.full(5, 0.25 + 0.1j)
        cross[0] = cross[-1] = 0.25
        with_cross = Covariance.like(
            data, {"A": np.full(5, 2.0), "E": np.full(5, 2.0), "AE": cross}
        )
        total = real + with_cross
        assert total.blocks.dtype == np.complex128
        np.testing.assert_array_equal(total.A, 3.0)
        np.testing.assert_array_equal(total.E, 3.0)
        np.testing.assert_array_equal(total.AE, cross)
        np.testing.assert_array_equal(total.EA, np.conj(cross))
        np.testing.assert_array_equal((with_cross + real).blocks, total.blocks)

    def test_the_sum_passes_the_constructor_checks(self):
        # Each term is positive definite; the sum goes through the same Cholesky.
        data = time_data()
        tight = Covariance.like(
            data, {"A": np.ones(8), "E": np.ones(8), "AE": np.full(8, 0.99)}
        )
        total = tight + tight
        np.testing.assert_array_equal(total.A, 2.0)
        np.testing.assert_array_equal(total.AE, 1.98)

    def test_grids_must_match(self):
        data = time_data()
        left = white_covariance(data, 1.0)
        with pytest.raises(ValueError, match="does not add across grids") as info:
            left + white_covariance(data.to_frequency(), 1.0)
        assert "data_domain='time'" in str(info.value)
        assert "data_domain='frequency'" in str(info.value)
        with pytest.raises(ValueError, match="does not add across grids"):
            left + white_covariance(time_data(num_time_samples=4), 1.0)

    def test_channel_order_must_match(self):
        data = time_data()
        reordered = replace(data, channel_names=("E", "A"))
        with pytest.raises(ValueError, match=r"channel_names \('E', 'A'\)") as info:
            white_covariance(data, 1.0) + white_covariance(reordered, 1.0)
        assert "('A', 'E')" in str(info.value)

    def test_other_operands_are_not_implemented(self):
        cov = white_covariance(time_data(), 1.0)
        assert cov.__add__(1.0) is NotImplemented
        with pytest.raises(TypeError):
            cov + 1.0
        with pytest.raises(TypeError):
            1.0 + cov
        with pytest.raises(TypeError):
            cov + None

    def test_the_sum_is_validated_like_any_construction(self):
        data = time_data()
        huge = Covariance.like(data, {"A": np.full(8, 1e308), "E": np.full(8, 1e308)})
        with pytest.raises(ValueError, match="finite"):
            huge + huge

    def test_a_subclass_operand_yields_a_plain_covariance_in_both_orders(self):
        class Sub(Covariance):
            pass

        data = time_data()
        plain = white_covariance(data, 1.0)
        sub = Sub(plain.blocks, channel_names=plain.channel_names, grid=plain.grid)
        for total in (sub + plain, plain + sub):
            assert type(total) is Covariance
            np.testing.assert_array_equal(total.A, 2.0)
