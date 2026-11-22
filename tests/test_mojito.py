"""load_mojito_l1: a Mojito L1 HDF5 window as fractional-frequency L1Data."""

import os

import numpy as np
import pytest

from conftest import (
    MOJITO_DT,
    MOJITO_ECLIPTIC_Z_M,
    MOJITO_LASER_FREQUENCY_HZ,
    MOJITO_ORBIT_DT,
    MOJITO_ORBIT_SIZE,
    MOJITO_ORBIT_T0,
    MOJITO_T0,
    write_mojito_file,
)
from enchilada import NumericOrbit, decimate, load_mojito_l1

N = 4096


@pytest.fixture
def mojito_file(tmp_path):
    path = tmp_path / "mojito.h5"
    xyz, aet = write_mojito_file(path, n=N)
    return path, xyz, aet


def test_default_channels_units_and_grid(mojito_file):
    path, xyz, aet = mojito_file
    data = load_mojito_l1(path)
    assert data.channel_names == ("A", "E", "T")
    assert data.tdi_generation == "2.0"
    assert data.physical_observable == "fractional_frequency"
    assert data.sample_rate_hz == 4.0
    assert data.num_time_samples == N
    assert data.start_time_gps == MOJITO_T0
    assert data.data_domain == "time"
    for row, name in zip(aet, "AET", strict=True):
        np.testing.assert_array_equal(
            data.channel_data[name], row / MOJITO_LASER_FREQUENCY_HZ
        )


def test_xyz_channels_in_the_requested_order(mojito_file):
    path, xyz, _ = mojito_file
    data = load_mojito_l1(path, channels=("Z", "X"), orbit=False)
    assert data.channel_names == ("Z", "X")
    np.testing.assert_array_equal(
        data.channel_data["Z"], xyz[2] / MOJITO_LASER_FREQUENCY_HZ
    )
    np.testing.assert_array_equal(
        data.channel_data["X"], xyz[0] / MOJITO_LASER_FREQUENCY_HZ
    )


def test_unknown_channel_is_refused(mojito_file):
    path, _, _ = mojito_file
    with pytest.raises(ValueError, match=r"A, E, T, X, Y, Z"):
        load_mojito_l1(path, channels=("A", "B"))


def test_missing_dataset_is_refused(tmp_path):
    path = tmp_path / "no_t.h5"
    write_mojito_file(path, drop=("T2",))
    with pytest.raises(ValueError, match="tdis/T2"):
        load_mojito_l1(path)


def test_window_rounds_the_start_and_floors_the_count(mojito_file):
    path, _, aet = mojito_file
    data = load_mojito_l1(path, start_offset_s=10.1, duration_s=100.9, orbit=False)
    # 10.1 / 0.25 = 40.4 -> 40; 100.9 / 0.25 = 403.6 -> 403
    assert data.num_time_samples == 403
    assert data.start_time_gps == MOJITO_T0 + 40 * MOJITO_DT
    np.testing.assert_array_equal(
        data.channel_data["A"], aet[0, 40:443] / MOJITO_LASER_FREQUENCY_HZ
    )


def test_omitted_duration_reads_to_the_end(mojito_file):
    path, _, aet = mojito_file
    data = load_mojito_l1(path, start_offset_s=1000.0, orbit=False)
    assert data.num_time_samples == N - 4000
    np.testing.assert_array_equal(
        data.channel_data["E"], aet[1, 4000:] / MOJITO_LASER_FREQUENCY_HZ
    )


@pytest.mark.parametrize(
    "window",
    [
        dict(start_offset_s=2000.0),  # starts past the end
        dict(start_offset_s=1000.0, duration_s=100.0),  # ends past the end
        dict(start_offset_s=0.0, duration_s=0.1),  # zero samples
        dict(start_offset_s=-1.0),  # before the first sample
    ],
)
def test_windows_outside_the_file_are_refused(mojito_file, window):
    path, _, _ = mojito_file
    last = MOJITO_T0 + (N - 1) * MOJITO_DT
    with pytest.raises(ValueError, match=rf"\[{MOJITO_T0}, {last}\]"):
        load_mojito_l1(path, orbit=False, **window)


def test_inconsistent_rate_attributes_are_refused(tmp_path):
    path = tmp_path / "bad_fs.h5"
    write_mojito_file(path, fs=3.9)
    with pytest.raises(ValueError, match=r"dt.*fs"):
        load_mojito_l1(path, orbit=False)


def test_size_attribute_larger_than_the_datasets_is_refused(tmp_path):
    path = tmp_path / "m.h5"
    write_mojito_file(path, size=5000)
    with pytest.raises(
        ValueError,
        match=r"tdis/tdi_flags has shape \(4096,\) but tdis/sampling says size = 5000",
    ):
        load_mojito_l1(path, duration_s=1000.0)


def _shorten_dataset(path, name, dtype):
    """Replace ``tdis/<name>`` with 100 zeros, leaving the size attribute alone."""
    import h5py

    with h5py.File(path, "r+") as f:
        del f["tdis"][name]
        f["tdis"].create_dataset(name, data=np.zeros(100, dtype=dtype))


def test_short_flags_dataset_is_refused(tmp_path):
    path = tmp_path / "m.h5"
    write_mojito_file(path)
    _shorten_dataset(path, "tdi_flags", np.int8)
    with pytest.raises(ValueError, match=r"tdis/tdi_flags has shape \(100,\)"):
        load_mojito_l1(path, orbit=False)


def test_short_channel_dataset_is_refused(tmp_path):
    path = tmp_path / "m.h5"
    write_mojito_file(path)
    _shorten_dataset(path, "A2", np.float64)
    with pytest.raises(ValueError, match=r"tdis/A2 has shape \(100,\)"):
        load_mojito_l1(path, orbit=False)


def test_nonpositive_laser_frequency_is_refused(tmp_path):
    path = tmp_path / "m.h5"
    write_mojito_file(path, laser_frequency=0.0)
    with pytest.raises(ValueError, match="laser_frequency must be a positive finite"):
        load_mojito_l1(path, orbit=False)


@pytest.mark.parametrize(
    ("member", "message"),
    [
        ("tdis/sampling", "has no tdis/sampling group"),
        ("tdis/tdi_flags", "has no dataset tdis/tdi_flags"),
        ("laser_frequency", "has no root attribute laser_frequency"),
    ],
)
def test_missing_mojito_layout_is_named(tmp_path, member, message):
    import h5py

    path = tmp_path / "m.h5"
    write_mojito_file(path)
    with h5py.File(path, "r+") as f:
        if member == "laser_frequency":
            del f.attrs[member]
        else:
            del f[member]
    with pytest.raises(ValueError, match=message):
        load_mojito_l1(path, orbit=False)


def test_file_without_tdis_is_refused_as_not_mojito(tmp_path):
    import h5py

    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("unrelated", data=np.arange(3))
    with pytest.raises(ValueError, match="is it a Mojito L1 file"):
        load_mojito_l1(path)


def test_flagged_samples_are_refused(tmp_path):
    path = tmp_path / "flagged.h5"
    write_mojito_file(path, flag_index=2000)
    with pytest.raises(ValueError, match=r"1 of 4096 .*sample 2000"):
        load_mojito_l1(path, orbit=False)
    with pytest.raises(ValueError, match="gap mask"):
        load_mojito_l1(path, start_offset_s=400.0, duration_s=200.0, orbit=False)
    data = load_mojito_l1(path, duration_s=400.0, orbit=False)
    assert data.num_time_samples == 1600


def test_orbit_is_attached_in_the_ecliptic_frame(mojito_file):
    path, _, _ = mojito_file
    data = load_mojito_l1(path)
    orbit = data.orbit_ephemeris
    assert isinstance(orbit, NumericOrbit)
    assert orbit.time_range_gps == (
        MOJITO_ORBIT_T0,
        MOJITO_ORBIT_T0 + (MOJITO_ORBIT_SIZE - 1) * MOJITO_ORBIT_DT,
    )
    nodes = MOJITO_ORBIT_T0 + np.arange(3) * MOJITO_ORBIT_DT
    _, _, z = orbit.positions(nodes)
    np.testing.assert_allclose(z, MOJITO_ECLIPTIC_Z_M, rtol=1e-6)
    assert load_mojito_l1(path, orbit=False).orbit_ephemeris is None


REAL_FILE = os.environ.get("ENCHILADA_MOJITO_L1")


@pytest.mark.skipif(not REAL_FILE, reason="set ENCHILADA_MOJITO_L1 to a Mojito L1 file")
def test_real_file_one_hour_window():
    import h5py

    with h5py.File(REAL_FILE, "r") as f:
        t0 = float(f["tdis/sampling"].attrs["t0"])
    data = load_mojito_l1(REAL_FILE, duration_s=3600.0)
    assert data.num_time_samples == 14400
    assert data.sample_rate_hz == 4.0
    assert data.start_time_gps == t0
    for name in data.channel_names:
        rms = float(np.sqrt(np.mean(data.channel_data[name] ** 2)))
        assert 1e-19 < rms < 1e-17, (name, rms)
    xyz = load_mojito_l1(
        REAL_FILE, channels=("X", "Y", "Z"), duration_s=3600.0, orbit=False
    )
    expected_a = (xyz.channel_data["Z"] - xyz.channel_data["X"]) / np.sqrt(2.0)
    # Absolute floor for zero crossings: rtol alone fails where A passes through 0.
    np.testing.assert_allclose(
        data.channel_data["A"],
        expected_a,
        rtol=1e-12,
        atol=1e-12 * np.max(np.abs(expected_a)),
    )
    first, last = data.orbit_ephemeris.time_range_gps
    assert first <= data.t0 and last >= data.t0 + (data.N - 1) * data.dt
    coarse = decimate(data, 10)
    assert coarse.num_time_samples == 1440
    assert coarse.sample_rate_hz == 0.4
    assert coarse.start_time_gps == t0
