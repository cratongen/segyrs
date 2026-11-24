"""Protocol version 1 tests for ``seismic_volume`` and ``prestack_volume``, and with geophysics."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DATA, grid

import segyrs

COMMON_KEYS = {"protocol", "kind", "columns", "units", "categories", "crs", "length_unit"}
SEISMIC_KEYS = COMMON_KEYS | {
    "data",
    "inlines",
    "crosslines",
    "samples",
    "trace_mask",
    "affine",
    "cdp_x",
    "cdp_y",
    "vertical_domain",
    "sample_unit",
    "attrs",
}
PRESTACK_KEYS = SEISMIC_KEYS | {"layout", "offsets", "source_coords", "receiver_coords"}


class Carry:
    """Foreign producer wrapper exposing ``__minetypes__()``."""

    def __init__(self, payload: dict):
        self._payload = payload

    def __minetypes__(self) -> dict:
        return self._payload


def ro(arr: np.ndarray) -> np.ndarray:
    """Return a C-contiguous, native-endian, read-only array."""
    out = np.ascontiguousarray(arr)
    out.setflags(write=False)
    return out


def make_seismic_volume(**overrides) -> dict:
    data = ro(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    base = {
        "protocol": 1,
        "kind": "seismic_volume",
        "columns": {},
        "units": {},
        "categories": {},
        "crs": "EPSG:31982",
        "length_unit": "m",
        "data": data,
        "inlines": ro(np.array([100, 110], dtype=np.int64)),
        "crosslines": ro(np.array([20, 21, 22], dtype=np.int64)),
        "samples": ro(np.array([0.0, 2000.0, 4000.0, 6000.0], dtype=np.float64)),
        "trace_mask": ro(np.ones((2, 3), dtype=bool)),
        "affine": (500_000.0, 25.0, 0.0, 7_500_000.0, 0.0, 12.5),
        "cdp_x": None,
        "cdp_y": None,
        "vertical_domain": "time",
        "sample_unit": "us",
        "attrs": {"survey": "synthetic"},
    }
    base.update(overrides)
    return base


def make_prestack_grid(**overrides) -> dict:
    data = ro(np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4))
    base = {
        "protocol": 1,
        "kind": "prestack_volume",
        "layout": "grid",
        "columns": {},
        "units": {},
        "categories": {},
        "crs": None,
        "length_unit": "ft",
        "data": data,
        "inlines": ro(np.array([10, 20], dtype=np.int64)),
        "crosslines": ro(np.array([1, 2, 3], dtype=np.int64)),
        "offsets": ro(np.array([50.0, 150.0], dtype=np.float64)),
        "samples": ro(np.array([10.0, 12.0, 14.0, 16.0], dtype=np.float64)),
        "trace_mask": ro(np.ones((2, 3, 2), dtype=bool)),
        "affine": (1000.0, 25.0, 0.0, 2000.0, 0.0, 25.0),
        "cdp_x": None,
        "cdp_y": None,
        "source_coords": None,
        "receiver_coords": None,
        "vertical_domain": "time",
        "sample_unit": "ms",
        "attrs": {},
    }
    base.update(overrides)
    return base


def make_prestack_traces(**overrides) -> dict:
    data = ro(np.arange(3 * 5, dtype=np.float64).reshape(3, 5))
    base = {
        "protocol": 1,
        "kind": "prestack_volume",
        "layout": "traces",
        "columns": {},
        "units": {},
        "categories": {},
        "crs": "EPSG:31982",
        "length_unit": "m",
        "data": data,
        "inlines": ro(np.array([5, 2, 5], dtype=np.int64)),
        "crosslines": ro(np.array([10, 30, 20], dtype=np.int64)),
        "offsets": ro(np.array([100.0, 200.0, 300.0], dtype=np.float64)),
        "samples": ro(np.array([0.0, 0.002, 0.004, 0.006, 0.008], dtype=np.float64)),
        "trace_mask": ro(np.array([True, False, True], dtype=bool)),
        "affine": None,
        "cdp_x": ro(np.array([500.5, np.nan, 520.5], dtype=np.float64)),
        "cdp_y": ro(np.array([1000.25, np.nan, 1020.25], dtype=np.float64)),
        "source_coords": ro(
            np.array(
                [[450.5, 1000.25, -5.0], [0.0, 0.0, 0.0], [370.5, 1020.25, -5.0]],
                dtype=np.float64,
            )
        ),
        "receiver_coords": ro(
            np.array(
                [[550.5, 1000.25, 0.0], [0.0, 0.0, 0.0], [670.5, 1020.25, 0.0]],
                dtype=np.float64,
            )
        ),
        "vertical_domain": "time",
        "sample_unit": "s",
        "attrs": {"component": "vz"},
    }
    base.update(overrides)
    return base


def _assert_protocol_arrays(d: dict) -> None:
    for key, val in d.items():
        if isinstance(val, np.ndarray):
            assert val.flags.c_contiguous, f"{key} is not C-contiguous"
            assert val.dtype.isnative, f"{key} is not native-endian"
            assert not val.flags.writeable, f"{key} is writeable"
            if val.dtype == np.bool_:
                assert np.all(val.view(np.uint8) <= 1), f"{key} has invalid bool bytes"


def test_produce_seismic_volume_contract_and_views():
    vol = segyrs.open(DATA / "small.sgy")
    d = vol.__minetypes__()
    assert set(d) == SEISMIC_KEYS
    assert type(d["protocol"]) is int and d["protocol"] == 1
    assert d["kind"] == "seismic_volume"
    assert d["columns"] == {} and d["units"] == {} and d["categories"] == {}
    assert d["crs"] is None and d["length_unit"] is None
    assert d["affine"] is None
    assert d["vertical_domain"] == "time" and d["sample_unit"] == "us"
    assert d["attrs"] == {}
    _assert_protocol_arrays(d)

    assert d["data"].shape == (5, 5, 50) and d["data"].dtype == np.float32
    assert d["inlines"].shape == (5,) and d["inlines"].dtype == np.int64
    assert d["crosslines"].shape == (5,) and d["crosslines"].dtype == np.int64
    assert d["samples"].shape == (50,) and d["samples"].dtype == np.float64
    assert d["trace_mask"].shape == (5, 5) and d["trace_mask"].dtype == np.bool_
    assert d["cdp_x"].shape == (5, 5) and d["cdp_x"].dtype == np.float64
    assert d["cdp_y"].shape == (5, 5) and d["cdp_y"].dtype == np.float64
    np.testing.assert_array_equal(d["data"], vol.numpy())
    np.testing.assert_array_equal(d["inlines"], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(d["crosslines"], [20, 21, 22, 23, 24])
    np.testing.assert_array_equal(d["samples"], np.arange(50, dtype=np.float64) * 4000.0)
    assert d["trace_mask"].all()

    # Stored float64 coordinate and bool mask buffers are exposed as read-only views.
    d2 = vol.__minetypes__()
    assert np.shares_memory(d["cdp_x"], d2["cdp_x"])
    assert np.shares_memory(d["cdp_y"], d2["cdp_y"])
    assert np.shares_memory(d["trace_mask"], d2["trace_mask"])
    assert np.shares_memory(vol._src.trace_coords()["trace_mask"], d["trace_mask"])
    # Decoded file samples are a fresh materialised buffer per call.
    assert not np.shares_memory(d["data"], d2["data"])
    with pytest.raises(ValueError):
        d["data"][0, 0, 0] = 1.0


def test_produce_selection_and_budget_check():
    vol = segyrs.open(DATA / "small.sgy")
    sub = vol.sel(inline=slice(2, 4), crossline=[21, 23], time=slice(8000, 20000))
    assert vol.traces_read == 0
    with pytest.raises(segyrs.BudgetError, match="needs 96 bytes, the RAM budget is 64 bytes"):
        sub.__minetypes__(budget=64)
    # Refused before decoding any sample bytes.
    assert vol.traces_read == 0

    d = sub.__minetypes__(budget=96)
    _assert_protocol_arrays(d)
    assert d["data"].shape == (3, 2, 4)
    assert d["data"].nbytes == sub.nbytes == 96
    np.testing.assert_array_equal(d["inlines"], [2, 3, 4])
    np.testing.assert_array_equal(d["crosslines"], [21, 23])
    np.testing.assert_array_equal(d["samples"], [8000.0, 12000.0, 16000.0, 20000.0])
    np.testing.assert_array_equal(d["data"], sub.numpy())

    # Decreasing line selection is still strictly monotonic and allowed.
    rev_il = vol.isel(inline=slice(4, 1, -1)).__minetypes__()
    np.testing.assert_array_equal(rev_il["inlines"], [5, 4, 3])


def test_produce_prestack_grid_and_traces(segy):
    ps = segyrs.open(DATA / "small-ps.sgy")
    dg = ps.__minetypes__()
    assert set(dg) == PRESTACK_KEYS
    assert dg["kind"] == "prestack_volume" and dg["layout"] == "grid"
    assert dg["source_coords"] is None and dg["receiver_coords"] is None
    assert dg["affine"] is None
    _assert_protocol_arrays(dg)
    assert dg["data"].shape == ps.shape
    assert dg["offsets"].dtype == np.float64 and dg["offsets"].shape == (ps.sizes["offset"],)
    assert dg["trace_mask"].shape == ps.shape[:-1]
    assert dg["cdp_x"].shape == (ps.sizes["inline"], ps.sizes["crossline"])

    # Selecting one offset collapses a 4D pre-stack grid to a 3D seismic_volume.
    one_off = ps.isel(offset=0).__minetypes__()
    assert one_off["kind"] == "seismic_volume" and set(one_off) == SEISMIC_KEYS

    # Unstructured / filtered traces produce layout="traces".
    tr_vol = ps.filter(inline=(1, 2))
    dt = tr_vol.__minetypes__()
    assert set(dt) == PRESTACK_KEYS
    assert dt["kind"] == "prestack_volume" and dt["layout"] == "traces"
    _assert_protocol_arrays(dt)
    n_tr = tr_vol.sizes["trace"]
    assert dt["data"].shape == (n_tr, ps.sizes["time"])
    assert dt["inlines"].shape == (n_tr,) and dt["inlines"].dtype == np.int64
    assert dt["crosslines"].shape == (n_tr,) and dt["crosslines"].dtype == np.int64
    assert dt["offsets"].shape == (n_tr,) and dt["offsets"].dtype == np.float64
    assert dt["trace_mask"].shape == (n_tr,)
    assert dt["cdp_x"].shape == (n_tr,) and dt["cdp_y"].shape == (n_tr,)
    assert dt["source_coords"].shape == (n_tr, 3) and dt["source_coords"].dtype == np.float64
    assert dt["receiver_coords"].shape == (n_tr, 3) and dt["receiver_coords"].dtype == np.float64

    # Conflicting cdp_x within one (inline, crossline) bin across offsets is rejected.
    samples, il, xl = grid(2, 2, ns=3)
    il4 = np.repeat(il, 2)
    xl4 = np.repeat(xl, 2)
    off4 = np.tile([10, 20], 4)
    cdpx = np.array([100, 101, 200, 200, 300, 300, 400, 400], dtype=np.int32)
    path = segy(
        "bad_bin_cdp.sgy",
        np.repeat(samples, 2, axis=0),
        inline=il4,
        crossline=xl4,
        offset=off4,
        words={(181, "i4"): cdpx},
    )
    with pytest.raises(segyrs.SegyrsValueError, match="cdp_x"):
        segyrs.open(path).__minetypes__()


@pytest.mark.parametrize(
    ("fmt", "dtype"),
    [
        (2, np.int32),
        (3, np.int16),
        (5, np.float32),
        (6, np.float64),
        (8, np.int8),
    ],
)
def test_produce_preserves_allowed_dtypes_and_decodes_foreign_endian(segy, fmt, dtype):
    samples, il, xl = grid(2, 2, ns=4)
    arr = samples.astype(dtype)
    path = segy(f"proto_{fmt}.sgy", arr, fmt=fmt, inline=il, crossline=xl, little=True)
    d = segyrs.open(path).__minetypes__()
    assert d["data"].dtype == dtype
    assert d["data"].dtype.isnative
    np.testing.assert_array_equal(d["data"], arr.reshape(2, 2, 4))


def test_produce_rejects_bad_selections_dtypes_and_live_inf(segy):
    vol = segyrs.open(DATA / "small.sgy")
    with pytest.raises(segyrs.SegyrsValueError, match="sample axis"):
        vol.isel(time=0).__minetypes__()
    with pytest.raises(segyrs.SegyrsValueError, match="kind"):
        vol.isel(inline=0).__minetypes__()
    with pytest.raises(segyrs.SegyrsValueError, match="kind"):
        vol.isel_points(inline=[0, 1], crossline=[0, 1]).__minetypes__()
    with pytest.raises(segyrs.SegyrsValueError, match="inlines"):
        vol.isel(inline=[2, 0, 1]).__minetypes__()
    with pytest.raises(segyrs.SegyrsValueError, match="samples"):
        vol.isel(time=slice(4, 1, -1)).__minetypes__()

    # uint16 is not a protocol v1 seismic dtype.
    samples, il, xl = grid(2, 2, ns=3)
    u16_path = segy("u16.sgy", samples.astype(np.uint16), fmt=11, inline=il, crossline=xl)
    with pytest.raises(segyrs.SegyrsValueError, match="dtype"):
        segyrs.open(u16_path).__minetypes__()

    # Live trace with IBM overflow (Inf) is rejected; dead trace with Inf is allowed.
    ibm_inf = np.array([[0x7FFFFFFF, 0x41100000], [0x41100000, 0x41100000]], dtype=np.uint32)
    inf_path = segy(
        "inf.sgy",
        ibm_inf,
        fmt=1,
        inline=[1, 1],
        crossline=[10, 11],
        words={(29, "i2"): [2, 1]},
    )
    with pytest.raises(segyrs.SegyrsValueError, match="infinity"):
        segyrs.open(inf_path).__minetypes__()
    dead_ok = segyrs.open(inf_path, dead_trace="trace_id").__minetypes__()
    assert dead_ok["trace_mask"].tolist() == [[False, True]]


def test_consume_seismic_volume_round_trip(tmp_path):
    spec = make_seismic_volume()
    out = tmp_path / "seismic.sgy"
    assert segyrs.create(out, Carry(spec), sample_format="ieee_float32") > 0
    vol = segyrs.open(out, dead_trace="trace_id")
    assert vol.shape == (2, 3, 4)
    assert vol.attrs["sample_interval_us"] == 2000
    assert vol.attrs["measurement_system"] == "meters"
    np.testing.assert_array_equal(vol.numpy(), spec["data"])
    np.testing.assert_array_equal(vol.coords["inline"], [100, 110])
    np.testing.assert_array_equal(vol.coords["crossline"], [20, 21, 22])
    np.testing.assert_array_equal(vol.coords["time"], [0, 2000, 4000, 6000])
    np.testing.assert_allclose(
        vol.coords["cdp_x"],
        [[500_000.0, 500_000.0, 500_000.0], [500_025.0, 500_025.0, 500_025.0]],
    )
    np.testing.assert_allclose(
        vol.coords["cdp_y"],
        [[7_500_000.0, 7_500_012.5, 7_500_025.0], [7_500_000.0, 7_500_012.5, 7_500_025.0]],
    )

    # Rule 4: segyrs.create also accepts a segyrs.Volume directly.
    out_vol = tmp_path / "from_vol.sgy"
    segyrs.create(out_vol, vol, sample_format="ieee_float32")
    np.testing.assert_array_equal(segyrs.open(out_vol).numpy(), spec["data"])

    # Explicit cdp_x/cdp_y with a dead trace holding NaN and Inf sample on the dead trace.
    data_dead = spec["data"].copy()
    data_dead[0, 1, 0] = np.inf
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 1] = False
    cdp_x = np.full((2, 3), 123.25, dtype=np.float64)
    cdp_y = np.full((2, 3), 456.75, dtype=np.float64)
    cdp_x[0, 1] = np.nan
    cdp_y[0, 1] = np.nan
    dead_spec = make_seismic_volume(
        data=ro(data_dead),
        trace_mask=ro(mask),
        affine=None,
        cdp_x=ro(cdp_x),
        cdp_y=ro(cdp_y),
    )
    out_dead = tmp_path / "dead.sgy"
    segyrs.create(out_dead, dead_spec, sample_format="ieee_float32")
    back_dead = segyrs.open(out_dead, dead_trace="trace_id")
    np.testing.assert_array_equal(back_dead.coords["trace_mask"], mask)
    assert back_dead.coords["cdp_x"][0, 0] == pytest.approx(123.25)


def test_consume_prestack_grid_and_traces_round_trip(tmp_path):
    grid_spec = make_prestack_grid()
    out_grid = tmp_path / "ps_grid.sgy"
    segyrs.create(out_grid, grid_spec, sample_format="ieee_float32")
    vg = segyrs.open(out_grid)
    assert vg.dims == ("inline", "crossline", "offset", "time")
    assert vg.shape == (2, 3, 2, 4)
    assert vg.attrs["measurement_system"] == "feet"
    np.testing.assert_array_equal(vg.numpy(), grid_spec["data"])
    np.testing.assert_array_equal(vg.coords["inline"], [10, 20])
    np.testing.assert_array_equal(vg.coords["crossline"], [1, 2, 3])
    np.testing.assert_array_equal(vg.coords["offset"], [50, 150])
    np.testing.assert_array_equal(vg.coords["time"], [10_000, 12_000, 14_000, 16_000])

    tr_spec = make_prestack_traces()
    out_tr = tmp_path / "ps_traces.sgy"
    segyrs.create(out_tr, Carry(tr_spec), sample_format="ieee_float64")
    vt = segyrs.open(out_tr, dead_trace="trace_id")
    assert vt.dims == ("trace", "time")
    assert vt.shape == (3, 5)
    np.testing.assert_array_equal(vt.numpy(), tr_spec["data"])
    np.testing.assert_array_equal(vt.coords["inline"], [5, 2, 5])
    np.testing.assert_array_equal(vt.coords["crossline"], [10, 30, 20])
    np.testing.assert_array_equal(vt.coords["offset"], [100, 200, 300])
    np.testing.assert_array_equal(vt.coords["trace_mask"], [True, False, True])
    np.testing.assert_allclose(vt.coords["cdp_x"][[0, 2]], [500.5, 520.5])
    np.testing.assert_allclose(vt.coords["source_x"][[0, 2]], [450.5, 370.5])
    np.testing.assert_allclose(vt.coords["group_x"][[0, 2]], [550.5, 670.5])
    np.testing.assert_allclose(vt.coords["source_elev"][[0, 2]], [-5.0, -5.0])

    # Single-sample trace volume requires explicit sample_interval_us.
    one_sample = make_prestack_traces(
        data=ro(np.ones((3, 1), dtype=np.float32)),
        samples=ro(np.array([0.004], dtype=np.float64)),
    )
    with pytest.raises(segyrs.SegyrsValueError, match="sample_interval_us"):
        segyrs.create(tmp_path / "one_fail.sgy", one_sample, sample_format="ieee_float32")
    segyrs.create(
        tmp_path / "one_ok.sgy",
        one_sample,
        sample_format="ieee_float32",
        sample_interval_us=2000,
    )
    assert segyrs.open(tmp_path / "one_ok.sgy").coords["time"].tolist() == [4000]


def test_consumer_rejects_invalid_protocol_dicts(tmp_path):
    dest = tmp_path / "bad.sgy"

    for key in SEISMIC_KEYS:
        broken = make_seismic_volume()
        del broken[key]
        with pytest.raises(segyrs.SegyrsValueError, match=key):
            segyrs.create(dest, broken, sample_format="ieee_float32")

    for key in ("layout", "offsets", "source_coords", "receiver_coords"):
        broken = make_prestack_grid()
        del broken[key]
        with pytest.raises(segyrs.SegyrsValueError, match=key):
            segyrs.create(dest, broken, sample_format="ieee_float32")

    cases = [
        (make_seismic_volume(protocol=2), "protocol"),
        (make_seismic_volume(protocol=True), "protocol"),
        (make_seismic_volume(kind="grid"), "kind"),
        (make_seismic_volume(columns={"a": ro(np.zeros(1))}), "columns"),
        (make_seismic_volume(units={"a": "m"}), "units"),
        (make_seismic_volume(categories={"a": {}}), "categories"),
        (make_seismic_volume(crs="   "), "crs"),
        (make_seismic_volume(length_unit=""), "length_unit"),
        (make_seismic_volume(attrs={"bad": float("nan")}), "attrs"),
        (make_seismic_volume(vertical_domain="frequency"), "vertical_domain"),
        (make_seismic_volume(vertical_domain="depth", sample_unit="m"), "sample_unit"),
        (make_seismic_volume(sample_unit="m"), "sample_unit"),
        (make_seismic_volume(data=ro(np.zeros((2, 3, 4), dtype=np.uint16))), "data"),
        (make_seismic_volume(data=ro(np.zeros((2, 3), dtype=np.float32))), "data"),
        (
            make_seismic_volume(data=np.zeros((4, 3, 4), dtype=np.float32)[::2]),
            "data",
        ),
        (
            make_seismic_volume(data=np.zeros((2, 3, 4), dtype=">f4")),
            "data",
        ),
        (
            make_seismic_volume(data=ro(np.full((2, 3, 4), np.inf, dtype=np.float32))),
            "data",
        ),
        (
            make_seismic_volume(inlines=ro(np.array([10, 10], dtype=np.int64))),
            "inlines",
        ),
        (
            make_seismic_volume(inlines=ro(np.array([10, 2**32], dtype=np.int64))),
            "inlines",
        ),
        (
            make_seismic_volume(crosslines=ro(np.array([2, 1, 3], dtype=np.int64))),
            "crosslines",
        ),
        (
            make_seismic_volume(samples=ro(np.array([0.0, 2.0, 2.0, 4.0], dtype=np.float64))),
            "samples",
        ),
        (
            make_seismic_volume(samples=ro(np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64))),
            "samples",
        ),
        (
            make_seismic_volume(samples=ro(np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64))),
            "samples",
        ),
        (
            make_seismic_volume(
                trace_mask=np.frombuffer(b"\x01\x02\x01\x01\x01\x01", dtype=bool).reshape(2, 3)
            ),
            "trace_mask",
        ),
        (make_seismic_volume(affine=(1.0, 2.0, 3.0)), "affine"),
        (make_seismic_volume(affine=(True, 0.0, 0.0, 0.0, 0.0, 0.0)), "affine"),
        (make_seismic_volume(cdp_x=ro(np.zeros((2, 3))), cdp_y=None), "cdp_x"),
        (
            make_seismic_volume(cdp_x=ro(np.zeros((2, 3))), cdp_y=ro(np.zeros((2, 3)))),
            "affine",
        ),
        (
            make_seismic_volume(
                affine=None,
                cdp_x=ro(np.full((2, 3), np.nan)),
                cdp_y=ro(np.zeros((2, 3))),
            ),
            "cdp_x",
        ),
        (make_prestack_grid(layout=" cube "), "layout"),
        (
            make_prestack_grid(offsets=ro(np.array([10.5, 20.5], dtype=np.float64))),
            "offsets",
        ),
        (
            make_prestack_grid(source_coords=ro(np.zeros((2, 3), dtype=np.float64))),
            "source_coords",
        ),
        (make_prestack_traces(affine=(0.0, 1.0, 0.0, 0.0, 0.0, 1.0)), "affine"),
        (
            make_prestack_traces(source_coords=ro(np.full((3, 3), np.nan, dtype=np.float64))),
            "source_coords",
        ),
    ]
    for bad, match in cases:
        with pytest.raises(segyrs.SegyrsValueError, match=match):
            segyrs.create(dest, bad, sample_format="ieee_float32")

    with pytest.raises(segyrs.SegyrsValueError, match="sample_interval_us"):
        segyrs.create(
            dest,
            make_seismic_volume(),
            sample_format="ieee_float32",
            sample_interval_us=9999,
        )
    with pytest.raises(segyrs.SegyrsValueError, match="protocol object"):
        segyrs.create(
            dest,
            make_seismic_volume(),
            sample_format="ieee_float32",
            inline=[1, 2],
        )
    with pytest.raises(segyrs.SegyrsValueError, match="must return a dict"):
        segyrs.create(dest, Carry(None), sample_format="ieee_float32")  # type: ignore[arg-type]


def test_geophysics_synthetic_cube_writes_to_segy_and_reads_back_equal(tmp_path):
    gp = pytest.importorskip("geophysics")
    grid = gp.TensorGrid.regular([1000.0, 2000.0, -100.0], [25.0, 12.5, 10.0], [3, 4, 2])
    if not hasattr(grid, "seismic_volume"):
        pytest.skip("installed geophysics TensorGrid has no seismic_volume")
    cube = np.linspace(-1.5, 2.25, 3 * 4 * 8, dtype=np.float32).reshape(3, 4, 8)
    vol_proto = grid.seismic_volume(cube, 2e-3, t0=0.01)
    path = tmp_path / "synthetic_cube.sgy"
    segyrs.create(path, vol_proto, sample_format="ieee_float32")

    opened = segyrs.open(path)
    assert opened.dims == ("inline", "crossline", "time")
    assert opened.shape == (3, 4, 8)
    np.testing.assert_array_equal(opened.numpy(), cube)
    np.testing.assert_array_equal(opened.coords["inline"], [1, 2, 3])
    np.testing.assert_array_equal(opened.coords["crossline"], [1, 2, 3, 4])
    np.testing.assert_array_equal(opened.coords["time"], 10_000 + np.arange(8) * 2_000)


def test_geophysics_gather_round_trips_with_segyrs(tmp_path):
    gp = pytest.importorskip("geophysics")
    data = np.arange(12.0, dtype=np.float64).reshape(3, 4)
    src = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    rcv = np.array([[100.0, 0.0, 0.0], [110.0, 0.0, 0.0], [220.0, 0.0, 0.0]])
    gather = gp.Gather(data, 2e-3, t0=0.01, sources=src, receivers=rcv, component="vz")
    if not hasattr(gather, "__minetypes__"):
        pytest.skip("installed geophysics Gather has no __minetypes__")
    path = tmp_path / "gather.sgy"
    segyrs.create(path, gather, sample_format="ieee_float64")

    opened = segyrs.open(path)
    np.testing.assert_array_equal(opened.numpy(), data)
    rebuilt = gp.Gather(opened, component="vz")
    np.testing.assert_allclose(rebuilt.data, gather.data)
    np.testing.assert_allclose(rebuilt.sources, gather.sources)
    np.testing.assert_allclose(rebuilt.receivers, gather.receivers)
    assert rebuilt.dt == pytest.approx(gather.dt)
    assert rebuilt.t0 == pytest.approx(gather.t0)
