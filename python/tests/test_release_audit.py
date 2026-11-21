"""Release audit on the fixtures: segyrs against an independent decoder, against
segyio when it is installed, label selection against the headers, and
thread-count independence through the Python API."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import segyrs
from segyrs import _segyrs

sys.path.insert(0, str(Path(__file__).parent))
import _oracle

DATA = Path(__file__).resolve().parents[2] / "test-data"
FIXTURES = sorted(p.relative_to(DATA).as_posix() for p in DATA.rglob("*.sgy"))

# Fixtures segyrs refuses on purpose (docs/writing.md), with the error it raises.
REFUSED = {
    "broken.sgy": segyrs.SampleCountMismatch,
    "decrement.sgy": segyrs.NonUniformTraceLength,
    "decrement-lsb.sgy": segyrs.NonUniformTraceLength,
    "increment.sgy": segyrs.NonUniformTraceLength,
    "increment-lsb.sgy": segyrs.NonUniformTraceLength,
    "long.sgy": segyrs.NonUniformTraceLength,
    "mapping-empty.sgy": segyrs.SampleCountMismatch,
    "mapping-invalid-value-size-scale6.sgy": segyrs.SampleCountMismatch,
    "mapping-shifted.sgy": segyrs.GeometryError,
}
# The Format4 fixtures hold IEEE float words under format code 4, so they
# cannot check the fixed-point decoder.
NOT_FORMAT_4 = {"multiformats/Format4lsb.sgy", "multiformats/Format4msb.sgy"}
END_TEXT = "stanzas-unknown-count.sgy"


def _order(name: str) -> str:
    return "<" if "lsb" in name else ">"


def _all(f: _segyrs.File) -> np.ndarray:
    return np.asarray(f.read(np.arange(f.n_traces, dtype=np.int64), 0, f.n_samples))


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_matches_independent_decoder(name):
    if name in NOT_FORMAT_4:
        pytest.skip("no genuine format 4 fixture")
    if name in REFUSED:
        with pytest.raises(REFUSED[name]):
            segyrs.open(DATA / name)
        return
    lay, want, headers = _oracle.read_file(DATA / name, _order(name))
    f = _segyrs.File(str(DATA / name), scan="full")
    assert (f.n_traces, f.n_samples, f.data_start) == (lay.n_traces, lay.ns, lay.data_start)
    got = _all(f)
    assert got.dtype == want.dtype
    assert got.tobytes() == want.tobytes()
    for t in range(f.n_traces):
        assert f.header_bytes(t) == headers[t, :240].tobytes()


def test_end_text_in_any_case_and_spacing_ends_the_extended_headers():
    lay, want, _ = _oracle.read_file(DATA / END_TEXT, ">")
    f = _segyrs.File(str(DATA / END_TEXT), scan="full")
    assert len(f.textual_header()) == 4
    assert (f.data_start, f.n_traces) == (lay.data_start, lay.n_traces)
    assert _all(f).tobytes() == want.tobytes()


SEGYIO_FORMATS = {1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 16}


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_matches_segyio(name):
    segyio = pytest.importorskip("segyio")
    if name in REFUSED or name in NOT_FORMAT_4:
        pytest.skip("not opened by segyrs")
    if name == END_TEXT:
        pytest.skip("segyio counts 56 traces; the independent decoder finds 6 after EndText")
    endian = "little" if "lsb" in name else "big"
    f = _segyrs.File(str(DATA / name), scan="full")
    if f.binary_format_code not in SEGYIO_FORMATS:
        pytest.skip("segyio does not decode this format")
    try:
        s = segyio.open(DATA / name, ignore_geometry=True, endian=endian)
    except (RuntimeError, IndexError):
        pytest.skip("segyio does not open it")
    with s:
        assert f.n_traces == s.tracecount
        if f.n_traces == 0:
            return
        assert f.n_samples == len(s.samples)
        theirs = s.trace.raw[:].reshape(f.n_traces, f.n_samples)
        ours = _all(f)
        assert ours.dtype == theirs.dtype
        assert ours.tobytes() == theirs.tobytes()
        assert f.textual_header()[0][2] == (DATA / name).read_bytes()[:3200]
        raw = [f.header_bytes(t) for t in range(f.n_traces)]
        for byte in (1, 9, 21, 37, 71, 115, 181, 185, 189, 193):
            want = s.attributes(byte)[:]
            width = 2 if byte in (71, 115) else 4
            order = "little" if endian == "little" else "big"
            got = [int.from_bytes(r[byte - 1 : byte - 1 + width], order, signed=True) for r in raw]
            assert list(want) == got, byte


def _grid_fixtures():
    out = []
    for name in FIXTURES:
        if name in REFUSED or name in NOT_FORMAT_4:
            continue
        if segyrs.open(DATA / name).geometry.kind not in ("unstructured", "unsorted", "sparse"):
            out.append(name)
    return out


@pytest.mark.parametrize("name", _grid_fixtures())
def test_volume_labels_select_the_trace_with_those_headers(name):
    lay, want, headers = _oracle.read_file(DATA / name, _order(name))
    order = "little" if _order(name) == "<" else "big"

    def word(t, byte):
        return int.from_bytes(headers[t, byte - 1 : byte + 3].tobytes(), order, signed=True)

    by_key = {}
    for t in range(lay.n_traces):
        by_key[(word(t, 189), word(t, 193), word(t, 37))] = want[t]
    vol = segyrs.open(DATA / name)
    cube = np.asarray(vol.numpy())
    il, xl = vol.coords["inline"], vol.coords["crossline"]
    offs = vol.coords["offset"] if "offset" in vol.dims else None
    for i, a in enumerate(il):
        for j, b in enumerate(xl):
            if offs is None:
                key = next(k for k in by_key if k[:2] == (a, b))
                assert cube[i, j].tobytes() == by_key[key].tobytes()
                one = np.asarray(vol.sel(inline=a, crossline=b).numpy())
                assert one.tobytes() == by_key[key].tobytes()
            else:
                for k, c in enumerate(offs):
                    assert cube[i, j, k].tobytes() == by_key[(a, b, c)].tobytes()


_DIGEST = r"""
import hashlib, sys
from pathlib import Path
import numpy as np
import segyrs
h = hashlib.sha256()
for p in sys.argv[1:]:
    v = segyrs.open(p)
    h.update(np.ascontiguousarray(v.numpy()).tobytes())
    for k in sorted(v.headers):
        h.update(np.ascontiguousarray(v.headers[k]).tobytes())
    h.update(repr(sorted(v.stats().items())).encode())
    h.update(np.ascontiguousarray(v.rms()).tobytes())
print(h.hexdigest())
"""


def test_one_thread_and_many_threads_give_identical_bytes_from_python():
    names = ["small.sgy", "small-ps.sgy", "f3-lsb.sgy", "shot-gather.sgy"]
    paths = [str(DATA / n) for n in names]
    digests = []
    for threads in ("1", "7"):
        env = dict(os.environ, RAYON_NUM_THREADS=threads)
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST, *paths],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.append(out.stdout.strip())
    assert digests[0] == digests[1]


def test_python_volume_agrees_with_the_rust_file():
    for name in ("small.sgy", "f3.sgy", "small-ps-dec-il-xl-off.sgy", "shot-gather.sgy"):
        vol = segyrs.open(DATA / name)
        f = _segyrs.File(str(DATA / name), scan="full")
        cube = np.asarray(vol.numpy()).reshape(-1, f.n_samples)
        rust = _all(f)
        assert sorted(map(bytes, cube)) == sorted(map(bytes, rust))
        h = vol.headers
        assert np.array_equal(np.sort(h["inline"]), np.sort(f.scanned_columns()["inline"]))
        assert vol.binary_header_bytes == f.binary_header_bytes()
        assert vol.textual_header_bytes == [raw for _, _, raw in f.textual_header()]


def test_unselected_crossline_major_volume_copies_byte_for_byte(segy, tmp_path):
    il, xl = np.meshgrid(np.arange(1, 4), np.arange(10, 13), indexing="xy")
    il, xl = il.ravel(), xl.ravel()
    samples = np.arange(il.size * 4, dtype=np.float32).reshape(il.size, 4)
    src = segy("xl-major.sgy", samples, inline=il, crossline=xl)
    vol = segyrs.open(src)
    assert vol.geometry.kind == "crossline_major"
    out = tmp_path / "copy.sgy"
    vol.to_segy(out, sync=False)
    assert out.read_bytes() == src.read_bytes()


def _xl_major(segy):
    il, xl = np.meshgrid(np.arange(1, 5), np.arange(10, 15), indexing="xy")
    il, xl = il.ravel(), xl.ravel()
    samples = np.arange(il.size * 4, dtype=np.float32).reshape(il.size, 4)
    return segy("xl-major.sgy", samples, inline=il, crossline=xl), il, xl


@pytest.mark.parametrize("sample_format", [None, "ibm_float32"])
def test_crossline_major_selection_writes_in_file_order(segy, tmp_path, sample_format):
    src, il, xl = _xl_major(segy)
    vol = segyrs.open(src)
    assert vol.geometry.kind == "crossline_major"
    out = tmp_path / "crop.sgy"
    vol.sel(inline=slice(2, 3), crossline=slice(11, 13)).to_segy(
        out, sample_format=sample_format, sync=False
    )
    keep = np.flatnonzero((il >= 2) & (il <= 3) & (xl >= 11) & (xl <= 13))
    got = segyrs.open(out)
    assert got.geometry.kind == "crossline_major"
    assert np.array_equal(got.headers["inline"], il[keep])
    assert np.array_equal(got.headers["crossline"], xl[keep])
    samples = np.arange(il.size * 4, dtype=np.float32).reshape(il.size, 4)
    assert np.array_equal(_all(_segyrs.File(str(out))), samples[keep])


def test_crossline_major_clip_and_filter_keep_file_order(segy, tmp_path):
    src, il, _ = _xl_major(segy)
    vol = segyrs.open(src)
    vol.clip(-1e9, 1e9, tmp_path / "clip.sgy", sync=False)
    assert (tmp_path / "clip.sgy").read_bytes() == src.read_bytes()
    kept = vol.filter(inline=(2, 3))
    assert np.array_equal(kept.coords["trace"], np.flatnonzero((il >= 2) & (il <= 3)))


def test_file_system_errors_name_the_path_and_are_oserrors(tmp_path):
    missing = tmp_path / "surey.sgy"
    with pytest.raises(FileNotFoundError, match="check its spelling") as e:
        segyrs.open(missing)
    assert isinstance(e.value, segyrs.SegyrsError)
    assert e.value.filename == str(missing) and e.value.errno == 2
    with pytest.raises(IsADirectoryError, match="is a directory") as e:
        segyrs.open(tmp_path)
    assert isinstance(e.value, segyrs.SegyrsIOError)
    with pytest.raises(IsADirectoryError):
        segyrs.validate(tmp_path)


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores modes")
def test_edit_of_a_read_only_file_is_a_permission_error(tmp_path):
    path = tmp_path / "ro.sgy"
    path.write_bytes((DATA / "small.sgy").read_bytes())
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="check the permissions") as e:
        segyrs.edit(path)
    assert e.value.filename == str(path)
    assert isinstance(e.value, segyrs.SegyrsError)


def test_files_that_are_not_segy_say_so(tmp_path):
    with pytest.raises(segyrs.NotSegyError, match="Seismic Unix"):
        segyrs.open(DATA / "small.su")
    text = tmp_path / "notes.txt"
    text.write_text("hello world\n")
    with pytest.raises(segyrs.TruncatedError, match="is not SEG-Y"):
        segyrs.open(text)
    rng = np.random.default_rng(7)
    noise = tmp_path / "noise.bin"
    noise.write_bytes(rng.integers(0, 256, 5000, dtype=np.uint8).tobytes())
    with pytest.raises(segyrs.SegyrsError, match=r"not (a )?SEG-Y"):
        segyrs.open(noise)


def test_create_refuses_a_zero_sample_interval(tmp_path):
    path = tmp_path / "z.sgy"
    data = np.zeros((2, 3), np.float32)
    with pytest.raises(segyrs.SpecError, match="sample_interval_us is 0"):
        segyrs.create(path, data, sample_format="ieee_float32", sample_interval_us=0)
    assert not path.exists()
    segyrs.create(path, data, sample_format="ieee_float32", sample_interval_us=4000)
    segyrs.validate(path)


ERRORS = [
    getattr(segyrs, name)
    for name in segyrs.__all__
    if isinstance(getattr(segyrs, name), type) and issubclass(getattr(segyrs, name), BaseException)
]


@pytest.mark.parametrize("cls", ERRORS, ids=lambda c: c.__name__)
def test_every_error_pickles_to_the_same_class(cls):
    import pickle

    assert cls.__module__ in ("segyrs", "segyrs._segyrs")
    back = pickle.loads(pickle.dumps(cls("message")))
    assert type(back) is cls
    assert str(back) == "message"


def test_validation_problems_survive_pickling(tmp_path):
    import pickle

    bad = tmp_path / "bad.sgy"
    bad.write_bytes(b"\0" * 3600 + b"\1" * 7)
    with pytest.raises(segyrs.ValidationError) as info:
        segyrs.validate(bad)
    back = pickle.loads(pickle.dumps(info.value))
    assert type(back) is segyrs.ValidationError
    assert back.problems == info.value.problems
    assert str(back) == str(info.value)


def _open_in_worker(path):
    segyrs.open(path)


def test_an_error_raised_in_a_worker_process_reaches_the_parent(tmp_path):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    bad = tmp_path / "not-segy.sgy"
    bad.write_bytes(b"PK\x03\x04" + b"\0" * 4000)
    with pytest.raises(segyrs.SegyrsError) as here:
        segyrs.open(bad)
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(1, mp_context=spawn) as pool, pytest.raises(type(here.value)) as there:
        pool.submit(_open_in_worker, bad).result()
    assert str(there.value) == str(here.value)


def test_a_missing_label_names_the_labels_that_exist():
    vol = segyrs.open(DATA / "small.sgy")
    with pytest.raises(
        KeyError, match=r"inline=1500 is not a label; the 5 inline labels run 1\.\.5"
    ):
        vol.sel(inline=1500)
    with pytest.raises(KeyError) as info:
        vol.sel_points(inline=np.array([99]), crossline=[20])
    assert "np." not in str(info.value)
    assert "inline=99 is not a label; the 5 inline labels run 1..5" in str(info.value)


@pytest.mark.parametrize(
    "call, hint",
    [
        (lambda v: v.sel(iline=3), "inline"),
        (lambda v: v.isel(xline=0), "crossline"),
        (lambda v: v.isel_points(il=[0]), "inline"),
        (lambda v: v.sel(crosline=21), "crossline"),
    ],
)
def test_a_misspelt_dim_suggests_the_dim(call, hint):
    vol = segyrs.open(DATA / "small.sgy")
    with pytest.raises(ValueError, match=f"did you mean '{hint}'"):
        call(vol)


def test_an_axis_the_file_does_not_have_is_a_geometry_error():
    vol = segyrs.open(DATA / "small.sgy")
    for call in (lambda: vol.sel(offset=1), lambda: vol.isel(trace=0)):
        with pytest.raises(segyrs.GeometryError, match="geometry kind is 'inline_major'"):
            call()


def _to_zarr_with_bad_chunks(vol, tmp):
    return vol.to_zarr(tmp / "z.zarr", chunks=[1])


@pytest.mark.parametrize(
    "call, cls, text",
    [
        (
            lambda v, d: v.filter(nosuch=1),
            segyrs.SegyrsValueError,
            "no standard header field named nosuch",
        ),
        (lambda v, d: v.resort(d / "r.sgy", by="nosuch"), segyrs.SegyrsValueError, "nosuch"),
        (lambda v, d: v.clip(3, 1, d / "c.sgy"), segyrs.SegyrsValueError, "are not ordered"),
        (_to_zarr_with_bad_chunks, segyrs.SegyrsValueError, "3 entries"),
        (lambda v, d: v.to_regular(budget=10), segyrs.BudgetError, "the RAM budget is 10 bytes"),
    ],
)
def test_argument_and_budget_mistakes_are_not_file_system_errors(call, cls, text, tmp_path):
    vol = segyrs.open(DATA / "small.sgy")
    with pytest.raises(cls, match=re.escape(text)) as info:
        call(vol, tmp_path)
    assert isinstance(info.value, segyrs.SegyrsError)
    assert isinstance(info.value, MemoryError if cls is segyrs.BudgetError else ValueError)
    assert not isinstance(info.value, OSError)
    assert not str(info.value).startswith("io error")


@pytest.mark.skipif(
    os.environ.get("SEGYRS_CHECK_BUILD_PATHS") != "1" or "site-packages" not in segyrs.__file__,
    reason="checks a wheel from `mise run wheel`, installed by `mise run test-wheel`",
)
def test_an_installed_wheel_carries_no_build_machine_paths():
    native = next(p for p in Path(segyrs.__file__).parent.glob("_segyrs*") if p.suffix != ".pyi")
    data = native.read_bytes()
    for leak in {str(Path.home()), str(Path(__file__).resolve().parents[2])}:
        assert leak.encode() not in data, f"{native.name} holds {leak}"


@pytest.mark.parametrize("extra", [1, 239, 241])
def test_trailing_bytes_raise_and_the_documented_cut_opens_the_traces(tmp_path, extra):
    src = tmp_path / "padded.sgy"
    segyrs.create(
        src,
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        sample_format="ieee_float32",
        sample_interval_us=4000,
    )
    expected = src.stat().st_size
    with src.open("ab") as f:
        f.write(b"\0" * extra)
    with pytest.raises(segyrs.SampleCountMismatch) as err:
        segyrs.open(src)
    message = str(err.value)
    assert f"expected={expected}" in message and f"file_len={expected + extra}" in message
    cut = tmp_path / "cut.sgy"
    cut.write_bytes(src.read_bytes())
    os.truncate(cut, int(re.search(r"expected=(\d+)", message).group(1)))
    assert np.array_equal(
        segyrs.open(cut).numpy(), np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    )
