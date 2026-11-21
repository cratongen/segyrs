# Benchmark harness

The scripts behind every number on the [Benchmarks](https://cratongen.github.io/segyrs/benchmarks/)
page, kept exactly as they ran on 2026-11-17. They are not part of the package and
not part of `mise run test`.

Each script reads and writes next to itself, so copy this folder to a disk with
room for the data (19 GB for the full set) and run it there, never inside the checkout.
`gen_matern.py` holds the whole field in memory: a 1000 x 1000 x 1000 field is 4 GB
of float32 plus its FFT, so check the free memory first or pass a smaller size.

| Script | Makes or measures |
| --- | --- |
| `gen_matern.py [n] [ns]` | One seeded anisotropic Matern random field (seed 20261117), written as three SEG-Y files: IEEE big-endian inline-sorted, IBM big-endian crossline-sorted, int16 little-endian rev 2. `1000` gives the 4.24 GB files, `1000 100` the 0.64 GB ones in `1000x1000x100/`. Needs numpy and scipy. |
| `gen_irregular.py` | The four irregular files and `truth.npz` in `irregular/`, cut from the cached `1000x1000x100` field. |
| `bench_vs_segyio.py PYTHON ROUNDS [OUT.json]` | The regular-file tables. Set `SEGYIO_MMAP=1` for the segyio-with-`mmap` side. |
| `bench_irregular.py PYTHON ROUNDS` | The irregular-file tables. |
| `bench_write_vs_segyio.py PYTHON ROUNDS SRC OUTDIR` | The writing table. |
| `check_irregular.py` | The correctness table, against `truth.npz`. |

`PYTHON` is an interpreter with segyrs and segyio installed. The published run used
segyio 1.9.15, numpy 2.5.5 and segyrs 0.1.0 from the checkout of that day in one environment, with 5 rounds. The commands, from the usage line of each script:

```bash
python gen_matern.py 1000
python gen_matern.py 1000 100
python gen_irregular.py
SEGYIO_MMAP=1 python bench_vs_segyio.py "$PY" 5 bench_mmap.json
python bench_vs_segyio.py "$PY" 5 bench_default.json
python bench_irregular.py "$PY" 5
python bench_write_vs_segyio.py "$PY" 5 1000x1000x100/matern_ieee_be_inline.sgy out
python check_irregular.py
```

Run them through the project's mise toolchain, for example
`mise exec -- uv run --no-project --with numpy --with scipy python gen_matern.py 1000`.
