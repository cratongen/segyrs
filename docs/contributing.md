# Contributing

## Issues first

Issues are open: report a bug, ask about a file that does not open, or propose a change at [GitHub issues](https://github.com/cratongen/segyrs/issues/new/choose). A code contribution starts with an issue. Describe the change there and wait for an answer before you open a pull request; a pull request that links no accepted issue is closed. [CONTRIBUTING.md](https://github.com/cratongen/segyrs/blob/main/CONTRIBUTING.md) has the steps.

Never attach or commit a SEG-Y file you may not share. Tests and examples use files made by `segyrs.create` or small public files whose licence allows redistribution.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
type(scope): summary
```

`feat(core):` for the reader, `fix(py):` for the bindings, `docs:` for these pages, `chore:` for toolchain pins. A break takes `!` after the type, or a `BREAKING CHANGE:` footer.

Do not add an AI trailer. Do not add a `Co-Authored-By` trailer.

## Clean room

Write the reader and the writer from the SEG-Y standards: rev 0 (1975), rev 1 (May 2002), rev 2.0 (January 2017), and rev 2.1 (October 2023), as published by the SEG Technical Standards Committee.

segyio is a behavioural reference. Read its documentation and its GitHub issues. Do not read, copy, translate, or paraphrase segyio source.

Upstream problems are tracked as GitHub issues on this repository. Each title starts with a ledger id such as `[S-01]` or `[X-10]`. Name the test after that id (`test_s01`, `test_x16`).

## Tasks

Every repeatable command is a mise task. Do not call `cargo`, `python`, `pip`, or `uv` on their own.

- `mise install` installs the pinned toolchain.
- `mise run build` builds the segyrs extension (with object-store reads and Zarr export) and the experimental `segyrs-view` extension into `.venv`. The viewer is not released yet, so this is the way to get it.
- `mise run test` runs the Rust tests and the Python tests. The Python tests require the viewer that `mise run build` installs.
- `mise run lint` runs `cargo clippy` with warnings denied, `cargo fmt --check`, and ruff.
- `mise run test-interop` runs the segyrs-interop tests with the `cloud` and `zarr` features.
- `mise run wheel` builds the release abi3 wheel of segyrs into `dist/`. `mise run test-wheel [python]` installs it in a fresh venv (default: the 3.11 floor) and runs pytest on it alone.
- `mise run sdist` builds the segyrs sdist. `mise run test-sdist` builds segyrs from it in a fresh venv, opens a fixture by path and as a `file://` URL, and exports it to Zarr.
- `mise run check-msrv` checks `segyrs` and `segyrs-draw` with Rust 1.85.1, and `segyrs-py` with Rust 1.89.0, the oldest Rust the segyrs sdist supports.
- `mise run lint-ci` runs actionlint and shellcheck over `.github/workflows`.
- `mise run example` runs the worked example in `examples/`, which asserts its own results, redraws its figures into `docs/examples/img`, and fails when they differ from the committed ones. `mise run test` checks that the example page quotes the script and its output verbatim.
- `mise run docs` serves this site with live reload. `mise run docs-build` builds it with `mkdocs build --strict`, the check the docs workflow runs.
- `mise run bench-write <dir>` runs the write and in-place edit benchmarks on the disk that holds `<dir>`. Run it when a change touches a hot path, or when a measurement was asked for.
- `mise run test-ibm-exhaustive` checks IBM encode and decode over all 2^32 bit patterns.
- `mise run test-data` checks large local SEG-Y files in `$SEGYRS_DATA` (default `../segyrs-data`) against an independent decoder and segyio: sampled traces and headers, lines, labels, whole-file and cropped round trips, and 1 against many threads. It never reads a whole file into memory, but it writes copies of up to 1.3 GB under `target/`; run it under a memory limit. `mise run test` skips these files. The test dependency segyio is installed by `mise run build` and its tests are skipped when it is missing.
- `SEGYRS_FUZZ_CASES=3000 mise run test-rs` runs the seeded mutation fuzz of the reader with more cases per fixture (default 150).

A number you report comes from a benchmark you ran: before and after, same machine, same session, interleaved rounds, medians, with the load written down. When you did not measure, write "not measured". [Benchmarks](benchmarks.md) shows the format.

Write docs in the voice of the existing pages: short declarative sentences, the reader as "you", no em dashes, and facts checked against the code. The look of the site and the logo files follow [Brand](brand.md).

## Files

Leave the fixtures in `test-data/` unchanged. Add a new fixture beside them. Do not commit a large `.sgy`. Generate it in the test, or keep it outside the repository.

An `unsafe` block needs a `// SAFETY:` comment that says why the block is sound.

A hot path uses rayon or a memory map, in the process that called open. segyrs does not start a process pool at import or at open.
