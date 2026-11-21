# Release checklist

The first public release is segyrs 0.1.0 on PyPI. Versions follow semantic
versioning. This page records the go/no-go check, what is still open, the
decisions only the owner can take, and the steps the owner does by hand.

## Packaging decision of 2026-11-21

0.1.0 publishes one PyPI project, `segyrs`. Object-store reads
(`segyrs.open` on URLs) and Zarr export (`Volume.to_zarr`) are built into its
wheel; there are no extras. The native viewer window (`segyrs-view`,
`crates/segyrs-view-py`) stays in the repository, builds with
`mise run build` and is tested there, but it is not part of the release and
will be published later. This replaces the decision of 2026-11-20 (core by
default, viewer, cloud and Zarr as extras in three projects).

## Verdict: Linux x86_64 and the sdist can go up by hand; the rest waits for Actions

State on 2026-11-21:

- **Done by the owner.** The git history is decided: a fresh history of one
  root commit (`bc78304`), pushed to `main` on 2026-11-21. The repository
  `cratongen/segyrs` is public. The PyPI pending publisher for `segyrs` exists
  (values in the owner steps below). The GitHub environment `pypi` exists,
  limited to tags `v*`, with a required reviewer. Pages is enabled.
- **GitHub Actions are blocked by billing.** GitHub refuses to start any job
  for the `cratongen` organization ("recent account payments have failed or
  your spending limit needs to be increased"). No GitHub run of `ci`, `docs`
  or `release` has ever executed a job. That is the only reason the
  six-platform release matrix has not run, and why the docs site is not
  deployed: Pages is enabled, but nothing has been published to
  `https://cratongen.github.io/segyrs/`, so the README wordmark and the docs
  links on the PyPI page do not resolve until the `docs` workflow runs. This
  is a payment control; it is not worked around (no forks, mirrors or other
  CI).
- **What can go up today.** The sdist and the `manylinux_2_28_x86_64` wheel,
  built and tested on a Linux x86_64 machine (the table
  [Local release build](#local-release-build-2026-11-21) below), uploaded by the
  owner with `mise run publish-local` and a PyPI API token. On every other
  platform `pip install segyrs` then builds from the sdist (Rust 1.89 or
  newer and a C compiler, see
  [What 0.1.0 has on PyPI](install-and-release.md#what-010-has-on-pypi)).
- **Not ready.** The macOS and Windows wheels cannot be built on a Linux
  machine; the musllinux and aarch64 wheels build here but cannot run here
  without a container, so they are untested and not uploaded. They wait for
  the release workflow.

## minetypes adoption: gate lifted on 2026-11-21

The owner lifted the minetypes gate on 2026-11-21: 0.1.0 is published without
it, and segyrs adopts minetypes, the shared types package of the cratongen
family (in the sibling folder `../minetypes`), in a later release. minetypes
is not a dependency of segyrs yet, and no new container type was added.
Adoption follows minetypes' adoption notes and keeps operations as segyrs
functions, not methods added to minetypes types.

Public segyrs types that minetypes will replace as the hand-over format:

| segyrs today | minetypes type | Notes |
|--------------|----------------|-------|
| A `segyrs.Volume` with dims `(inline, crossline, time)` or `depth`, handed over as `Volume.numpy()` plus `coords` and `attrs` | `minetypes.SeismicVolume` | The lazy, file-backed `Volume` stays: it reads on demand. minetypes replaces the in-memory form that goes to other libraries. |
| A `segyrs.Volume` with dims `(inline, crossline, offset, time)` | `minetypes.PrestackVolume`, grid layout | Same rule. |
| A `segyrs.Volume` with dims `(trace, time)` (unsorted, sparse or unstructured files) | `minetypes.PrestackVolume`, traces layout | Same rule. |
| The `trace_mask` coordinate | the `trace_mask` field of both volume types | Same meaning: dead traces and grid gaps, with integer samples kept. |

`segyrs.Geometry`, `segyrs.Picture`, `segyrs.Editor` and the exception
classes have no minetypes counterpart and stay. The Rust crates keep their
own types: minetypes has no SEG-Y concern, so the Rust side needs no change.

## Go/no-go, 2026-11-21

The checks marked with a result ran on Linux x86_64 (Intel i9-13980HX, 30 GB
RAM), one at a time, under a memory limit of 8 GB, with
`CARGO_BUILD_JOBS=4 RUST_TEST_THREADS=4` and `-j 1` passed to each task,
while another build workflow ran on the machine. Commit `b0b04a7` is the
tree they ran on; the commit after it only records the results here.
"Needs a rerun" means the check last ran on 2026-11-20 on the three-project
layout, so its old result says nothing about this tree.

| Check | Command | Result on 2026-11-21 | Commit | Go |
|-------|---------|----------------------|--------|----|
| Lint | `mise run lint` | rustfmt and clippy `-D warnings` clean; ruff: all checks passed, 28 files formatted | `b0b04a7` | go |
| Workflow lint | `mise run lint-ci` | actionlint and shellcheck clean | `b0b04a7` | go |
| Build | `mise run build` | segyrs (with cloud reads and Zarr export) and segyrs-view installed into `.venv` | `b0b04a7` | go |
| Tests | `mise run test` | Rust: 139 passed, 0 failed, 2 ignored (the 2^32 IBM sweep of `mise run test-ibm-exhaustive`). Python, with the viewer required: 640 passed, 41 skipped (large-file audit without `SEGYRS_DATA`, segyio-only comparisons, no genuine format 4 fixture, installed-wheel checks) | `b0b04a7` | go |
| Interop crate | `mise run test-interop` | needs a rerun | none | needs a rerun |
| Rust floor | `mise run check-msrv` | `segyrs` and `segyrs-draw` check on Rust 1.85.1; `segyrs-py`, with `segyrs-interop` (cloud, zarr), checks on Rust 1.89.0 | `b0b04a7` | go |
| Wheel | `mise run wheel` | `segyrs` 0.1.0, `cp311-abi3`, local tag `linux_x86_64`, 5.8 MB (5,794,779 bytes, `stat -c %s`) | `b0b04a7` | go |
| Wheel on 3.11 | `mise run test-wheel` | the segyrs wheel alone, in a fresh 3.11 venv: 575 passed, 106 skipped (the viewer tests, segyio comparisons, large-file audit, no genuine format 4 fixture) | `b0b04a7` | go |
| Wheel on 3.15 | `mise run test-wheel 3.15` | needs a rerun | none | needs a rerun |
| sdist and a build from it | `mise run test-sdist` | one sdist built; segyrs compiled from it in a fresh 3.11 venv, opened `test-data/small.sgy` as (5, 5, 50) by path and as a `file://` URL, and exported it to Zarr | `b0b04a7` | go |
| Package metadata and PyPI README | `mise run check-dist` | twine 6.2.0 `check --strict`: 2 of 2 PASSED (1 wheel, 1 sdist) | `b0b04a7` | go |
| Documented code blocks | `mise exec -- .venv/bin/python -m pytest -q python/tests/test_doc_blocks.py python/tests/test_examples.py` | ran inside `mise run test` above; the separate count needs a rerun | none | needs a rerun |
| Worked examples | `mise run example` | the three scripts assert their results; figures unchanged | `b0b04a7` | go |
| Docs site | `mise run docs-build` | `mkdocs build --strict` clean; theme checks 5 passed | `b0b04a7` | go |
| Large local files | `systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 mise exec -- env SEGYRS_DATA=../segyrs-data .venv/bin/python -m pytest -q python/tests/test_release_audit_data.py` (the `mise run test-data` suite under a memory limit) | needs a rerun | none | needs a rerun |
| Audit blockers | the tests in the next table, by `pytest -k` and `cargo test` | the Python tests ran inside `mise run test` above; the separate run needs a rerun | none | needs a rerun |
| Release workflow on all six platforms | GitHub, Actions, release, Run workflow, `publish: none` | **not run**: GitHub refuses to start Actions jobs for `cratongen` (billing). No GitHub run has ever executed a job | none | **no-go** |

`b0b04a7` is a commit of the history replaced on 2026-11-21 by the single
root commit `bc78304`.

### Local release build, 2026-11-21

Built on Linux x86_64 because Actions are blocked. Docker was not usable
(the user is not in the `docker` group, so the socket refuses the
connection), so the manylinux and musllinux containers of the release
workflow were not used; `mise run wheel-linux` cross-links with zig instead
([Install and release](install-and-release.md#build-and-test-the-wheels-locally)).

| Check | Command | Result | Upload |
|-------|---------|--------|--------|
| sdist | `mise run sdist` | built | yes |
| manylinux_2_28 x86_64 wheel | `mise run wheel-linux` | built; auditwheel 6.8.2 `show`: consistent with `manylinux_2_28_x86_64` | yes |
| that wheel on 3.11 and 3.15 | `mise run test-wheel-file 3.11 <wheel>`, then `3.15` | 575 passed, 106 skipped on each | yes |
| musllinux_1_2 x86_64 wheel | `mise run wheel-linux x86_64-unknown-linux-musl musllinux_1_2` | built, **not tested**: needs a musl Python (an Alpine container) | no |
| manylinux_2_28 aarch64 wheel | `mise run wheel-linux aarch64-unknown-linux-gnu manylinux_2_28` | built, auditwheel consistent with `manylinux_2_28_aarch64`, **not tested**: needs an arm64 Python | no |
| macOS and Windows wheels | none | not built: they cannot be built legitimately on Linux | no |
| Metadata and PyPI README | `mise run check-dist` | twine 6.2.0 `check --strict`: 4 of 4 PASSED (3 wheels, 1 sdist) | |

### Audit blockers and the test that closes each

| Audit blocker | Closed by |
|---------------|-----------|
| Python 3.9 floor, Pre-Alpha classifier | `test_s44_python_floor_and_newest_match_the_mise_pins`, `test_classifiers_match_a_first_public_release` |
| URLs, publisher owner and docs site on the old owner | `test_project_urls_point_at_the_organization`, `test_no_published_text_names_the_old_repository_owner`, `test_mkdocs_site_and_repo_urls` |
| Viewer stack in the published wheel | `test_the_segyrs_wheel_links_cloud_and_zarr_but_no_viewer_stack`, `test_the_segyrs_wheel_alone_reads_urls_writes_zarr_and_says_how_to_build_the_viewer` |
| README not ready for PyPI | `test_readme_links_work_on_pypi`, `mise run check-dist` |
| False Rust floor claim | `test_each_crate_rust_version_covers_its_dependencies`, `test_the_msrv_task_checks_the_workspace_rust_version`, `mise run check-msrv` |
| `test-data/` licence | `test_test_data_carries_its_own_licence_notice`, `test_no_fixture_of_unknown_origin_and_segyrs_fixtures_are_reproducible` |
| Third-party licences missing from the wheels | `test_each_wheel_carries_the_licence_of_every_crate_it_links`, `test_built_packages_hold_no_test_data_and_ship_third_party_licences`, `test_an_installed_wheel_holds_its_third_party_licences` (under `mise run test-wheel`) |
| No CHANGELOG | `test_project_urls_point_at_the_organization` (Changelog URL), `CHANGELOG.md` section 0.1.0 |
| No CONTRIBUTING, issue templates or checklist | `test_a_code_contribution_starts_with_an_issue`, `test_the_release_checklist_names_the_gate_and_real_checks` |
| Two skipped data-correctness tests | `test_crossline_major_selection_writes_in_file_order`, `test_crossline_major_clip_and_filter_keep_file_order`, `test_unselected_crossline_major_volume_copies_byte_for_byte`, `test_end_text_in_any_case_and_spacing_ends_the_extended_headers`, Rust `end_text_stanza_is_found_in_any_case_and_spacing`, `unknown_text_count_without_end_text_is_bounded` |
| Status pages say pre-alpha | `test_status_pages_agree_that_m1_to_m5_are_implemented` |
| No documented Python block runs | `test_documented_python_blocks_run`, `test_readme_and_docs_home_share_one_quick_start_setup` |
| Docstrings not NumPy style | `test_public_docstring_is_numpy_style`, `test_docstring_examples_run` |
| Wrong path and non-SEG-Y errors | `test_files_that_are_not_segy_say_so`, `test_edit_of_a_read_only_file_is_a_permission_error`, Rust `a_path_the_file_system_refuses_names_the_path_and_what_to_check`, `files_that_are_not_segy_say_what_they_look_like` |
| Exceptions cannot be pickled | `test_every_error_pickles_to_the_same_class`, `test_validation_problems_survive_pickling`, `test_an_error_raised_in_a_worker_process_reaches_the_parent` |
| `create(sample_interval_us=0)` | `test_create_refuses_a_zero_sample_interval`, Rust `create_refuses_a_zero_sample_interval` |
| Performance claims without date, command or harness | `test_published_benchmarks_name_their_date_harness_and_versions` |

## Packaging, done

| Item | Check |
|------|-------|
| Version 0.1.0 in `Cargo.toml` and in the `segyrs==` pin of the unreleased viewer package | `test_segyrs_is_the_only_published_project_and_the_viewer_builds_from_the_repository` |
| CPython 3.11 and newer, one `cp311-abi3` wheel per platform | `test_s44_python_floor_and_newest_match_the_mise_pins` |
| Licence `Apache-2.0` only (owner decision of 2026-11-21) in `pyproject.toml` and every crate, with one `LICENSE` in each package | `test_no_crate_is_published_to_crates_io_yet`, `test_segyrs_is_the_only_published_project_and_the_viewer_builds_from_the_repository` |
| One project, `segyrs`, with cloud reads and Zarr export built in and no extras; the viewer builds from the repository and is not released | `test_segyrs_is_the_only_published_project_and_the_viewer_builds_from_the_repository`, `test_the_segyrs_wheel_links_cloud_and_zarr_but_no_viewer_stack`, `test_the_segyrs_wheel_alone_reads_urls_writes_zarr_and_says_how_to_build_the_viewer` |
| Every built platform tested before upload; one sdist and 6 wheels checked by the publish jobs | `test_s45_release_publishes_only_what_it_tests` |
| Crates stay off crates.io (`publish = false`) | `test_no_crate_is_published_to_crates_io_yet` |
| PyPI upload only from a `v*` tag of `cratongen/segyrs`; wheels build `--locked` | `test_release_publishes_only_from_this_repository_and_pypi_only_from_a_version_tag`, `test_release_wheels_build_from_the_committed_lockfile` |
| The wheel and the sdist carry `THIRD-PARTY-LICENSES.txt` for every crate the extension links on the six release targets (`mise run third-party-licenses` after a `Cargo.lock` change) | `test_cargo_about_covers_every_release_target` and the licence tests above |
| Docs deploy waits for a public repository | `test_docs_deploy_waits_for_a_public_repository` |
| Docs theme black, white and grey in both schemes, text at 4.5:1 or more, fonts served with the site | `test_every_material_colour_variable_is_set_by_the_theme`, `test_theme_admonition_rules_outrank_material_hued_ones`, `test_muted_text_meets_four_and_a_half_to_one_on_every_surface`, `test_the_theme_makes_no_outside_request` |

## Open findings that do not block

- Wheels from the release workflow are built without `--remap-path-prefix`,
  so they carry the CI runner's cargo-registry paths in panic locations (runner
  paths, not the owner's). The local `mise run wheel` maps them out.
- Some linked crates ship no licence file of their own (for the viewer
  package: eframe, egui, the objc2 crates and others; their count for the
  segyrs wheel with the object-store stack was not recounted). `THIRD-PARTY-LICENSES.txt` has the standard text, the
  authors and the source for each, but not the crate's own copyright line. A
  `[clarify]` entry per crate in `about.toml` would give exact notices.
- `FormatChangeError` ("format change is a new file") and `EndianMismatch`
  ("endian mismatch: signature=0x04030201") do not say which property differs
  or what to do. The pages explain them.
- mypy over segyrs' own source reports about 140 errors, mostly unannotated
  private `__slots__` attributes. Users' type checking is unaffected: the
  package ships `py.typed` and `_segyrs.pyi`, checked by `python/tests/test_typing.py`.
- The `rust` blocks in `docs/user-guide.md`, `docs/writing.md`,
  `docs/viewer.md` and `docs/cloud-and-zarr.md` are not compiled by any test.
  Only the Python blocks are.
- A file with bytes after its last whole trace raises `SampleCountMismatch`.
  This is documented with a workaround and tested; there is no open option to
  ignore them.
- On a file whose delay varies per trace, `sel(time=...)` and `slice_horizon`
  use trace 0's time axis. Documented, with the per-trace correction in the
  pre-stack example.
- `geometry.missing_pairs` lists only empty (inline, crossline) positions, so
  it is empty on a varying-fold file whose gaps are offsets.
- A `to_segy(endian="little")` copy keeps the source revision and writes no
  byte-order signature. Documented on the writing example page.
- `Volume.plot` on `isel(inline=[0, 3])` draws the first inline; the user guide
  may read as if a scattered selection shows text instead.
  `sel(method="nearest")` on a tie picks the lower label (documented as
  observed).
- `Editor.set_header(..., scalars=...)` encodes the value but does not write
  `scalco` or `scalel`. Documented; writing them would silently rescale the
  other coordinates on those traces.
- Not measured: the wall time of the chunked, page-releasing full header scan
  against the earlier single-pass scan.
- The segyio comparison skips `stanzas-unknown-count.sgy`: segyio counts 56
  traces where the independent decoder and `test-data/README.md` give 6.
- A wheel built outside the manylinux container gets a local tag (`linux_x86_64`
  for the segyrs wheel of `mise run wheel` on this machine). The release
  workflow builds in the manylinux_2_28 container; `mise run wheel-linux`
  builds the upload tag locally with zig.
- Crates are not crates.io-ready, by decision (see Later).

## Decisions for the owner

- **The release date of 0.1.0.** The minetypes gate is lifted (2026-11-21);
  adoption follows in a later release.
- **The wheel on every platform.** The segyrs wheel now links `aws-lc-sys`
  (C and assembly, through `object_store` and `rustls`). If it fails in the
  release workflow on a platform, the choices are to fix the build, to drop
  that platform (the publish checks would change), or to delay the release.
- **When to publish the viewer.** `segyrs-view` gets its own PyPI project, a
  `view` extra and its own pending publisher in that later release.
- **Git history: decided.** A fresh history of one root commit
  (`bc78304`), pushed on 2026-11-21. The segyio history is not in the
  repository.
- **Upload 0.1.0 by hand or wait for Actions.** A manual upload now gives
  PyPI the sdist and the Linux x86_64 wheel only, and the README on the 0.1.0
  page says so for good (PyPI keeps each version's description). Waiting
  gives all six wheels in one release once billing is fixed.
- **Licence expression of the wheels.** `pyproject.toml` states
  `Apache-2.0` (owner decision of 2026-11-21; it was `MIT OR Apache-2.0`). Each wheel also contains statically linked
  code under other permissive licences (BSD-2-Clause, BSD-3-Clause, ISC,
  Unicode-3.0, and for the unreleased `segyrs-view` also Zlib, OFL-1.1 and
  Ubuntu-font-1.0), with their texts in `THIRD-PARTY-LICENSES.txt`. PEP 639
  asks the expression to cover the whole distribution; keep the short
  expression or list those licences in it.
- **F3 data in `test-data/`.** `f3.sgy`, `f3-lsb.sgy` and `multiformats/`
  are CC BY-SA 3.0 (dGB Earth Sciences, checked on TerraNubis on
  2026-11-20) and carry that attribution in `test-data/LICENSE.md`. Keep
  them under that notice, or replace them by generated fixtures.
- **Benchmark harness in `benchmarks/`.** The six scripts (no data) were
  copied from `../segyrs-data`. Confirm they may be public.
- **Geist fonts with the docs site** (SIL OFL 1.1, about 115 KB, docs only),
  or go back to a Google Fonts import.
- **A manual `publish: pypi` run from a `v*` tag** (for a retry) is allowed
  today. Removing that option leaves the tag push as the only route.
- **API questions left as they are:** an open option to ignore trailing
  bytes; the split between `GeometryError` (a dim the volume lacks) and
  `ValueError` (a dim name segyrs does not know); `set_header(scalars=...)`
  leaving `scalco` alone; `SegyrsValueError` and `BudgetError` as new public
  names.
- **The repository is public** (done, 2026-11-21).

## Owner steps to publish

Steps 2, 3, 4 and 6 and the first half of step 7 are done (2026-11-21).
While Actions are blocked by billing, step 5, the docs run of step 7 and
step 9 cannot run; the manual route is
[Publish from a local build](install-and-release.md#publish-from-a-local-build)
with `mise run publish-local`, after step 8. A project created by that upload
may not take over the pending publisher: then add the same trusted publisher
under the project's Settings, Publishing, before the next tag.

1. **Run every check in the go/no-go table on the release commit**, the ones
   marked "needs a rerun" included:

   ```bash
   mise run lint && mise run lint-ci && mise run build && mise run test
   mise run test-interop && mise run check-msrv && mise run test-sdist
   mise run test-wheel && mise run test-wheel 3.15 && mise run check-dist
   mise run docs-build && mise run example
   ```

2. **PyPI pending publisher.** On pypi.org, Account settings, then
   Publishing, add one pending publisher (GitHub), for `segyrs` only, with:

   | Field | Value |
   |-------|-------|
   | PyPI project name | `segyrs` |
   | Owner | `cratongen` |
   | Repository name | `segyrs` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   For a TestPyPI dry run, do the same on test.pypi.org with environment name
   `testpypi`. Turn on two-factor authentication if it is not on.
3. **GitHub environments.** In Settings, then Environments, create `pypi`
   with yourself as required reviewer and deployments limited to tags `v*`.
   Create `testpypi` the same way if you use it.
4. **Push `main`** (`git push origin main`). Done on 2026-11-21: a fresh
   single-commit history.
5. **Dry run.** Actions, then release, then Run workflow on `main` with
   `publish: none`. All six platforms must build and pass their tests. This is
   the first run of the segyrs wheel with the object-store stack outside Linux
   x86_64. Then run it again
   with `publish: testpypi` and, in a fresh Python 3.11 venv, `pip install
   --index-url https://test.pypi.org/simple/ --extra-index-url
   https://pypi.org/simple/ segyrs` and run the README quick start.
6. **Decide the git history** (decisions above) and apply it. Done.
7. **Make the repository public**, then Settings, Pages, Source: GitHub
   Actions, then Actions, docs, Run workflow on `main`. Check that
   `https://cratongen.github.io/segyrs/` serves the site and
   `https://cratongen.github.io/segyrs/brand/wordmark-black.svg` loads.
8. **Date the changelog.** Change `## 0.1.0 (unreleased)` in `CHANGELOG.md`
   to the release date and commit it.
9. **Tag and push the tag:**

   ```bash
   git tag -a v0.1.0 -m "segyrs 0.1.0"
   git push origin v0.1.0
   ```

   The sdist job fails the run if the tag does not equal the workspace
   version (0.1.0). After the build and test jobs pass, approve the `pypi`
   deployment.

10. **Check after the upload:**
    - `https://pypi.org/p/segyrs` shows 0.1.0 with one sdist and six wheels (manylinux_2_28 x86_64 and aarch64, musllinux_1_2 x86_64,
      macOS 10.15 x86_64, macOS 11.0 arm64, win_amd64), and the pending
      publisher became a normal one.
    - The segyrs project page renders the README with the wordmark and working
      links, `Requires: Python >=3.11`, licence `Apache-2.0`, and the
      Homepage, Documentation, Repository, Issues and Changelog links.
    - In a fresh Python 3.11 venv on another machine, `pip install segyrs`
      installs a wheel (not the sdist), the README quick start runs, and
      `segyrs.view(vol)` raises `ImportError` that names `mise run build`.
    - PyPI never accepts the same version twice: a broken upload needs 0.1.1.

## Later

- **crates.io.** Every crate has `publish = false`. Before publishing, each
  needs a `readme`, `keywords` and `categories`, licence files inside the
  crate, versions next to its path dependencies, and tests that do not read
  `../../test-data`. The order would be `segyrs`, then `segyrs-draw`,
  `segyrs-interop` and `segyrs-view`; the `-py` crates stay unpublished.
- **conda-forge.** There is no recipe yet.
