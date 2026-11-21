# Install and release

## Install

```bash
pip install segyrs
```

That installs everything in the 0.1.0 release: reading, selection, header
columns, writing, in-place edits, the streaming whole-file operations, the
notebook picture, `segyrs.open` on `s3://`, `gs://`, `az://`, `http(s)://` and
`file://` URLs, and `Volume.to_zarr`, the Zarr v3 export. Numpy 2 is its only
runtime dependency. Object-store reads and Zarr export are experimental: their
API and behaviour can change in any release. segyrs is the only package of the
0.1.0 release, and it has no extras.

The native viewer window (`segyrs.view`) is not part of the 0.1.0 release on
PyPI; it will be published later. It is the `segyrs-view` extension in this
repository (`crates/segyrs-view-py`), and `mise run build` installs it with
segyrs into the checkout's `.venv`. Without it, `segyrs.view` raises
`ImportError` that says so and names that task; `Volume.plot` and the notebook
picture need no viewer.

The segyrs wheel links the object-store and Zarr crates (`segyrs-interop` with
its `cloud` and `zarr` features, and through them `object_store`, `tokio` and
`serde_json`) and no window or GPU crate.
`test_the_segyrs_wheel_links_cloud_and_zarr_but_no_viewer_stack` checks the
cargo graph of the extension, so a viewer dependency added by mistake fails the
tests. On this machine (Linux x86_64, `mise run wheel`, 2026-11-21) the wheel is
5.8 MB (5,794,779 bytes, `stat -c %s`).

Each platform gets one wheel. The wheels are `abi3` (`cp311-abi3`), so the same
file works on every CPython from 3.11 up. The release workflow builds and
tests the wheels of this table; [What 0.1.0 has on PyPI](#what-010-has-on-pypi)
says what pip does on each platform.

| Platform | Wheel tag | Minimum OS |
|----------|-----------|------------|
| Linux x86_64, glibc | `manylinux_2_28_x86_64` | glibc 2.28 (RHEL 8, Debian 10, Ubuntu 18.10) |
| Linux aarch64, glibc | `manylinux_2_28_aarch64` | glibc 2.28 |
| Linux x86_64, musl | `musllinux_1_2_x86_64` | musl 1.2 (Alpine 3.13) |
| macOS Intel | `macosx_10_15_x86_64` | macOS 10.15 |
| macOS Apple silicon | `macosx_11_0_arm64` | macOS 11 |
| Windows x64 | `win_amd64` | Windows 10 |

Supported Pythons are CPython 3.11 through 3.15. The release workflow tests every
wheel before upload. Linux x86_64 (glibc and musl) runs the tests on each of
3.11, 3.12, 3.13, 3.14 and 3.15. Linux aarch64, macOS and Windows run them on
3.11, 3.13 and 3.15. PyPy and the free-threaded builds (`3.13t`, `3.14t`) have
no wheel, because pip does not install a stable-ABI (`abi3`) wheel on them. They
can build from the sdist.

The Rust code is linked into the extension statically. The wheel holds one
native file (`segyrs/_segyrs.abi3.so`, `.pyd` on Windows) and no sidecar
library, so a PyInstaller bundle needs no hook (S-43). The release build fails
if the wheel holds a `.libs` folder or a DLL.

### What 0.1.0 has on PyPI

0.1.0 is published by the release workflow: the sdist and the six wheels of
the table above.

| Platform | What `pip install segyrs` does |
|----------|--------------------------------|
| A platform of the table above | installs its wheel |
| Any other platform | builds segyrs from the sdist |

A build from the sdist needs:

- Rust 1.89 or newer (`rustup.rs`), the `rust-version` of `segyrs-py` (below).
- A C compiler: the Xcode Command Line Tools on macOS, the Visual Studio C++
  Build Tools on Windows (the Rust `msvc` toolchain needs them anyway), gcc or
  clang on Linux. `aws-lc-sys`, linked through `object_store` and `rustls`,
  compiles C and assembly with the `cc` crate.

It needs no CMake, NASM or libclang: `aws-lc-sys` 0.47 ships pregenerated
bindings for each of these targets and builds with `cc` when they exist, and
on Windows x64 the `prebuilt-nasm` feature, which `object_store` enables
through `aws-lc-rs`, supplies the assembled objects. pip fetches maturin, the
build backend, by itself.

### Other platforms

A platform with no wheel in the table has no published wheel (S-45). pip falls
back to the sdist. The sdist builds with Rust 1.89 or newer, the
`rust-version` of `segyrs-py`, which is the highest one its linked dependencies
declare (`segyrs-interop` and its `object_store` stack). `mise run check-msrv`
checks `segyrs` and `segyrs-draw` with Rust 1.85.1, the workspace
`rust-version`, and `segyrs-py` with Rust 1.89.0. The viewer crates need Rust
1.95 and are built and tested with the pinned Rust 1.97.1 only.
`test_each_crate_rust_version_covers_its_dependencies` fails when a dependency
raises its `rust-version` above the crate's.

```bash
pip install segyrs --no-binary segyrs   # needs a Rust toolchain (rustup.rs)
```

From a checkout, `mise run wheel` builds the wheel and `mise run sdist` the
sdist.

### conda-forge

There is no conda-forge package. A recipe is future work.

## Python version policy (S-44)

The Python versions come from one place: `mise.toml` pins the oldest supported
Python, the default development Python and the newest Python:

```toml
python = ["3.13.18", "3.11.18", "3.15.2"]
```

The oldest pin is the `requires-python` floor in `pyproject.toml` and the
`abi3-py311` feature in `crates/segyrs-py/Cargo.toml`. The classifiers run from the
floor to the newest pin. The release test matrix covers the same range, the
publish jobs expect the `cp311-abi3` tag, `mise run test-wheel` defaults to the
floor, and CI runs `test-wheel` on the floor and the newest pin.
`test_s44_python_floor_and_newest_match_the_mise_pins` fails when any of these
disagree.

To add a new Python, pin it with `mise use`, add its classifier, add it to the
`python` lists in `.github/workflows/release.yml`, and run `mise run test-wheel
<version>`. To raise the floor, change all of these places together.

## Build and test the wheels locally

```bash
mise run wheel             # the release abi3 wheel into dist/
mise run test-wheel        # fresh venv on the 3.11 floor: the wheel alone, pytest
mise run test-wheel 3.15   # the same on another pinned Python
mise run sdist             # the sdist into dist/
mise run test-sdist        # build the sdist from source in a fresh 3.11 venv; open a path and a file:// URL, export Zarr
mise run check-dist        # twine check --strict of everything in dist/; uploads nothing
mise run wheel-linux                                    # manylinux_2_28 x86_64 wheel with its upload tag (zig, no container)
mise run wheel-linux x86_64-unknown-linux-musl musllinux_1_2
mise run wheel-linux aarch64-unknown-linux-gnu manylinux_2_28
mise run test-wheel-file 3.11 dist/segyrs-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl   # test one wheel, no rebuild
mise run lint-ci           # actionlint and shellcheck over .github/workflows
```

`test-wheel` installs the wheel, not the source tree, checks that `segyrs` was
imported from `site-packages`, and runs pytest on the wheel alone, as a PyPI
user gets it: the object-store and Zarr tests run, and the viewer tests skip.
`mise run test` runs them with the viewer that `mise run build` installs, and
sets `SEGYRS_REQUIRE_EXTRAS=1` so a missing viewer fails instead of skipping.
The wheel and sdist tasks build with
the floor Python that mise pins (`--interpreter python3.11`), not a system
Python. `mise run wheel` maps the checkout and the
cargo registry out of the panic locations the dependencies compile in, and
`test-wheel` fails if the extension still holds the home directory or the
checkout path. The release workflow builds without that mapping, so its wheels
carry the CI runner's registry paths.

`mise run wheel-linux` builds a wheel with its upload tag on a Linux host
without a container: maturin's `--zig` support links against glibc 2.28 (or
musl 1.2) and `--compatibility` sets the tag, with the arguments the release
workflow passes (`--release --locked`) plus `--interpreter python3.11` and the
same path mapping as `mise run wheel`. maturin's auditwheel check fails the
build if the wheel needs a newer glibc symbol or a library outside the policy.
It differs from the release workflow in one way: the release workflow builds
in the manylinux_2_28 and musllinux_1_2 containers of `maturin-action`, and
`wheel-linux` cross-links with zig 0.15.2 (pinned in `mise.toml`) instead.
`mise run test-wheel-file <python> <wheel>` is `test-wheel` on a given file,
with no rebuild. A musl or aarch64 wheel cannot run on a glibc x86_64 host
without a container, so those two build here but are tested only by the
release workflow (or in a `python:<version>-alpine` or arm64 container, as
`test-linux` does).

### Licences in the packages

segyrs is licensed under Apache-2.0 (`LICENSE`). The wheel links its Rust
dependencies statically, so it also carries their licences:
`THIRD-PARTY-LICENSES.txt` sits beside `LICENSE` in `license-files`, and lands
in the wheel's `.dist-info/licenses/` and in the sdist. `THIRD-PARTY-LICENSES.txt` at the
root is the file of `segyrs`, which includes the object-store and Zarr crates;
`crates/segyrs-view-py/THIRD-PARTY-LICENSES.txt` is ready for the viewer
package when it is published. cargo-about writes them from `Cargo.lock`, `about.toml` and `about.hbs`, over the six release
targets:

```bash
mise run third-party-licenses   # after any Cargo.lock change, then commit the two files
```

`test_each_wheel_carries_the_licence_of_every_crate_it_links` fails when a
crate linked on any release target (from `cargo tree`) is missing from its
package's file, and `test_cargo_about_covers_every_release_target` fails when
`about.toml` and the release matrix name different targets. Some crates ship
no licence file of their own; for those the file holds the standard licence
text with the crate's authors and source as its `Cargo.toml` states them.

The test files in `test-data/` are never in a package
(`test_built_packages_hold_no_test_data_and_ship_third_party_licences` checks
whatever is in `dist/`). They have their own licences:
[test-data/LICENSE.md](https://github.com/cratongen/segyrs/blob/main/test-data/LICENSE.md).

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request. It
runs `mise run lint`, `lint-ci`, `build`, `test`, `test-interop`, `check-msrv`, `test-sdist`, then
`test-wheel` on 3.11 and 3.15.

`.github/workflows/release.yml` builds the segyrs sdist and six wheels (one per
platform) with `PyO3/maturin-action`, tests each wheel, and publishes. It
builds no viewer package. Every action is pinned to
a commit SHA.

| Job | What it does |
|-----|--------------|
| `sdist` | Builds the sdist. On a tag, checks that the tag equals the workspace version. |
| `wheels` | Builds the wheel of one platform and checks that it is the only one and has no sidecar library. |
| `test-linux` | Installs the Linux wheel in the official `python:<version>-slim` or `-alpine` image (aarch64 under QEMU) and runs pytest. |
| `test-other` | The same for the macOS and Windows wheels, in a uv-managed CPython. |
| `publish-pypi` | Checks for exactly one sdist and six wheels, one per platform tag, then uploads with Trusted Publishing. Runs only in `cratongen/segyrs` and only on a `v*` tag: the tag push, or a manual run started from that tag with `publish: pypi`. |
| `publish-testpypi` | The same, to TestPyPI. Manual runs only, from any ref, in `cratongen/segyrs` only. |

The wheel builds pass `--locked`, so a `Cargo.lock` that does not match the
manifests fails the build instead of resolving new versions. Because PyPI
uploads need a `v*` tag, the `sdist` job always checks the tag against the
workspace version before anything is uploaded; the `pypi` environment rule
(below) is a second guard, not the only one.

`test_s45_release_publishes_only_what_it_tests` fails when a built platform has
no test job, or when a publish job does not wait for the tests.
`test_release_publishes_only_from_this_repository_and_pypi_only_from_a_version_tag`
and `test_release_wheels_build_from_the_committed_lockfile` fail when those
guards are removed.

## Cut a release

The [release checklist](release-checklist.md) has the go/no-go verdict, the
result of every check with its date and commit, and the owner's steps for
0.1.0.

1. Set `version` under `[workspace.package]` in `Cargo.toml`. The Python
   packages take their version from there. Set the same version in the
   `segyrs==` dependency of `crates/segyrs-view-py/pyproject.toml`, which is
   not released yet;
   `test_segyrs_is_the_only_published_project_and_the_viewer_builds_from_the_repository`
   fails until they agree.
2. Run `mise run lint && mise run test && mise run test-wheel` and commit.
3. Optional dry run: in GitHub, open Actions, then release, then Run workflow.
   `publish: none` builds and tests only. `publish: testpypi` also uploads to
   TestPyPI. `publish: pypi` uploads only when the run was started from a `v*`
   tag; from a branch it builds and tests, and the upload job is skipped.
4. Tag the commit and push the tag:

   ```bash
   git tag -a v0.1.0 -m "segyrs 0.1.0"
   git push origin v0.1.0
   ```

5. The release workflow builds, tests, and then waits for the `pypi`
   environment. If the environment has required reviewers, approve the
   deployment. Upload starts after that.

PyPI never accepts the same version twice. A failed upload needs a new version.

### Publish from a local build

The release workflow is the route for every release. If Actions cannot run,
the owner can upload by hand from a Linux x86_64 checkout, with a PyPI API
token instead of Trusted Publishing:

```bash
rm -rf dist && mise run sdist && mise run wheel-linux
mise run test-wheel-file 3.11 dist/segyrs-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl
mise run test-wheel-file 3.15 dist/segyrs-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl
mise run test-sdist
export UV_PUBLISH_TOKEN=pypi-...   # a PyPI API token; the first upload needs an account-wide one
mise run publish-local
```

`publish-local` runs `check-dist` first, refuses to run when
`UV_PUBLISH_TOKEN` (the variable `uv publish` reads) is unset, and uploads
exactly the sdist and the `manylinux_2_28_x86_64` wheel in `dist/`, the files
tested on that machine. It leaves out any musl or aarch64 wheel in `dist/`,
because those were not run there. `--check-url` makes a rerun skip files that
PyPI already has. After the first upload, replace the account-wide token with
one scoped to the `segyrs` project.

## One-time setup (repository owner)

These steps are done once, by hand, before the first release.

### PyPI

1. Check that the name `segyrs` is free on pypi.org.
2. Sign in to pypi.org and open Account settings, then Publishing.
3. Add one pending publisher, for the project `segyrs` only, with these
   values:
   - PyPI project name: `segyrs`
   - Owner: `cratongen` (the organization)
   - Repository name: `segyrs`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
4. Turn on two-factor authentication if it is not on yet.

The first upload creates the project and turns its pending publisher into a
normal one. The viewer gets its own pending publisher (`segyrs-view`) in the
release that publishes it, not in 0.1.0. There is no API token and no repository secret.

### TestPyPI (optional)

Do the same on test.pypi.org (a separate account) with environment name
`testpypi`.

### GitHub

1. Open Settings, then Environments, and create `pypi`. Add yourself as a
   required reviewer, and limit deployments to tags that match `v*`.
2. Create `testpypi` the same way if you use TestPyPI.
3. Under Settings, then Actions, then General, allow GitHub Actions and the
   actions this repository uses. All of them are pinned to commit SHAs.

While the repository is private, every job uses billed Actions minutes, and
macOS minutes cost more than Linux ones. A public repository uses none.

## Docs site

The site is MkDocs with the Material theme, the same stack as boitatá. The
pages are `docs/*.md`, the navigation is `mkdocs.yml`, and the monotone theme is
`docs/brand/tokens.css` plus `docs/stylesheets/segyrs.css`.

```bash
mise run docs          # serve on http://127.0.0.1:8000 with live reload
mise run docs-build    # mkdocs build --strict into site/
```

Both tasks run mkdocs-material 9.7.7 in a throwaway uv environment, so the
docs need neither the Rust build nor the extension. `--strict` fails the build
on a broken link, a missing anchor, or a page left out of the navigation.

`.github/workflows/docs.yml` runs `mise run docs-build` on every push to `main`
that touches the docs, and on a manual run. It uploads `site/` and deploys it
to GitHub Pages at `https://cratongen.github.io/segyrs/`.

The `cratongen` organization is on the free plan, which serves Pages from
public repositories only. While the repository is private the workflow builds
the site strictly and skips the deploy job, so the run stays green. Once the
owner makes the repository public:

1. Open Settings, then Pages, and set Source to GitHub Actions.
2. Open Actions, then docs, then Run workflow on `main`. The deploy job runs
   and the site is live at `https://cratongen.github.io/segyrs/`.
