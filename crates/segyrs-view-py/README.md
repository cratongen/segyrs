# segyrs-view

The experimental native viewer window of [segyrs](https://cratongen.github.io/segyrs/).
It is a separate package so that the `segyrs` wheel carries no window or GPU stack.

It is not part of the segyrs 0.1.0 release on PyPI; it will be published later. Build it
from a clone of the repository, which installs `segyrs` and this package into `.venv`:

```bash
mise run build
```

Then `segyrs.view(volume)` opens a window. The [viewer page](https://cratongen.github.io/segyrs/viewer/)
covers its options. This package has no public Python API of its own, and it is
experimental: its behaviour can change in any release.

Licence: Apache-2.0.
