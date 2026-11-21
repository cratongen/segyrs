# Licence of the test data

The files in `test-data/` are not under the segyrs licence (MIT OR
Apache-2.0), except the folder `test-data/segyrs/`. None of them is in the
`segyrs`, `segyrs-view` or `segyrs-interop` packages on PyPI; they are in the
source repository only, for the tests.

## segyio test files: LGPL-3.0

Every `.sgy` and `.su` file in `test-data/` outside `test-data/segyrs/`,
and `README.md`, comes from the test data of segyio
(<https://github.com/equinor/segyio>), copyright Equinor ASA and the segyio
contributors. segyio is licensed under the GNU Lesser General Public License,
version 3 (its package metadata says `LGPL-3.0`; its classifier says version 3
or later). The files are redistributed unchanged under that licence, except
`README.md`, which segyrs edited (see its first section). The licence texts are
in `LICENSES/LGPL-3.0.txt` and `LICENSES/GPL-3.0.txt`.

## F3 Netherlands field data: CC BY-SA 3.0

These 30 files hold a cropped and converted part of the F3 block in the Dutch
North Sea:

- `f3.sgy` (its textual header names the source and the crop: inlines 111 to
  133, crosslines 875 to 892, samples 0 to 300)
- `f3-lsb.sgy` (the same, little-endian)
- the 28 files in `multiformats/` (the same traces, converted to each sample
  format and byte order)

The data set "Netherlands Offshore F3 Block - Complete" is published by dGB
Earth Sciences on TerraNubis
(<https://terranubis.com/datainfo/Netherlands-Offshore-F3-Block-Complete>)
under the Creative Commons Attribution-ShareAlike 3.0 licence
(<https://creativecommons.org/licenses/by-sa/3.0/>), checked on 2026-11-20.
The acknowledgement TerraNubis asks for:

> We thank dGB Earth Sciences for making the data available as an OpendTect
> project via their TerraNubis portal terranubis.com

The segyio authors cropped and converted the data, so these files are an
adaptation. Anyone who shares them, or an adaptation of them, must keep this
attribution and share them under CC BY-SA 3.0 or a compatible licence. The
segyio changes to them are under LGPL-3.0, as above.

## segyrs fixtures: Apache-2.0

The files in `test-data/segyrs/` are made by segyrs with
`test-data/segyrs/make_fixtures.py` and are under the segyrs licence, like
the rest of the repository.

`segyrs/delay-scalar.sgy` replaces segyio's `delay-scalar.sgy`, which segyio
describes as "likely real file, cropped" with no source or licence. It was
removed from the tree on 2026-11-20. The replacement keeps the header values
the tests need (delay 10000, time scalar -10, an ASCII textual header) and
holds synthetic samples.
