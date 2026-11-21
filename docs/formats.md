# Sample formats

The binary header stores the sample format in bytes 3225-3226. segyrs decodes that code. Pass `sample_format=` to `segyrs.open` when the header code is wrong. The override is the name in the table, and `attrs["sample_format"]` records the name that was used, so you can see that an override happened.

Integers stay integers. IBM float (code 1) decodes to float32, and the attr stays `"ibm_float32"`. Overflow becomes infinity, which is [S-01] IBM360Converter converts to IEEE NaN for some values. There is no default cast to float64 ([S-04] Support for more numerical formats, [X-08] Investigate Loading data as float32 not float64).

Code 4 is fixed-point with gain. The SEG text disagrees on the sign of the exponent between rev 0 and rev 2, so segyrs raises `Format4NeedsGain` unless you pass `gain="rev0"` (`value * 2^gain`) or `gain="rev2"` (`value * 2^-gain`). That is [S-02] Fixed point with gain data format. The standard marks code 4 obsolete.

Codes 7 and 15 are three bytes on disk. They decode to int32 and uint32, so a value that does not fit in int16 survives ([S-04]).

An unknown code raises `UnknownFormat`. Codes 13 and 14 are unassigned in the standard, so they take that path.

Byte order is separate from the format code. See the user guide.

## Codes

Rev 0 is the 1975 SEG-Y exchange format. Rev 1 is May 2002. In that binary-header table, codes 6 and 7 are printed as "Not currently used", and codes 5 and 8 are the formats rev 1 adds beside the original four. Rev 2.0 (January 2017) assigns codes 6, 7, 9, 10, 11, 12, 15, and 16. Rev 2.1 (October 2023) adds no sample format code.

| Code | Name | Bytes | On disk | Added in | Numpy dtype |
| --- | --- | --- | --- | --- | --- |
| 1 | `ibm_float32` | 4 | IBM floating point | rev 0 (1975) | float32, attr `ibm_float32` |
| 2 | `int32` | 4 | two's complement integer | rev 0 (1975) | int32 |
| 3 | `int16` | 2 | two's complement integer | rev 0 (1975) | int16 |
| 4 | `fixed_point_gain` | 4 | fixed point with gain | rev 0 (1975) | decoded only when `gain=` is set |
| 5 | `ieee_float32` | 4 | IEEE binary32 | rev 1 (2002) | float32 |
| 6 | `ieee_float64` | 8 | IEEE binary64 | rev 2.0 (2017) | float64 |
| 7 | `int24` | 3 | two's complement integer | rev 2.0 (2017) | int32 |
| 8 | `int8` | 1 | two's complement integer | rev 1 (2002) | int8 |
| 9 | `int64` | 8 | two's complement integer | rev 2.0 (2017) | int64 |
| 10 | `uint32` | 4 | unsigned integer | rev 2.0 (2017) | uint32 |
| 11 | `uint16` | 2 | unsigned integer | rev 2.0 (2017) | uint16 |
| 12 | `uint64` | 8 | unsigned integer | rev 2.0 (2017) | uint64 |
| 13 |  |  | unassigned |  | `UnknownFormat` |
| 14 |  |  | unassigned |  | `UnknownFormat` |
| 15 | `uint24` | 3 | unsigned integer | rev 2.0 (2017) | uint32 |
| 16 | `uint8` | 1 | unsigned integer | rev 2.0 (2017) | uint8 |

`sample_format=` also accepts the code as a decimal string (`"1"`, `"5"`, and the rest) and these aliases: `ibm`, `ibm32`, `ieee32`, `float32`, `ieee64`, `float64`, `gain`. The name stored on the volume is the name column.

On write, the file keeps the format recorded at open. `sample_format=` is the conversion. An opened IBM file stays IBM until you pass that argument ([S-03] Tell segyio that a file has a different format than what the header says, [S-35] Change segy format).
