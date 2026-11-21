# Brand

The segyrs mark is one SEG-Y trace drawn the way a seismic display draws it: a solid block for the 240-byte trace header, then the samples as a variable-area wiggle with the positive lobes filled. Everything else follows from that drawing. The identity uses black, white and greys, two typefaces, and heavy horizontal rules that echo the header block.

segyrs shares its tooling and its grid with [boitatá](https://cratongen.github.io/boitata/): the same docs engine, square blocks, numbered sections and underlined links. The mark and the typefaces are its own.

## Mark and wordmark

![segyrs wordmark](brand/wordmark-black.svg#only-light){ width="372" }
![segyrs wordmark](brand/wordmark-white.svg#only-dark){ width="372" }

| File | Use |
| --- | --- |
| [`brand/wordmark-black.svg`](brand/wordmark-black.svg) | Mark plus name, black ink on a light background. The README and the docs home use it. |
| [`brand/wordmark-white.svg`](brand/wordmark-white.svg) | The same in white ink, for a dark background. |
| [`brand/mark-black.svg`](brand/mark-black.svg) | The trace alone, black ink. Use it where the name is already printed beside it, such as the docs header. |
| [`brand/mark-white.svg`](brand/mark-white.svg) | The trace alone, white ink. |
| [`brand/favicon.svg`](brand/favicon.svg) | White trace on a black square, with thicker strokes so it reads at 16 px. |

Every file is hand-written SVG with no gradient, no filter and no embedded font. The letters of the wordmark are monoline paths on a 20-unit x-height, so the name renders the same on GitHub, in a browser and in a slide deck.

The name is always lowercase: `segyrs`. Set it in code font when you mean the package (`import segyrs`) and in the text face when you mean the project.

## Clear space and size

Keep clear space around the mark equal to the height of the header block on every side. In the 48-unit drawing the block is 6 units tall, so a mark drawn 48 px tall needs 6 px of empty space around it. The wordmark takes the same margin, measured from its outermost ink.

Draw the mark no smaller than 16 px tall, and only as the favicon at that size. Below 24 px, use `favicon.svg`: its strokes are thicker. Draw the wordmark no smaller than 96 px wide.

## Palette

The palette is strict: black, white, and the greys between them. No hue appears in the interface, in a chart frame, or in the mark. Seismic data keeps its own colormaps (`grey`, `seismic`, `amber`) inside the picture, never around it.

| Token | Hex | Role |
| --- | --- | --- |
| `--sg-black` | `#000000` | Ink in light mode, page in dark mode |
| `--sg-grey-900` | `#171717` | Code and note blocks in dark mode |
| `--sg-grey-800` | `#262626` | Hairlines in dark mode |
| `--sg-grey-600` | `#525252` | Muted text and code comments in light mode |
| `--sg-grey-500` | `#737373` | Link underlines only. It is 3.8:1 to 4.7:1 against the four surfaces, enough for a line but not for text |
| `--sg-grey-400` | `#a3a3a3` | Muted text and code comments in dark mode |
| `--sg-grey-200` | `#e5e5e5` | Hairlines in light mode |
| `--sg-grey-100` | `#f5f5f5` | Code and note blocks in light mode, ink in dark mode |
| `--sg-white` | `#ffffff` | Page in light mode |

The tokens live in [`docs/brand/tokens.css`](https://github.com/cratongen/segyrs/blob/main/docs/brand/tokens.css). The docs theme in `docs/stylesheets/segyrs.css` maps them onto both colour schemes and sets every colour variable of Material for MkDocs, so no Material hue (blue code highlights, yellow marks, red and green edits, tinted dark greys, coloured note boxes) reaches a page. Text and muted text meet 4.5:1 contrast on the page and on code and note blocks in both modes: the lowest pair is 7.1:1.

## Type

| Role | Face | Fallback |
| --- | --- | --- |
| Text and headings | [Geist](https://fonts.google.com/specimen/Geist), weights 400 to 800 | `system-ui`, Helvetica, Arial |
| Code, figures, section numbers | [Geist Mono](https://fonts.google.com/specimen/Geist+Mono), weights 400 to 700 | `ui-monospace`, Menlo, Consolas |

Both are variable fonts served with the site from `docs/brand/fonts/` (Geist 1.3.1 by Vercel, SIL Open Font License 1.1, text in `fonts/OFL.txt`), so the theme makes no request to another server. Headings are heavy (800), tight (negative tracking) and upright; nothing in the identity is set in italic. Tables of numbers use tabular figures so columns line up.

## Rules and blocks

The header block of the mark becomes the rule system of the page: a 6 px rule under the page title, a 3 px rule over each section, and 1 px hairlines between table rows. Blocks are square, filled and flat, with no shadow and no radius. Links are underlined instead of coloured.

## Copying the theme

The sibling libraries copy this theme as it is. It is four pieces with no outside request:

| Piece | What it holds |
| --- | --- |
| `docs/brand/tokens.css` and `docs/brand/fonts/` | The palette, the type stacks, the rules, the spacing, and the two font files with their licence |
| `docs/stylesheets/segyrs.css` | The mapping onto Material for MkDocs for both schemes |
| The `theme`, `extra_css` and `markdown_extensions` blocks of `mkdocs.yml` | `font: false`, the two `custom` palettes with their toggles |
| `python/tests/test_docs_theme.py` | The check: grey tokens only, no literal colour in the theme, every Material colour variable set, admonition rules that outrank Material's coloured ones, and 4.5:1 for text. `mise run docs-build` runs it on the built site |

Replace the mark files and the `--sg-` prefix if you like; keep the check.

## Do

- Put the black mark on white or light grey, and the white mark on black or dark grey.
- Scale the files as they are. Keep the trace vertical, with the header block on top.
- Use the wordmark once per page, at the top.
- Print numbers in Geist Mono when they line up in a column.

## Do not

- Recolour the mark, add a gradient, or put it on a photograph.
- Rotate the trace, mirror it, or fill the negative lobes. The positive-fill convention is the point of the drawing.
- Redraw the wordmark in another typeface, or set "segyrs" with a capital letter.
- Add a hue to the interface to signal state. Use weight, a rule, or a word.
- Stretch the mark, crop the header block, or add an outline or a drop shadow.
