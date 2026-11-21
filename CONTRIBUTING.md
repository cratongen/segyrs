# Contributing to segyrs

Issues are open to everyone. Bug reports, questions about a SEG-Y file that
does not open, and ideas are welcome as
[GitHub issues](https://github.com/cratongen/segyrs/issues/new/choose).

## Ways to contribute

- Report a bug, or ask why a file does not open, with the
  [bug form](https://github.com/cratongen/segyrs/issues/new?template=bug.yml).
- Propose a change, or offer to write it, with the
  [proposal form](https://github.com/cratongen/segyrs/issues/new?template=proposal.yml).
- Improve the docs, the examples, or a test, once the matching issue is accepted.
- Report a vulnerability in private. The steps are in [SECURITY.md](SECURITY.md).

Follow the [code of conduct](CODE_OF_CONDUCT.md) in issues, pull requests,
and any other discussion about the project.

## Reporting a bug

Use the bug form. Say what you ran, the segyrs version, and the error text.
Describe the file (revision, sample format, byte order) when you know it.
Never attach or commit a SEG-Y file you may not share.

## Proposing a change

A code contribution starts with an issue. Open that issue before you write
code, and wait for a reply. Say what you want to change and why.

A pull request that does not link an accepted issue is closed, with a pointer
back to this page. The reply tells you whether the change fits. That keeps
two people from building the same thing, and it keeps the API small.

## Development setup

1. Fork the repository and branch from `main`.
2. Install [mise](https://mise.jdx.dev/), then run `mise install` and
   `mise run build`.
3. Every command is a mise task. Do not call `cargo`, `python`, `pip`, or
   `uv` on their own.

The docs list every task, including lint, tests, wheels, and the site:
[Contributing](https://cratongen.github.io/segyrs/contributing/).

## Pull requests

1. Make one logical change per commit. Use the commit rules under
   Project standards.
2. Add a test that fails without your change. Update the docs in the same
   change.
3. Run `mise run lint` and `mise run test`.
4. Open the pull request with `Closes #N` for the accepted issue, and fill
   in the pull request template.

A maintainer reads the linked issue, the diff, and the tests. They may ask
you to split the change, add a test, or adjust the docs. They reply on the
pull request.

## Project standards

- **Clean room.** segyrs is written from the SEG-Y rev 0, 1, 2.0 and 2.1
  standards. segyio is a behavioural reference only: read its documentation
  and issues, never its source.
- **No data you may not share.** Tests and examples use generated files or
  small public files whose licence allows redistribution. Never attach or
  commit a SEG-Y file you may not share. A file over 1 MB does not go in git.
- **Measured numbers only.** A performance number in an issue, a commit, or
  the docs comes from a benchmark you ran: before and after, same machine,
  interleaved rounds, medians, load noted. Otherwise write "not measured".
- **numpy is the only runtime dependency.**
- **Conventional Commits.** One logical change per commit, with
  [Conventional Commits](https://www.conventionalcommits.org/) messages
  (`fix(core): ...`, `feat(py): ...`, `docs: ...`). Add no AI trailer and no
  `Co-Authored-By` trailer.

## Licensing

segyrs is licensed under Apache-2.0. See [LICENSE](LICENSE).

By contributing you agree that your contribution is licensed under
Apache-2.0, the licence of this project, without additional terms. That is
section 5 of the licence: unless you explicitly state otherwise, a
contribution you submit for inclusion in segyrs is under Apache-2.0, without
additional terms.

## Security

Report a vulnerability in private, as [SECURITY.md](SECURITY.md) describes.
Do not open a public issue for it.
