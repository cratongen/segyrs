# Contributing to segyrs

Issues are open to everyone. Bug reports, questions about a SEG-Y file that
does not open, and ideas are welcome as
[GitHub issues](https://github.com/cratongen/segyrs/issues/new/choose).

## A code contribution starts with an issue

Open an issue before you write code, and wait for an answer on it. Say what
you want to change and why. A pull request that does not link an accepted
issue is closed with a pointer back here. This keeps work from being done
twice and keeps the API small.

When the issue is accepted:

1. Fork the repository and branch from `main`.
2. Install [mise](https://mise.jdx.dev/), then run `mise install` and
   `mise run build`. Every command is a mise task; do not call `cargo`,
   `python`, `pip` or `uv` on their own.
3. Make one logical change per commit, with
   [Conventional Commits](https://www.conventionalcommits.org/) messages
   (`fix(core): ...`, `feat(py): ...`, `docs: ...`). Add no AI or
   `Co-Authored-By` trailer.
4. Add a test that fails without your change. Update the docs in the same
   change.
5. Run `mise run lint` and `mise run test`, then open the pull request with
   `Closes #N` for the issue.

## Rules that are not negotiable

- **Clean room.** segyrs is written from the SEG-Y rev 0, 1, 2.0 and 2.1
  standards. segyio is a behavioural reference only: read its documentation
  and issues, never its source.
- **No client data.** Tests and examples use generated files or small
  public files whose licence allows redistribution. Never attach or commit
  a SEG-Y file you may not share. A file over 1 MB does not go in git.
- **Measured numbers only.** A performance number in an issue, a commit or
  the docs comes from a benchmark you ran: before and after, same machine,
  interleaved rounds, medians, load noted. Otherwise write "not measured".
- **No new runtime dependency.** numpy is the only one.

By contributing you agree that your contribution is licensed under
MIT OR Apache-2.0, the licence of this project, without additional terms.

The full contributor guide, with every mise task, is on the docs site:
[Contributing](https://cratongen.github.io/segyrs/contributing/).
