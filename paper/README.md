# Paper build

The manuscript is written for the official AAAI-27 LaTeX format.

## Build

```bash
make paper
make supplement
```

The repository already contains the unmodified official author kit under
`../AAAI_AuthorKit27/`. The Makefile exposes that directory through `TEXINPUTS`
and `BSTINPUTS`, so `main.tex` and `supplement.tex` automatically load
`aaai2027.sty` and `aaai2027.bst`. Do not edit the official style or
bibliography-style files.

A local `aaai27draft.sty` fallback remains available only for environments in
which the author-kit directory is absent. Output produced with that fallback is
not a submission artifact.

## Submission checks

Before submission, compile from a clean checkout, verify the official page
limit, include the conference reproducibility checklist when required, inspect
all figure and table placement, and confirm that the PDF contains no Type 3
fonts or unresolved citations.

- `main.tex`: main paper
- `supplement.tex`: full proofs and experimental details
- `references.bib`: bibliography
- `PROOF_NOTES.md`: proof status and next theoretical milestones
- `REVIEWER_CHECKLIST.md`: claim and submission audit
