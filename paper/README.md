# Paper build

The manuscript is written for the AAAI-27 format.

## Draft build

```bash
make paper
```

The repository includes `aaai27draft.sty`, a local layout fallback used only so
that the draft builds without downloading external files. It is **not** an
official conference style and must not be used for submission.

## Final submission build

Download the official AAAI-27 author kit from the conference website and place
`aaai27.sty` and `aaai27.bst` in this directory. Both `main.tex` and
`supplement.tex` automatically use the official style when it is present.

- `main.tex`: main paper
- `supplement.tex`: full proofs and experimental details
- `references.bib`: bibliography
- `PROOF_NOTES.md`: proof status and next theoretical milestones
- `REVIEWER_CHECKLIST.md`: claim and submission audit
