# Contrastive Coverage for Offline Policy Improvement

This repository develops a theory-first answer to a basic offline-RL question:

> When can logged data certify that one policy is better than another, even if
> neither policy value is individually identifiable?

The central result is **uniform contrastive coverage**. On every trajectory
prefix that the logger assigns zero probability, the candidate and baseline
must assign the same cumulative policy-path probability. This condition is
necessary and sufficient for uniform nonparametric identification of the value
difference over a rich class of finite-horizon controlled processes.

## Current results

- Necessary-and-sufficient uniform identification theorem.
- Signed per-decision importance-sampling representation.
- Sharp partial-identification bounds on deterministic history trees.
- High-confidence deployment certificate for a fixed policy pair.
- Exact common-component cancellation and a mismatch sensitivity law.
- Policy-library equivalence classes and an `O(KN)` explicit-tree audit.
- One-step semiparametric efficiency bound and an attaining stratified estimator.
- Laptop-scale experiments validating all width and variance identities.

Headline numerical checks:

- Shared unsupported mass `q=0.4`: contrastive width `0`, separate-value width
  `0.8`.
- Maximum tree width-identity error: `2.22e-16`.
- Signed-IPS/efficiency-bound variance ratio in the bandit example: `2.56x`.
- Across 5,000 random bandits, median separate/contrastive width ratio: `7.5x`;
  90th percentile: `38.1x`.

## Reproduce

```bash
python -m pip install -r requirements.txt
make test
make experiments
make paper
```

All experiments are CPU-only and complete with exact tabular calculations or
small Monte Carlo loops. `make paper` regenerates the figures before compiling
the manuscript.

## Repository layout

- `src/contrastive_coverage/`: sharp intervals, signed estimators, and exact
  history-tree calculations, including policy-library signature grouping.
- `tests/`: deterministic theorem-level checks.
- `experiments/run_all.py`: regenerates every table and figure.
- `results/`: CSV data, figures, and a concise results summary.
- `paper/main.tex`: AAAI-style manuscript.
- `paper/supplement.tex`: full proofs and experimental details.
- `paper/PROOF_NOTES.md`: proof audit and next milestones.

## Scope and limitations

The strict gain occurs when two policies are compared against a third logger
and share off-support behavior. If the baseline equals the logger, contrastive
coverage reduces to ordinary candidate coverage. The current certificate also
assumes known logging propensities, bounded rewards, independent episodes, and
a policy pair fixed before certification.
