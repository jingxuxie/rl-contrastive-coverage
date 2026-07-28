# Contrastive Coverage for Offline Policy Improvement

This repository gives a theory-first answer to a basic offline-RL question:

> When can logged data certify that one policy is better than another, even when
> neither policy value is individually identifiable?

The answer is **uniform contrastive coverage (UCC)**. On every state-action
history assigned zero probability by the logger, the candidate and baseline
must assign the same cumulative policy-path probability. UCC is necessary and
sufficient for uniform nonparametric identification of the value difference
`V^pi - V^pi0` over finite-horizon controlled processes with bounded rewards.
It is strictly weaker than covering each policy separately.

## Main results

### Identification and partial identification

- Necessary-and-sufficient uniform identification theorem.
- Instance-specific characterization by signed-measure domination.
- Signed per-decision importance-sampling representation.
- Strict separation from individual policy coverage.
- Exact reduction to ordinary coverage when the baseline is the logger.
- Sharp partial-identification intervals on deterministic history trees.
- Interval width equals discounted unmatched off-support path mass.
- Policy-library equivalence classes and an `O(KN)` explicit-tree audit.

### Efficient sequential comparison

- A direct contrastive Bellman recursion that does not require two separately
  defined value functions.
- **Contrastive augmented PDIS (CA-PDIS)**, conditionally unbiased for every
  independently fitted continuation model whose action average is computed
  under the known logger.
- The oracle continuation is the efficient influence function in the
  nonparametric sequential model with known logging propensities.
- Exact variance identity
  ```text
  Var(PDIS) = V_eff + V_action,
  ```
  plus a general nuisance-error decomposition.
- Cross-fitted asymptotic efficiency under action-centered `L2` consistency,
  without a product-rate remainder.

### Search-safe deployment

- Fixed-pair empirical-Bernstein deployment certificates.
- A simultaneous certificate for a fixed library of `K` candidates, controlling
  family-wise false improvement after policy selection.
- A sample-complexity decomposition showing that CA-PDIS and signed PDIS pay the
  same `log(K/delta)` search cost, while signed PDIS additionally pays removable
  action-composition variance.
- Sharp ambiguity corrections for candidates outside the baseline's UCC class.

### Lightweight validation

Every experiment is CPU-only. Headline results from seed `20260726` are:

- shared unsupported mass `q=0.4`: contrastive width `0`, separate-value width
  `0.8`;
- sharp width identities hold to at most `4.45e-16` in the controlled sequential
  examples and `1.95e-16` across 5,000 random bandits;
- exact sequential variance decomposition holds to `2.14e-14` through horizon
  16;
- PDIS/efficiency variance ratio grows from `2.19x` at horizon 1 to `13.69x` at
  horizon 16;
- two-fold learned CA-PDIS reduces MSE by `8.49x` at `n=2,000` in the chain;
- in a stochastic five-state MDP with a shared logger-null branch, tabular
  CA-PDIS reduces MSE by `7.59x` at `n=5,000`;
- with 1,000 split episodes, CA-PDIS certifies all 500 positive-improvement
  repetitions, versus one for signed PDIS, while both cover in all repetitions;
- under a global null with 100 searched candidates, uncorrected pointwise Wald
  inference falsely deploys with probability `0.927`; simultaneous EB rules make
  no false deployment in 300 trials;
- with 50 candidates and 6,000 episodes, simultaneous CA-PDIS certifies safely
  with probability `0.897`, versus `0.143` for simultaneous signed PDIS.

See `results/RESULTS_SUMMARY.md` for the complete reproducible snapshot.

## Reproduce

```bash
python -m pip install -r requirements.txt
make test
make experiments
make paper
make supplement
make preflight
```

A full local submission bundle is generated with:

```bash
make submission-bundle
```

This creates the main PDF, proof supplement, an anonymized code-and-data ZIP,
and SHA-256 manifest under `submission/`.

## Repository layout

- `src/contrastive_coverage/`: sharp intervals, signed estimators, contrastive
  augmentation, stochastic tabular validation, and history-tree audits.
- `tests/`: 17 deterministic theorem-level and estimator tests.
- `experiments/run_all.py`: regenerates all tables, figures, and headline
  metrics using independent child random streams.
- `results/`: machine-readable outputs and paper figures.
- `paper/main.tex`: seven-page AAAI-27 technical manuscript, references, and the
  official reproducibility checklist.
- `paper/supplement.tex`: complete proofs and experimental details.
- `scripts/preflight_submission.py`: technical-page, reference-page, font,
  citation, style, figure, and checklist audit.
- `scripts/package_submission.py`: deterministic anonymous bundle creation.

## Scope and limitations

The strict identification gain arises when a third logger compares policies that
share off-support behavior. If the baseline is the logger, UCC reduces to
ordinary candidate coverage. Finite-sample guarantees assume known logging
propensities, bounded rewards, independent episodes, and a fixed pair or finite
library defined before the certification split. Adaptive candidate generation,
estimated propensities, continuous actions, and sharp partial identification
with unknown off-support transitions remain outside the present scope.
