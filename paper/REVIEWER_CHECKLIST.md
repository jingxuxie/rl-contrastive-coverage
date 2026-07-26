# Internal Reviewer Checklist

## Claims that are currently supported

- Necessary-and-sufficient **uniform** identification theorem.
- Strict separation from individual policy coverage.
- Exact collapse to ordinary coverage when the baseline equals the logger.
- Sharp partial-identification interval for deterministic history trees.
- Valid fixed-pair empirical-Bernstein certificate under known propensities.
- Sharp bandit width comparison and maximal population-level abstention result.
- Policy-library equivalence classes with an `O(KN)` explicit-tree audit.
- One-step semiparametric efficiency bound and attaining stratified estimator.
- Exact numerical verification of the population width and variance identities.

## Claims deliberately not made

- No claim of a general minimax lower bound for arbitrary MDPs.
- No claim that contrastive coverage helps when comparing a policy directly
  against the logging policy.
- No claim of broad empirical superiority on D4RL or deep-RL benchmarks.
- No claim that the condition is necessary under additional smoothness, known
  dynamics, parametric rewards, or other structural assumptions.
- No claim of post-selection validity when the policy pair is chosen on the
  same certification data.

## Before submission

- Replace the local draft style with the official AAAI-27 author kit.
- Confirm the official page limit and supplementary-material policy.
- Re-run every experiment from a clean environment.
- Add an independent proof audit of the necessity construction and the exact
  empirical-Bernstein constants.
- Independently audit the one-step efficiency calculation and decide whether a
  sequential doubly robust extension is needed for the final submission.
- Check every bibliography entry against the publisher record.
