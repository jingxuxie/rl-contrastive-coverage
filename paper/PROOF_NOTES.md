# Proof Status and Research Notes

## Completed theorem package

1. **Uniform identification iff uniform contrastive coverage.**
   For every formal state-action prefix `z_t`, require
   `L_beta(z_t)=0 => L_pi(z_t)=L_pi0(z_t)`. Sufficiency follows by
   factoring each prefix law into a common environment factor and a policy-path
   likelihood. Necessity follows from two observationally equivalent
   deterministic history-tree environments that differ only in reward on one
   logger-null prefix.

2. **Strict separation from individual coverage.**
   Both target values can be unidentified while their difference is point
   identified. The three-action construction is the minimal example.

3. **Logger-baseline boundary.**
   If the comparison baseline is the logger, contrastive coverage is exactly
   ordinary path coverage of the candidate. The paper states this prominently
   to prevent overclaiming.

4. **Sharp tree partial identification.**
   On deterministic history trees with independently bounded off-support
   reward means, the interval endpoints are attained coordinatewise. Width is
   the discounted sum of absolute unmatched path mass times reward range.

5. **Finite-sample certificate.**
   Signed per-decision importance sampling plus a two-sided empirical-Bernstein
   radius controls false deployment for a fixed candidate/baseline pair.

6. **Bandit sharpness and maximal abstention.**
   The contrastive width is the unmatched unsupported `L1` mass, while separate
   bounds charge total unsupported mass; the improvement factor is unbounded.
   The sharp lower endpoint exactly characterizes when any uniformly sound
   population-level improvement certificate can answer rather than abstain.

7. **Coverage equivalence classes.**
   Equality of off-support path signatures partitions a policy library into
   maximal classes whose pairwise contrasts are uniformly identified. An
   explicit-tree audit costs `O(KN)` for `K` policies and `N` prefix nodes.

8. **One-step efficiency.**
   The efficient influence function and semiparametric variance bound are
   derived for finite-action bandits. A stratified plug-in estimator attains the
   bound, while signed IPS carries an explicit nonnegative composition-noise
   term.

## Important modeling decisions

- Use cumulative policy-path likelihoods, not local action probabilities or
  marginal state-action occupancies. Unsupported excursions can re-enter an
  observed state, and local agreement does not imply cancellation.
- The central theorem is **uniform over a rich environment class**. More
  structure can identify additional contrasts, so UCC is not claimed necessary
  under every restricted model.
- The current certificate assumes known logging propensities, independent
  episodes, bounded rewards, and policies fixed before the certification split.

## Strong next theoretical extensions

- Derive a marginalized signed-density-ratio estimator and a Rao-Blackwell
  variance comparison.
- Extend the one-step efficiency calculation to sequential marginalized or
  doubly robust signed estimators, with cross-fitting under function approximation.
- Extend sharp partial identification beyond deterministic trees to support-
  censored MDPs, carefully accounting for shared downstream states.
- Develop simultaneous certification for a data-dependent library of policy
  pairs using sample splitting, PAC-Bayes, or max-t bootstrap methods.
