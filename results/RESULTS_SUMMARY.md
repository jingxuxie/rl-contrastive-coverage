# Submission-Ready Experimental Results

All experiments are lightweight and use exact tabular calculations or i.i.d.
Monte Carlo sampling. The fixed random seed is `20260726`.

## Main findings

1. **Shared unsupported behavior cancels exactly.** In the three-action bandit
   with common unsupported mass `q=0.4`, the sharp contrastive identified set
   has width `0`, while separately bounding the two
   policy values gives width `0.8`.
2. **Only unmatched mass controls partial identification.** Across the full
   bandit grid, the contrastive width equals `|epsilon|` up to numerical
   precision. The maximum error in 5,000 random bandits is
   `1.943e-16`.
3. **The sequential sharpness prediction is exact.** The shared-branch
   population width is `|epsilon| H`; the maximum numerical discrepancy is
   `4.441e-16`.
4. **Contrastive augmentation removes exactly identifiable action noise.** The
   exact identity `Var(PDIS) = V_eff + V_action` holds through horizon 16 with
   maximum error `7.105e-15`.
   The PDIS/efficiency-bound ratio grows from
   `2.19x` at horizon 1 to
   `13.69x` at horizon 16.
5. **A learned nuisance model nearly attains the sequential efficiency bound.**
   With two-fold cross-fitting at `n=2,000`, learned CA-PDIS reduces MSE by
   `8.49x` relative to
   signed PDIS and has empirical bias `1.22e-04`.
6. **Variance reduction translates into certification power.** With 1,000 total
   episodes split equally between nuisance fitting and certification, signed
   PDIS certifies the positive contrast in
   `0.002` of repetitions, whereas CA-PDIS
   certifies it in `1.000`. At the
   largest sample size, empirical coverages are
   `1.000` and
   `1.000`, respectively.
7. **The gain persists in a stochastic MDP beyond the independent chain.**
   In a five-state, six-step process with random transitions and a shared
   logger-null initial branch, two-fold tabular CA-PDIS reduces MSE by
   `7.59x` at `n=5,000`, with bias
   `-2.12e-03`. Neither individual policy
   value is identified because the shared branch is never logged.
8. **Joint identification can be much sharper.** In random bandits, the median
   ratio of separate-value width to contrastive width is
   `7.3x` and the 90th percentile is
   `36.4x`.
9. **Structural screening and simultaneous inference make library search safe.**
   Under a global null with 100 fixed candidates sharing the same unsupported
   component, uncorrected pointwise Wald intervals falsely deploy with
   probability `0.927`,
   while Bonferroni Wald is `0.027`
   and both empirical-Bernstein procedures make no false deployment in the
   300 repetitions. With 50 candidates and 6,000 episodes, simultaneous
   CA-PDIS certifies a genuinely positive selected policy with probability
   `0.897`
   versus `0.143`
   for simultaneous signed PDIS.

## Interpretation and limitations

The experiments validate the algebraic identification transition, the exact
variance decomposition, and finite-sample concentration; they are not intended
as evidence of broad empirical superiority on deep-RL benchmarks. The
certificate assumes known logging propensities, bounded rewards, independent
episodes, and a fixed policy pair or finite library defined before the
certification split. The chain experiment uses a deliberately simple stationary reward model, and
the stochastic validation uses a smoothed tabular model; neural function
approximation is outside the paper's scope.
When the comparison baseline is the logger itself, contrastive coverage reduces
to ordinary candidate coverage, so the strict identification gain arises when
two policies share off-support behavior relative to a third logging policy.
