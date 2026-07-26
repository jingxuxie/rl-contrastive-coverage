# Initial Experimental Results

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
   `1.804e-16`.
3. **The sequential prediction is exact in the shared-branch model.** The
   population width is `|epsilon| H`; the grid maximum numerical discrepancy is
   `4.441e-16`.
4. **The confidence certificate is conservative and valid in these tests.** At
   `n=5,000`, the bandit interval covers the true contrast with frequency
   `0.998` and certifies the positive improvement with
   frequency `1.000`. At `n=10,000`, the
   sequential frequencies are `1.000` and
   `1.000`, respectively.
5. **Joint identification can be much sharper.** In random bandits, the median
   ratio of separate-value width to contrastive width is
   `7.5x` and the 90th percentile is
   `38.1x`.
6. **Identification and efficiency are distinct.** In the bandit example,
   signed IPS has asymptotic variance
   `0.1786`, while the
   semiparametric efficiency bound is
   `0.0697`, a
   `2.56x` gap removed
   by the stratified plug-in estimator.

## Interpretation and limitations

The simulations validate the algebraic identification transition and the
finite-sample concentration implementation; they are not intended as evidence
of broad empirical superiority on deep-RL benchmarks. The current certificate
assumes known logging propensities and fixed policies. The paper explicitly
states that when the comparison baseline is the logger itself, contrastive
coverage reduces to ordinary coverage of the candidate, so the strict gain
arises when two policies share off-support behavior relative to a third logging
policy.
