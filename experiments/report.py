"""Write machine-readable headline metrics and the results summary."""

import json
import numpy as np
import pandas as pd

from common import DATA_DIR, SEED, SUMMARY_PATH

def write_summary(
    bandit_population: pd.DataFrame,
    bandit_certificate: pd.DataFrame,
    sequential_population: pd.DataFrame,
    sequential_certificate: pd.DataFrame,
    tree_identity: pd.DataFrame,
    random_bandits: pd.DataFrame,
    bandit_efficiency: pd.DataFrame,
) -> None:
    bp0 = bandit_population[
        (np.isclose(bandit_population["q"], 0.4))
        & (np.isclose(bandit_population["epsilon"], 0.0))
    ].iloc[0]
    bc_last = bandit_certificate.iloc[-1]
    sc_last = sequential_certificate.iloc[-1]
    metrics = {
        "bandit_exact_contrastive_width": float(bp0["contrastive_width"]),
        "bandit_separate_width_q_0_4": float(bp0["separate_width"]),
        "bandit_true_difference": float(bc_last["true_difference"]),
        "bandit_coverage_n_5000": float(bc_last["coverage"]),
        "bandit_certification_n_5000": float(bc_last["certification_rate"]),
        "sequential_true_difference": float(sc_last["true_difference"]),
        "sequential_coverage_n_10000": float(sc_last["coverage"]),
        "sequential_certification_n_10000": float(sc_last["certification_rate"]),
        "tree_max_identity_error": float(tree_identity["absolute_error"].max()),
        "random_bandit_max_identity_error": float(random_bandits["identity_error"].max()),
        "random_bandit_median_width_ratio": float(random_bandits["width_ratio"].median()),
        "random_bandit_p90_width_ratio": float(random_bandits["width_ratio"].quantile(0.9)),
        "bandit_ips_efficiency_variance_ratio": float(bandit_efficiency.iloc[0]["asymptotic_variance_ratio"]),
        "bandit_empirical_mse_ratio_n_2000": float(bandit_efficiency.iloc[-1]["ips_mse"] / bandit_efficiency.iloc[-1]["stratified_mse"]),
    }
    (DATA_DIR / "headline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    text = f"""# Initial Experimental Results

All experiments are lightweight and use exact tabular calculations or i.i.d.
Monte Carlo sampling. The fixed random seed is `{SEED}`.

## Main findings

1. **Shared unsupported behavior cancels exactly.** In the three-action bandit
   with common unsupported mass `q=0.4`, the sharp contrastive identified set
   has width `{bp0['contrastive_width']:.3g}`, while separately bounding the two
   policy values gives width `{bp0['separate_width']:.3g}`.
2. **Only unmatched mass controls partial identification.** Across the full
   bandit grid, the contrastive width equals `|epsilon|` up to numerical
   precision. The maximum error in 5,000 random bandits is
   `{random_bandits['identity_error'].max():.3e}`.
3. **The sequential prediction is exact in the shared-branch model.** The
   population width is `|epsilon| H`; the grid maximum numerical discrepancy is
   `{(sequential_population['contrastive_width'] - sequential_population['predicted_width']).abs().max():.3e}`.
4. **The confidence certificate is conservative and valid in these tests.** At
   `n=5,000`, the bandit interval covers the true contrast with frequency
   `{bc_last['coverage']:.3f}` and certifies the positive improvement with
   frequency `{bc_last['certification_rate']:.3f}`. At `n=10,000`, the
   sequential frequencies are `{sc_last['coverage']:.3f}` and
   `{sc_last['certification_rate']:.3f}`, respectively.
5. **Joint identification can be much sharper.** In random bandits, the median
   ratio of separate-value width to contrastive width is
   `{random_bandits['width_ratio'].median():.1f}x` and the 90th percentile is
   `{random_bandits['width_ratio'].quantile(0.9):.1f}x`.
6. **Identification and efficiency are distinct.** In the bandit example,
   signed IPS has asymptotic variance
   `{bandit_efficiency.iloc[0]['ips_asymptotic_variance']:.4f}`, while the
   semiparametric efficiency bound is
   `{bandit_efficiency.iloc[0]['efficiency_bound']:.4f}`, a
   `{bandit_efficiency.iloc[0]['asymptotic_variance_ratio']:.2f}x` gap removed
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
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")
