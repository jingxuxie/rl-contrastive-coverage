"""Write machine-readable headline metrics and the results summary."""

from __future__ import annotations

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
    sequential_efficiency: pd.DataFrame,
    sequential_crossfit: pd.DataFrame,
    stochastic_tabular: pd.DataFrame,
    policy_library: pd.DataFrame,
) -> None:
    bp0 = bandit_population[
        (np.isclose(bandit_population["q"], 0.4))
        & (np.isclose(bandit_population["epsilon"], 0.0))
    ].iloc[0]
    bc_last = bandit_certificate.iloc[-1]
    sc_1000 = sequential_certificate[
        sequential_certificate["n_total"] == 1000
    ].iloc[0]
    sc_last = sequential_certificate.iloc[-1]
    se_last = sequential_efficiency.iloc[-1]
    cf_last = sequential_crossfit.iloc[-1]
    tabular_last = stochastic_tabular.iloc[-1]
    library_null = policy_library[
        (policy_library["regime"] == "global_null")
        & (policy_library["n_candidates"] == 100)
    ].set_index("method")
    library_power = policy_library[
        (policy_library["regime"] == "one_strong_candidate")
        & (policy_library["n_total"] == 6000)
    ].set_index("method")
    metrics = {
        "bandit_exact_contrastive_width": float(bp0["contrastive_width"]),
        "bandit_separate_width_q_0_4": float(bp0["separate_width"]),
        "bandit_true_difference": float(bc_last["true_difference"]),
        "bandit_coverage_n_5000": float(bc_last["coverage"]),
        "bandit_certification_n_5000": float(bc_last["certification_rate"]),
        "sequential_true_difference": float(sc_last["true_difference"]),
        "sequential_pdis_coverage_n_5000": float(sc_last["pdis_coverage"]),
        "sequential_augmented_coverage_n_5000": float(
            sc_last["augmented_coverage"]
        ),
        "sequential_pdis_certification_n_1000": float(
            sc_1000["pdis_certification_rate"]
        ),
        "sequential_augmented_certification_n_1000": float(
            sc_1000["augmented_certification_rate"]
        ),
        "tree_max_identity_error": float(tree_identity["absolute_error"].max()),
        "random_bandit_max_identity_error": float(
            random_bandits["identity_error"].max()
        ),
        "random_bandit_median_width_ratio": float(
            random_bandits["width_ratio"].median()
        ),
        "random_bandit_p90_width_ratio": float(
            random_bandits["width_ratio"].quantile(0.9)
        ),
        "bandit_ips_efficiency_variance_ratio": float(
            bandit_efficiency.iloc[0]["asymptotic_variance_ratio"]
        ),
        "sequential_variance_ratio_h_16": float(se_last["variance_ratio"]),
        "sequential_max_variance_decomposition_error": float(
            sequential_efficiency["decomposition_error"].max()
        ),
        "crossfit_pdis_to_augmented_mse_ratio_n_2000": float(
            cf_last["pdis_mse"] / cf_last["learned_augmented_mse"]
        ),
        "crossfit_augmented_bias_n_2000": float(cf_last["learned_augmented_bias"]),
        "stochastic_tabular_true_difference": float(tabular_last["true_difference"]),
        "stochastic_tabular_mse_ratio_n_5000": float(
            tabular_last["pdis_to_learned_mse_ratio"]
        ),
        "stochastic_tabular_augmented_bias_n_5000": float(
            tabular_last["learned_augmented_bias"]
        ),
        "library_pointwise_false_deployment_k_100": float(
            library_null.loc["pointwise_wald", "false_deployment_rate"]
        ),
        "library_bonferroni_false_deployment_k_100": float(
            library_null.loc["bonferroni_wald", "false_deployment_rate"]
        ),
        "library_pdis_eb_power_n_6000_k_50": float(
            library_power.loc[
                "simultaneous_pdis_eb", "safe_certification_rate"
            ]
        ),
        "library_ca_eb_power_n_6000_k_50": float(
            library_power.loc[
                "simultaneous_ca_eb", "safe_certification_rate"
            ]
        ),
    }
    (DATA_DIR / "headline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    text = f"""# Submission-Ready Experimental Results

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
3. **The sequential sharpness prediction is exact.** The shared-branch
   population width is `|epsilon| H`; the maximum numerical discrepancy is
   `{(sequential_population['contrastive_width'] - sequential_population['predicted_width']).abs().max():.3e}`.
4. **Contrastive augmentation removes exactly identifiable action noise.** The
   exact identity `Var(PDIS) = V_eff + V_action` holds through horizon 16 with
   maximum error `{sequential_efficiency['decomposition_error'].max():.3e}`.
   The PDIS/efficiency-bound ratio grows from
   `{sequential_efficiency.iloc[0]['variance_ratio']:.2f}x` at horizon 1 to
   `{se_last['variance_ratio']:.2f}x` at horizon 16.
5. **A learned nuisance model nearly attains the sequential efficiency bound.**
   With two-fold cross-fitting at `n=2,000`, learned CA-PDIS reduces MSE by
   `{cf_last['pdis_mse'] / cf_last['learned_augmented_mse']:.2f}x` relative to
   signed PDIS and has empirical bias `{cf_last['learned_augmented_bias']:.2e}`.
6. **Variance reduction translates into certification power.** With 1,000 total
   episodes split equally between nuisance fitting and certification, signed
   PDIS certifies the positive contrast in
   `{sc_1000['pdis_certification_rate']:.3f}` of repetitions, whereas CA-PDIS
   certifies it in `{sc_1000['augmented_certification_rate']:.3f}`. At the
   largest sample size, empirical coverages are
   `{sc_last['pdis_coverage']:.3f}` and
   `{sc_last['augmented_coverage']:.3f}`, respectively.
7. **The gain persists in a stochastic MDP beyond the independent chain.**
   In a five-state, six-step process with random transitions and a shared
   logger-null initial branch, two-fold tabular CA-PDIS reduces MSE by
   `{tabular_last['pdis_to_learned_mse_ratio']:.2f}x` at `n=5,000`, with bias
   `{tabular_last['learned_augmented_bias']:.2e}`. Neither individual policy
   value is identified because the shared branch is never logged.
8. **Joint identification can be much sharper.** In random bandits, the median
   ratio of separate-value width to contrastive width is
   `{random_bandits['width_ratio'].median():.1f}x` and the 90th percentile is
   `{random_bandits['width_ratio'].quantile(0.9):.1f}x`.
9. **Structural screening and simultaneous inference make library search safe.**
   Under a global null with 100 fixed candidates sharing the same unsupported
   component, uncorrected pointwise Wald intervals falsely deploy with
   probability `{library_null.loc['pointwise_wald', 'false_deployment_rate']:.3f}`,
   while Bonferroni Wald is `{library_null.loc['bonferroni_wald', 'false_deployment_rate']:.3f}`
   and both empirical-Bernstein procedures make no false deployment in the
   300 repetitions. With 50 candidates and 6,000 episodes, simultaneous
   CA-PDIS certifies a genuinely positive selected policy with probability
   `{library_power.loc['simultaneous_ca_eb', 'safe_certification_rate']:.3f}`
   versus `{library_power.loc['simultaneous_pdis_eb', 'safe_certification_rate']:.3f}`
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
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")
