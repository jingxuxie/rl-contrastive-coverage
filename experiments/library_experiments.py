"""Policy-library selection under contrastive coverage and multiplicity."""

from __future__ import annotations

from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from common import DATA_DIR, FIG_DIR


def _policy_library(
    *,
    n_supported: int = 20,
    n_candidates: int = 100,
    common_unsupported_mass: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return logger, baseline, candidate library, and alternative rewards."""

    beta = np.concatenate(
        [np.full(n_supported, 1.0 / n_supported), np.array([0.0])]
    )
    baseline = np.concatenate(
        [
            np.full(
                n_supported,
                (1.0 - common_unsupported_mass) / n_supported,
            ),
            np.array([common_unsupported_mass]),
        ]
    )

    library_rng = np.random.default_rng(87123)
    supported = library_rng.dirichlet(
        np.full(n_supported, 8.0), size=n_candidates
    )
    supported *= 1.0 - common_unsupported_mass

    reward_means = np.concatenate(
        [np.linspace(0.4, 0.6, n_supported), np.array([0.5])]
    )
    # Include one deliberately strong candidate, then fill the remainder with
    # fixed random proposals.  Every candidate shares exactly the same
    # logger-null action mass as the baseline.
    tilt = np.exp(4.0 * (reward_means[:n_supported] - 0.5))
    supported[0] = (1.0 - common_unsupported_mass) * tilt / tilt.sum()
    candidates = np.column_stack(
        [supported, np.full(n_candidates, common_unsupported_mass)]
    )
    return beta, baseline, candidates, reward_means


def _empirical_bernstein_radii(
    samples: np.ndarray,
    *,
    delta: float,
    range_bounds: np.ndarray,
) -> np.ndarray:
    """Vectorized Maurer--Pontil radii for one row per candidate."""

    n = samples.shape[1]
    variances = np.var(samples, axis=1, ddof=1)
    log_term = np.log(4.0 / delta)
    return np.sqrt(2.0 * variances * log_term / n) + (
        7.0 * range_bounds * log_term / (3.0 * (n - 1))
    )


def _one_library_evaluation(
    rng: np.random.Generator,
    *,
    n_total: int,
    n_candidates: int,
    beta: np.ndarray,
    baseline: np.ndarray,
    candidates: np.ndarray,
    reward_means: np.ndarray,
    family_delta: float = 0.05,
) -> dict[str, np.ndarray]:
    """Fit a reward model on one split and evaluate every policy on the other."""

    n_supported = beta.size - 1
    n_train = n_total // 2
    n_eval = n_total - n_train

    train_actions = rng.choice(
        n_supported, size=n_train, p=beta[:n_supported]
    )
    train_rewards = rng.binomial(1, reward_means[train_actions]).astype(float)
    counts = np.bincount(train_actions, minlength=n_supported)
    reward_sums = np.bincount(
        train_actions, weights=train_rewards, minlength=n_supported
    )
    # The fixed Beta(1,1) smoothing keeps the fitted means in [0,1].  Exact
    # split-sample unbiasedness does not require the model to be correct.
    reward_model = (reward_sums + 1.0) / (counts + 2.0)

    eval_actions = rng.choice(n_supported, size=n_eval, p=beta[:n_supported])
    eval_rewards = rng.binomial(1, reward_means[eval_actions]).astype(float)

    differences = candidates[:n_candidates, :n_supported] - baseline[:n_supported]
    action_weights = differences[:, eval_actions] / beta[eval_actions]

    pdis_samples = action_weights * eval_rewards
    pdis_means = np.mean(pdis_samples, axis=1)
    weight_table = differences / beta[:n_supported]
    pdis_ranges = np.maximum(0.0, weight_table.max(axis=1)) - np.minimum(
        0.0, weight_table.min(axis=1)
    )

    fitted_values = differences @ reward_model
    ca_samples = fitted_values[:, None] + action_weights * (
        eval_rewards - reward_model[eval_actions]
    )
    ca_means = np.mean(ca_samples, axis=1)
    low_values = fitted_values[:, None] - weight_table * reward_model[None, :]
    high_values = fitted_values[:, None] + weight_table * (
        1.0 - reward_model[None, :]
    )
    ca_ranges = np.maximum(low_values.max(axis=1), high_values.max(axis=1)) - np.minimum(
        low_values.min(axis=1), high_values.min(axis=1)
    )

    pointwise_z = NormalDist().inv_cdf(1.0 - family_delta)
    bonferroni_z = NormalDist().inv_cdf(1.0 - family_delta / n_candidates)
    pdis_standard_errors = np.std(pdis_samples, axis=1, ddof=1) / np.sqrt(n_eval)

    pdis_eb = _empirical_bernstein_radii(
        pdis_samples,
        delta=family_delta / n_candidates,
        range_bounds=pdis_ranges,
    )
    ca_eb = _empirical_bernstein_radii(
        ca_samples,
        delta=family_delta / n_candidates,
        range_bounds=ca_ranges,
    )

    return {
        "pointwise_wald": pdis_means - pointwise_z * pdis_standard_errors,
        "bonferroni_wald": pdis_means - bonferroni_z * pdis_standard_errors,
        "simultaneous_pdis_eb": pdis_means - pdis_eb,
        "simultaneous_ca_eb": ca_means - ca_eb,
    }


def experiment_policy_library_selection(
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Measure post-selection error and power over a fixed policy library."""

    beta, baseline, candidates, alternative_means = _policy_library()
    null_means = np.concatenate(
        [np.full(beta.size - 1, 0.5), np.array([0.5])]
    )
    methods = (
        "pointwise_wald",
        "bonferroni_wald",
        "simultaneous_pdis_eb",
        "simultaneous_ca_eb",
    )

    rows: list[dict[str, float | int | str]] = []
    null_repetitions = 300
    for n_candidates in (1, 5, 20, 50, 100):
        counts = {method: 0 for method in methods}
        for _ in range(null_repetitions):
            lower_bounds = _one_library_evaluation(
                rng,
                n_total=4000,
                n_candidates=n_candidates,
                beta=beta,
                baseline=baseline,
                candidates=candidates,
                reward_means=null_means,
            )
            for method in methods:
                counts[method] += int(np.max(lower_bounds[method]) > 0.0)
        for method in methods:
            rate = counts[method] / null_repetitions
            rows.append(
                {
                    "regime": "global_null",
                    "method": method,
                    "n_total": 4000,
                    "n_candidates": n_candidates,
                    "repetitions": null_repetitions,
                    "false_deployment_rate": rate,
                    "safe_certification_rate": np.nan,
                    "monte_carlo_se": np.sqrt(
                        rate * (1.0 - rate) / null_repetitions
                    ),
                }
            )

    alternative_repetitions = 300
    true_differences = (candidates - baseline[None, :]) @ alternative_means
    for n_total in (1000, 2000, 3000, 4000, 6000, 8000):
        counts = {method: 0 for method in methods}
        for _ in range(alternative_repetitions):
            lower_bounds = _one_library_evaluation(
                rng,
                n_total=n_total,
                n_candidates=50,
                beta=beta,
                baseline=baseline,
                candidates=candidates,
                reward_means=alternative_means,
            )
            for method in methods:
                lcb = lower_bounds[method]
                selected = int(np.argmax(lcb))
                counts[method] += int(
                    lcb[selected] > 0.0 and true_differences[selected] > 0.0
                )
        for method in methods:
            rate = counts[method] / alternative_repetitions
            rows.append(
                {
                    "regime": "one_strong_candidate",
                    "method": method,
                    "n_total": n_total,
                    "n_candidates": 50,
                    "repetitions": alternative_repetitions,
                    "false_deployment_rate": np.nan,
                    "safe_certification_rate": rate,
                    "monte_carlo_se": np.sqrt(
                        rate * (1.0 - rate) / alternative_repetitions
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "policy_library_selection.csv", index=False)

    labels = {
        "pointwise_wald": "pointwise Wald (uncorrected)",
        "bonferroni_wald": "Bonferroni Wald",
        "simultaneous_pdis_eb": "simultaneous PDIS-EB",
        "simultaneous_ca_eb": "simultaneous CA-PDIS-EB",
    }
    markers = {
        "pointwise_wald": "o",
        "bonferroni_wald": "s",
        "simultaneous_pdis_eb": "^",
        "simultaneous_ca_eb": "D",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.85))
    null_frame = frame[frame["regime"] == "global_null"]
    for method in methods:
        group = null_frame[null_frame["method"] == method]
        axes[0].plot(
            group["n_candidates"],
            group["false_deployment_rate"],
            marker=markers[method],
            label=labels[method],
        )
    axes[0].axhline(0.05, linestyle="--", linewidth=1.0, label="nominal 5%")
    axes[0].set_xscale("log")
    axes[0].set_xticks((1, 5, 20, 50, 100))
    axes[0].set_xticklabels(("1", "5", "20", "50", "100"))
    axes[0].set_ylim(-0.025, 1.0)
    axes[0].set_xlabel("number of searched candidates $K$")
    axes[0].set_ylabel("false-deployment probability")
    axes[0].set_title("Post-selection error under the global null")
    axes[0].grid(alpha=0.25)

    alternative = frame[frame["regime"] == "one_strong_candidate"]
    for method in (
        "bonferroni_wald",
        "simultaneous_pdis_eb",
        "simultaneous_ca_eb",
    ):
        group = alternative[alternative["method"] == method]
        axes[1].plot(
            group["n_total"],
            group["safe_certification_rate"],
            marker=markers[method],
            label=labels[method],
        )
    axes[1].set_xscale("log")
    axes[1].set_xticks((1000, 2000, 4000, 8000))
    axes[1].set_xticklabels(("1k", "2k", "4k", "8k"))
    axes[1].set_ylim(-0.025, 1.025)
    axes[1].set_xlabel("total logged episodes")
    axes[1].set_ylabel("safe certification probability")
    axes[1].set_title("Power after searching $K=50$ candidates")
    axes[1].grid(alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.savefig(FIG_DIR / "policy_library_selection.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "policy_library_selection.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)
    return frame
