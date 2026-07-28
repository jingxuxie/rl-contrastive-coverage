"""Sequential shared-component experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from contrastive_coverage.bandit import empirical_bernstein_radius
from contrastive_coverage.efficient import empirical_pooled_action_means
from contrastive_coverage.sequential import (
    MixtureChain,
    contrastive_mixture_interval,
    separate_mixture_interval,
)
from common import DATA_DIR, FIG_DIR


def experiment_sequential_population() -> pd.DataFrame:
    env = MixtureChain(
        horizon=5,
        behavior_good_prob=0.5,
        reward_good_mean=0.75,
        reward_bad_mean=0.25,
        hidden_return=2.5,
    )
    q = 0.4
    target_good = 0.7
    baseline_good = 0.3
    rows: list[dict[str, float]] = []
    for epsilon in np.linspace(0.0, 0.40, 41):
        q_target = q + epsilon / 2.0
        q_baseline = q - epsilon / 2.0
        identified = env.identified_supported_difference(
            q_target, target_good, q_baseline, baseline_good
        )
        contrastive = contrastive_mixture_interval(
            identified,
            q_target,
            q_baseline,
            common_value_lower=0.0,
            common_value_upper=env.max_return,
        )
        target_supported = (1.0 - q_target) * env.supported_value(target_good)
        baseline_supported = (1.0 - q_baseline) * env.supported_value(baseline_good)
        separate = separate_mixture_interval(
            target_supported,
            baseline_supported,
            q_target,
            q_baseline,
            common_value_lower=0.0,
            common_value_upper=env.max_return,
        )
        rows.append(
            {
                "epsilon": float(epsilon),
                "true_difference": env.exact_difference(
                    q_target, target_good, q_baseline, baseline_good
                ),
                "contrastive_lower": contrastive.lower,
                "contrastive_upper": contrastive.upper,
                "contrastive_width": contrastive.width,
                "separate_lower": separate.lower,
                "separate_upper": separate.upper,
                "separate_width": separate.width,
                "predicted_width": float(epsilon * env.max_return),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "sequential_population_grid.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(
        frame["epsilon"],
        frame["contrastive_width"],
        linewidth=2.0,
        label="contrastive bound",
    )
    axis.plot(
        frame["epsilon"],
        frame["separate_width"],
        linewidth=2.0,
        label="separate-value bounds",
    )
    axis.plot(
        frame["epsilon"],
        frame["predicted_width"],
        linestyle="--",
        linewidth=1.2,
        label=r"$|\epsilon|H$ prediction",
    )
    axis.set_xlabel(r"common-component weight mismatch $|\epsilon|$")
    axis.set_ylabel("population interval width")
    axis.set_title("Sequential shared-branch cancellation")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sequential_identification_width.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "sequential_identification_width.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    return frame


def _bounded_mean_interval(
    samples: np.ndarray,
    *,
    lower_bound: float,
    upper_bound: float,
    delta: float = 0.05,
) -> tuple[float, float]:
    radius = empirical_bernstein_radius(
        samples,
        delta=delta,
        range_bound=upper_bound - lower_bound,
    )
    estimate = float(np.mean(samples))
    return estimate - radius, estimate + radius


def experiment_sequential_certificates(rng: np.random.Generator) -> pd.DataFrame:
    """Compare signed PDIS and split-fitted CA-PDIS certificates fairly.

    Half of each dataset is used only to fit the stationary two-mean nuisance
    model.  Both estimators are certified on the same untouched half.  Conditional
    on the training split, the augmented episode contributions are independent,
    unbiased, and deterministically bounded, so the same empirical-Bernstein
    argument applies.
    """

    env = MixtureChain(
        horizon=8,
        behavior_good_prob=0.5,
        reward_good_mean=0.7,
        reward_bad_mean=0.3,
        hidden_return=4.0,
    )
    q = 0.35
    target_good = 0.60
    baseline_good = 0.40
    true_delta = env.exact_difference(q, target_good, q, baseline_good)
    pdis_lower, pdis_upper = env.exact_contribution_range(
        estimator="pdis",
        target_common_weight=q,
        target_good_prob=target_good,
        baseline_common_weight=q,
        baseline_good_prob=baseline_good,
    )

    rows: list[dict[str, float]] = []
    repetitions = 500
    for n_total in (200, 500, 1000, 2000, 5000):
        pdis_covered = 0
        augmented_covered = 0
        pdis_certified = 0
        augmented_certified = 0
        pdis_widths: list[float] = []
        augmented_widths: list[float] = []
        pdis_squared_errors: list[float] = []
        augmented_squared_errors: list[float] = []
        for _ in range(repetitions):
            actions, rewards = env.sample_behavior(n_total, rng)
            permutation = rng.permutation(n_total)
            n_train = n_total // 2
            train = permutation[:n_train]
            evaluate = permutation[n_train:]
            fitted_means = empirical_pooled_action_means(
                actions[train],
                rewards[train],
                prior_count=1.0,
                prior_mean=0.5,
            )

            pdis_samples = env.signed_pdis_samples(
                actions[evaluate],
                rewards[evaluate],
                target_common_weight=q,
                target_good_prob=target_good,
                baseline_common_weight=q,
                baseline_good_prob=baseline_good,
            )
            augmented_samples = env.contrastive_augmented_samples(
                actions[evaluate],
                rewards[evaluate],
                target_common_weight=q,
                target_good_prob=target_good,
                baseline_common_weight=q,
                baseline_good_prob=baseline_good,
                stage_action_means=fitted_means,
            )
            augmented_lower, augmented_upper = env.exact_contribution_range(
                estimator="augmented",
                target_common_weight=q,
                target_good_prob=target_good,
                baseline_common_weight=q,
                baseline_good_prob=baseline_good,
                stage_action_means=fitted_means,
            )

            pdis_interval = _bounded_mean_interval(
                pdis_samples,
                lower_bound=pdis_lower,
                upper_bound=pdis_upper,
            )
            augmented_interval = _bounded_mean_interval(
                augmented_samples,
                lower_bound=augmented_lower,
                upper_bound=augmented_upper,
            )
            pdis_estimate = float(np.mean(pdis_samples))
            augmented_estimate = float(np.mean(augmented_samples))
            pdis_covered += int(pdis_interval[0] <= true_delta <= pdis_interval[1])
            augmented_covered += int(
                augmented_interval[0] <= true_delta <= augmented_interval[1]
            )
            pdis_certified += int(pdis_interval[0] > 0.0)
            augmented_certified += int(augmented_interval[0] > 0.0)
            pdis_widths.append(pdis_interval[1] - pdis_interval[0])
            augmented_widths.append(augmented_interval[1] - augmented_interval[0])
            pdis_squared_errors.append((pdis_estimate - true_delta) ** 2)
            augmented_squared_errors.append((augmented_estimate - true_delta) ** 2)

        rows.append(
            {
                "n_total": n_total,
                "n_train": n_total // 2,
                "n_certification": n_total - n_total // 2,
                "repetitions": repetitions,
                "true_difference": true_delta,
                "pdis_coverage": pdis_covered / repetitions,
                "augmented_coverage": augmented_covered / repetitions,
                "pdis_certification_rate": pdis_certified / repetitions,
                "augmented_certification_rate": augmented_certified / repetitions,
                "pdis_mean_width": float(np.mean(pdis_widths)),
                "augmented_mean_width": float(np.mean(augmented_widths)),
                "pdis_mse": float(np.mean(pdis_squared_errors)),
                "augmented_mse": float(np.mean(augmented_squared_errors)),
                "pdis_range_width": pdis_upper - pdis_lower,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "sequential_certificate.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(
        frame["n_total"],
        frame["pdis_certification_rate"],
        marker="o",
        label="signed PDIS: certifies",
    )
    axis.plot(
        frame["n_total"],
        frame["augmented_certification_rate"],
        marker="s",
        label="CA-PDIS: certifies",
    )
    axis.plot(
        frame["n_total"],
        frame["pdis_coverage"],
        linestyle="--",
        linewidth=1.0,
        label="PDIS coverage",
    )
    axis.plot(
        frame["n_total"],
        frame["augmented_coverage"],
        linestyle=":",
        linewidth=1.4,
        label="CA-PDIS coverage",
    )
    axis.axhline(0.95, linestyle="-.", linewidth=1.0, label="nominal coverage")
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("total logged episodes (half used for nuisance fitting)")
    axis.set_ylabel("probability")
    axis.set_title("Split-fitted sequential certification")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sequential_certificate.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "sequential_certificate.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)
    return frame


def experiment_sequential_efficiency() -> pd.DataFrame:
    """Verify the exact PDIS = efficient + action-noise decomposition."""

    rows: list[dict[str, float]] = []
    for horizon in range(1, 17):
        env = MixtureChain(
            horizon=horizon,
            behavior_good_prob=0.5,
            reward_good_mean=0.7,
            reward_bad_mean=0.3,
            hidden_return=horizon / 2.0,
        )
        result = env.exact_variance_decomposition(
            target_common_weight=0.35,
            target_good_prob=0.65,
            baseline_common_weight=0.35,
            baseline_good_prob=0.35,
        )
        rows.append(
            {
                "horizon": horizon,
                "pdis_variance": result.pdis_variance,
                "efficient_variance": result.efficient_variance,
                "action_noise_variance": result.action_noise_variance,
                "variance_ratio": result.variance_ratio,
                "action_noise_fraction": (
                    result.action_noise_variance / result.pdis_variance
                ),
                "decomposition_error": result.decomposition_error,
                "true_difference": result.true_difference,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "sequential_efficiency.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(
        frame["horizon"],
        frame["variance_ratio"],
        marker="o",
        linewidth=2.0,
        label=r"$\mathrm{Var}(\mathrm{PDIS})/\mathcal{V}_{\mathrm{eff}}$",
    )
    axis.axhline(1.0, linestyle="--", linewidth=1.0, label="efficient benchmark")
    axis.set_xlabel("horizon $H$")
    axis.set_ylabel("exact per-episode variance ratio")
    axis.set_title("Action-composition noise grows with horizon")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sequential_efficiency.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "sequential_efficiency.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)
    return frame


def experiment_crossfit_augmented_mse(rng: np.random.Generator) -> pd.DataFrame:
    """Compare PDIS, two-fold learned CA-PDIS, and oracle CA-PDIS."""

    env = MixtureChain(
        horizon=8,
        behavior_good_prob=0.5,
        reward_good_mean=0.7,
        reward_bad_mean=0.3,
        hidden_return=4.0,
    )
    q = 0.35
    target_good = 0.65
    baseline_good = 0.35
    true_delta = env.exact_difference(q, target_good, q, baseline_good)
    exact = env.exact_variance_decomposition(
        target_common_weight=q,
        target_good_prob=target_good,
        baseline_common_weight=q,
        baseline_good_prob=baseline_good,
    )
    rows: list[dict[str, float]] = []
    repetitions_by_n = {50: 1200, 100: 1000, 200: 800, 500: 600, 1000: 500, 2000: 500}
    for n, repetitions in repetitions_by_n.items():
        pdis_errors: list[float] = []
        learned_errors: list[float] = []
        oracle_errors: list[float] = []
        learned_biases: list[float] = []
        for _ in range(repetitions):
            actions, rewards = env.sample_behavior(n, rng)
            pdis = float(
                np.mean(
                    env.signed_pdis_samples(
                        actions,
                        rewards,
                        target_common_weight=q,
                        target_good_prob=target_good,
                        baseline_common_weight=q,
                        baseline_good_prob=baseline_good,
                    )
                )
            )
            oracle = float(
                np.mean(
                    env.contrastive_augmented_samples(
                        actions,
                        rewards,
                        target_common_weight=q,
                        target_good_prob=target_good,
                        baseline_common_weight=q,
                        baseline_good_prob=baseline_good,
                    )
                )
            )

            permutation = rng.permutation(n)
            first_fold = np.zeros(n, dtype=bool)
            first_fold[permutation[: n // 2]] = True
            learned_contributions = np.empty(n, dtype=float)
            for evaluation_mask in (first_fold, ~first_fold):
                training_mask = ~evaluation_mask
                fitted_means = empirical_pooled_action_means(
                    actions[training_mask],
                    rewards[training_mask],
                    prior_count=1.0,
                    prior_mean=0.5,
                )
                learned_contributions[evaluation_mask] = (
                    env.contrastive_augmented_samples(
                        actions[evaluation_mask],
                        rewards[evaluation_mask],
                        target_common_weight=q,
                        target_good_prob=target_good,
                        baseline_common_weight=q,
                        baseline_good_prob=baseline_good,
                        stage_action_means=fitted_means,
                    )
                )
            learned = float(np.mean(learned_contributions))
            pdis_errors.append((pdis - true_delta) ** 2)
            learned_errors.append((learned - true_delta) ** 2)
            oracle_errors.append((oracle - true_delta) ** 2)
            learned_biases.append(learned - true_delta)
        rows.append(
            {
                "n": n,
                "repetitions": repetitions,
                "true_difference": true_delta,
                "pdis_mse": float(np.mean(pdis_errors)),
                "learned_augmented_mse": float(np.mean(learned_errors)),
                "oracle_augmented_mse": float(np.mean(oracle_errors)),
                "learned_augmented_bias": float(np.mean(learned_biases)),
                "paired_mse_gain": float(
                    np.mean(np.asarray(pdis_errors) - np.asarray(learned_errors))
                ),
                "paired_mse_gain_se": float(
                    np.std(
                        np.asarray(pdis_errors) - np.asarray(learned_errors), ddof=1
                    )
                    / np.sqrt(repetitions)
                ),
                "n_times_pdis_mse": float(n * np.mean(pdis_errors)),
                "n_times_learned_augmented_mse": float(
                    n * np.mean(learned_errors)
                ),
                "n_times_oracle_augmented_mse": float(n * np.mean(oracle_errors)),
                "exact_pdis_variance": exact.pdis_variance,
                "efficiency_bound": exact.efficient_variance,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "sequential_crossfit_mse.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(
        frame["n"], frame["n_times_pdis_mse"], marker="o", label="signed PDIS"
    )
    axis.plot(
        frame["n"],
        frame["n_times_learned_augmented_mse"],
        marker="s",
        label="2-fold learned CA-PDIS",
    )
    axis.plot(
        frame["n"],
        frame["n_times_oracle_augmented_mse"],
        marker="^",
        label="oracle CA-PDIS",
    )
    axis.axhline(
        exact.pdis_variance, linestyle="--", linewidth=1.0, label="PDIS variance"
    )
    axis.axhline(
        exact.efficient_variance,
        linestyle=":",
        linewidth=1.4,
        label="efficiency bound",
    )
    axis.set_xscale("log")
    axis.set_xlabel("logged episodes $n$")
    axis.set_ylabel(r"$n \times$ mean squared error")
    axis.set_title("Learned contrastive augmentation approaches efficiency")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sequential_crossfit_mse.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "sequential_crossfit_mse.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)
    return frame
