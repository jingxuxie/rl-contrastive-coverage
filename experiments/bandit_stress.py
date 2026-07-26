"""Random-bandit stress tests and one-step efficiency experiments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from contrastive_coverage.bandit import (
    bandit_difference, bandit_efficiency_bound, sample_bandit,
    separate_difference_interval, sharp_difference_interval,
    signed_ips_population_variance, signed_ips_samples,
    stratified_difference_estimate,
)
from common import DATA_DIR, FIG_DIR, bandit_policies

def experiment_random_bandits(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    n_instances = 5000
    n_actions = 10
    for _ in range(n_instances):
        support_size = int(rng.integers(2, n_actions))
        support = np.zeros(n_actions, dtype=bool)
        support[rng.choice(n_actions, size=support_size, replace=False)] = True
        unsupported = ~support
        beta = np.zeros(n_actions)
        beta[support] = rng.dirichlet(np.ones(support_size))
        target_supported = rng.dirichlet(np.ones(support_size))
        baseline_supported = rng.dirichlet(np.ones(support_size))
        common_unsupported = rng.dirichlet(np.ones(n_actions - support_size))
        q = float(rng.uniform(0.05, 0.70))
        max_epsilon = min(0.20, 2 * q, 2 * (1 - q))
        epsilon = float(rng.uniform(0.001, max_epsilon))
        q_target = q + epsilon / 2
        q_baseline = q - epsilon / 2
        target = np.zeros(n_actions)
        baseline = np.zeros(n_actions)
        target[support] = (1 - q_target) * target_supported
        baseline[support] = (1 - q_baseline) * baseline_supported
        target[unsupported] = q_target * common_unsupported
        baseline[unsupported] = q_baseline * common_unsupported
        mu = rng.uniform(size=n_actions)
        sharp = sharp_difference_interval(beta, target, baseline, mu)
        separate = separate_difference_interval(beta, target, baseline, mu)
        rows.append(
            {
                "support_size": support_size,
                "q": q,
                "epsilon": epsilon,
                "contrastive_width": sharp.width,
                "separate_width": separate.width,
                "width_ratio": separate.width / sharp.width,
                "identity_error": abs(sharp.width - epsilon),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "random_bandits.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.scatter(
        frame["epsilon"],
        frame["width_ratio"],
        s=7,
        alpha=0.18,
        edgecolors="none",
    )
    axis.set_yscale("log")
    axis.set_xlabel(r"unmatched unsupported mass $|\epsilon|$")
    axis.set_ylabel("separate / contrastive width")
    axis.set_title("Random bandits: gain from joint identification")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "random_bandit_width_ratio.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "random_bandit_width_ratio.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame

def experiment_bandit_efficiency(rng: np.random.Generator) -> pd.DataFrame:
    """Compare signed IPS with the efficient within-action plug-in estimator."""

    beta = np.array([0.5, 0.5, 0.0])
    mu = np.array([0.8, 0.2, 0.55])
    target, baseline = bandit_policies(0.45, 0.0)
    true_delta = bandit_difference(target, baseline, mu)
    ips_var = signed_ips_population_variance(beta, target, baseline, mu)
    efficient_var = bandit_efficiency_bound(beta, target, baseline, mu)
    rows: list[dict[str, float]] = []
    repetitions = 2000
    for n in (50, 100, 200, 500, 1000, 2000):
        ips_errors: list[float] = []
        stratified_errors: list[float] = []
        for _ in range(repetitions):
            actions, rewards = sample_bandit(beta, mu, n, rng)
            ips = float(
                np.mean(signed_ips_samples(actions, rewards, beta, target, baseline))
            )
            stratified = stratified_difference_estimate(
                actions, rewards, target, baseline
            )
            ips_errors.append((ips - true_delta) ** 2)
            stratified_errors.append((stratified - true_delta) ** 2)
        rows.append(
            {
                "n": n,
                "repetitions": repetitions,
                "true_difference": true_delta,
                "ips_mse": float(np.mean(ips_errors)),
                "stratified_mse": float(np.mean(stratified_errors)),
                "n_times_ips_mse": float(n * np.mean(ips_errors)),
                "n_times_stratified_mse": float(n * np.mean(stratified_errors)),
                "ips_asymptotic_variance": ips_var,
                "efficiency_bound": efficient_var,
                "asymptotic_variance_ratio": ips_var / efficient_var,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "bandit_efficiency.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(
        frame["n"], frame["n_times_ips_mse"], marker="o", label="signed IPS"
    )
    axis.plot(
        frame["n"],
        frame["n_times_stratified_mse"],
        marker="s",
        label="stratified plug-in",
    )
    axis.axhline(ips_var, linestyle="--", linewidth=1.1, label="IPS variance")
    axis.axhline(
        efficient_var,
        linestyle=":",
        linewidth=1.4,
        label="efficiency bound",
    )
    axis.set_xscale("log")
    axis.set_xlabel("logged samples $n$")
    axis.set_ylabel(r"$n \times$ mean squared error")
    axis.set_title("Bandit estimation after identification")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bandit_efficiency.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "bandit_efficiency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame
