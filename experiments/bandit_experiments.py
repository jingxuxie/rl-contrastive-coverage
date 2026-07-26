"""Bandit population and finite-sample certificate experiments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from contrastive_coverage.bandit import (
    bandit_difference, partial_difference_confidence_interval, sample_bandit,
    separate_difference_interval, sharp_difference_interval,
)
from common import DATA_DIR, FIG_DIR, bandit_policies

def experiment_bandit_population() -> pd.DataFrame:
    beta = np.array([0.5, 0.5, 0.0])
    mu = np.array([0.8, 0.2, 0.55])
    rows: list[dict[str, float]] = []
    for q in (0.1, 0.25, 0.4, 0.55, 0.7):
        for epsilon in np.linspace(0.0, min(0.30, 2.0 * q), 31):
            target, baseline = bandit_policies(q, float(epsilon))
            sharp = sharp_difference_interval(beta, target, baseline, mu)
            separate = separate_difference_interval(beta, target, baseline, mu)
            rows.append(
                {
                    "q": q,
                    "epsilon": float(epsilon),
                    "true_difference": bandit_difference(target, baseline, mu),
                    "contrastive_lower": sharp.lower,
                    "contrastive_upper": sharp.upper,
                    "contrastive_width": sharp.width,
                    "separate_lower": separate.lower,
                    "separate_upper": separate.upper,
                    "separate_width": separate.width,
                    "unmatched_mass": abs(float(epsilon)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "bandit_population_grid.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45))
    for q, group in frame.groupby("q"):
        axes[0].plot(
            group["epsilon"], group["contrastive_width"], label=fr"$q={q:.2g}$"
        )
        axes[1].plot(group["epsilon"], group["separate_width"], label=fr"$q={q:.2g}$")
    axes[0].plot([0, 0.3], [0, 0.3], "k--", linewidth=1.0, label=r"$|\epsilon|$")
    axes[0].set_title("Contrastive identified-set width")
    axes[1].set_title("Separate-value bound width")
    for axis in axes:
        axis.set_xlabel(r"unmatched unsupported mass $|\epsilon|$")
        axis.set_ylabel("interval width")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bandit_identification_width.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "bandit_identification_width.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame

def experiment_bandit_certificates(rng: np.random.Generator) -> pd.DataFrame:
    beta = np.array([0.5, 0.5, 0.0])
    mu = np.array([0.8, 0.2, 0.55])
    q = 0.45
    target, baseline = bandit_policies(q, 0.0)
    true_delta = bandit_difference(target, baseline, mu)
    separate_population = separate_difference_interval(beta, target, baseline, mu)
    rows: list[dict[str, float]] = []
    repetitions = 1000
    for n in (100, 200, 500, 1000, 2000, 5000):
        covered = 0
        certified = 0
        widths: list[float] = []
        lower_bounds: list[float] = []
        for _ in range(repetitions):
            actions, rewards = sample_bandit(beta, mu, n, rng)
            interval = partial_difference_confidence_interval(
                actions, rewards, beta, target, baseline, delta=0.05
            )
            covered += int(interval.contains(true_delta))
            certified += int(interval.lower > 0.0)
            widths.append(interval.width)
            lower_bounds.append(interval.lower)
        rows.append(
            {
                "n": n,
                "repetitions": repetitions,
                "true_difference": true_delta,
                "coverage": covered / repetitions,
                "certification_rate": certified / repetitions,
                "mean_width": float(np.mean(widths)),
                "median_width": float(np.median(widths)),
                "mean_lower_bound": float(np.mean(lower_bounds)),
                "separate_population_lower": separate_population.lower,
                "separate_population_width": separate_population.width,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "bandit_certificate.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(frame["n"], frame["coverage"], marker="o", label="95% interval coverage")
    axis.plot(
        frame["n"],
        frame["certification_rate"],
        marker="s",
        label="certifies improvement",
    )
    axis.axhline(0.95, linestyle="--", linewidth=1.0, label="nominal coverage")
    axis.set_xscale("log")
    axis.set_ylim(-0.02, 1.03)
    axis.set_xlabel("logged episodes $n$")
    axis.set_ylabel("probability")
    axis.set_title("Bandit confidence and certification")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bandit_certificate.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "bandit_certificate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame
