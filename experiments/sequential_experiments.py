"""Sequential shared-component experiments."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from contrastive_coverage.sequential import (
    MixtureChain, contrastive_mixture_interval, separate_mixture_interval,
    sequential_confidence_interval,
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
    fig.savefig(FIG_DIR / "sequential_identification_width.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame

def experiment_sequential_certificates(rng: np.random.Generator) -> pd.DataFrame:
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
    true_delta = env.exact_difference(q, target_good, q, baseline_good)
    sample_bound = env.deterministic_sample_bound(
        target_common_weight=q,
        target_good_prob=target_good,
        baseline_common_weight=q,
        baseline_good_prob=baseline_good,
    )
    target_supported = (1.0 - q) * env.supported_value(target_good)
    baseline_supported = (1.0 - q) * env.supported_value(baseline_good)
    separate_population = separate_mixture_interval(
        target_supported,
        baseline_supported,
        q,
        q,
        common_value_lower=0.0,
        common_value_upper=env.max_return,
    )

    rows: list[dict[str, float]] = []
    repetitions = 600
    for n in (200, 500, 1000, 2000, 5000, 10000):
        covered = 0
        certified = 0
        widths: list[float] = []
        lower_bounds: list[float] = []
        for _ in range(repetitions):
            actions, rewards = env.sample_behavior(n, rng)
            samples = env.signed_pdis_samples(
                actions,
                rewards,
                target_common_weight=q,
                target_good_prob=target_good,
                baseline_common_weight=q,
                baseline_good_prob=baseline_good,
            )
            interval = sequential_confidence_interval(
                samples,
                target_common_weight=q,
                baseline_common_weight=q,
                common_value_lower=0.0,
                common_value_upper=env.max_return,
                sample_abs_bound=sample_bound,
                delta=0.05,
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
                "sample_abs_bound": sample_bound,
                "separate_population_lower": separate_population.lower,
                "separate_population_width": separate_population.width,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "sequential_certificate.csv", index=False)

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
    axis.set_title("Sequential confidence and certification")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sequential_certificate.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "sequential_certificate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame
