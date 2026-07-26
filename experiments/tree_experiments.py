"""Exact deterministic-history-tree validation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from contrastive_coverage.tree import (
    contrastive_coverage_violation, enumerate_prefix_records,
    individual_coverage_violation, sharp_tree_difference_interval,
    table_policy, unmatched_discounted_mass,
)
from common import DATA_DIR, FIG_DIR

def experiment_tree_identity(rng: np.random.Generator) -> pd.DataFrame:
    """Numerically verify sharp tree width = unmatched discounted mass."""

    horizon = 4
    n_actions = 3
    discount = 0.9
    # The behavior never takes root action 2. Target and baseline share that
    # unsupported branch when epsilon=0, including all downstream conditionals.
    hidden_default = np.array([0.15, 0.25, 0.60])
    rows: list[dict[str, float]] = []
    for epsilon in np.linspace(0.0, 0.20, 21):
        q = 0.35
        target_root = np.array([0.55 * (1 - q - epsilon / 2), 0.45 * (1 - q - epsilon / 2), q + epsilon / 2])
        baseline_root = np.array([0.30 * (1 - q + epsilon / 2), 0.70 * (1 - q + epsilon / 2), q - epsilon / 2])
        behavior = table_policy({(): [0.5, 0.5, 0.0]}, default=[0.4, 0.35, 0.25])
        target = table_policy({(): target_root}, default=hidden_default)
        baseline = table_policy({(): baseline_root}, default=hidden_default)
        records = enumerate_prefix_records(
            behavior, target, baseline, horizon=horizon, n_actions=n_actions
        )
        observed = {
            record.prefix: float(rng.uniform())
            for record in records
            if record.behavior_likelihood > 1e-12
        }
        interval = sharp_tree_difference_interval(
            records,
            observed,
            discount=discount,
            reward_lower=0.0,
            reward_upper=1.0,
        )
        unmatched = unmatched_discounted_mass(records, discount=discount)
        rows.append(
            {
                "epsilon": float(epsilon),
                "interval_width": interval.width,
                "unmatched_discounted_mass": unmatched,
                "absolute_error": abs(interval.width - unmatched),
                "contrastive_violation": contrastive_coverage_violation(records),
                "target_individual_violation": individual_coverage_violation(
                    records, which="target"
                ),
                "baseline_individual_violation": individual_coverage_violation(
                    records, which="baseline"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "tree_sharpness_identity.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(frame["epsilon"], frame["interval_width"], marker="o", label="sharp interval width")
    axis.plot(
        frame["epsilon"],
        frame["unmatched_discounted_mass"],
        linestyle="--",
        label="theorem: unmatched mass",
    )
    axis.set_xlabel(r"root unsupported-mass mismatch $|\epsilon|$")
    axis.set_ylabel("discounted width")
    axis.set_title("Sharp tree identification identity")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tree_sharpness_identity.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "tree_sharpness_identity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame
