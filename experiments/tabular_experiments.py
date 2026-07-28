"""Stochastic tabular validation beyond the independent-chain construction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

from contrastive_coverage.tabular import (
    fit_tabular_model,
    make_stochastic_validation_process,
)
from common import DATA_DIR, FIG_DIR


def experiment_stochastic_tabular_crossfit(
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compare signed PDIS and two-fold CA-PDIS in a stochastic MDP.

    The two target policies share a logger-null initial branch with probability
    0.25, so neither absolute value is identified.  On the logged branch the
    process has five states, stochastic transitions, and state/time-dependent
    rewards.  CA-PDIS fits a smoothed tabular model on the opposite fold.
    """

    process = make_stochastic_validation_process(
        horizon=6,
        n_states=5,
        common_weight=0.25,
        discount=0.98,
    )
    true_delta = process.true_difference()
    rows: list[dict[str, float]] = []
    repetitions_by_n = {250: 1000, 500: 800, 1000: 600, 2000: 400, 5000: 250}

    for n, repetitions in repetitions_by_n.items():
        pdis_errors: list[float] = []
        learned_errors: list[float] = []
        oracle_errors: list[float] = []
        learned_signed_errors: list[float] = []
        for _ in range(repetitions):
            states, actions, rewards = process.sample_behavior(n, rng)
            pdis_estimate = float(
                np.mean(process.signed_pdis_samples(states, actions, rewards))
            )
            oracle_estimate = float(
                np.mean(
                    process.contrastive_augmented_samples(
                        states,
                        actions,
                        rewards,
                        fitted_model=process.model,
                    )
                )
            )

            permutation = rng.permutation(n)
            first_fold = np.zeros(n, dtype=bool)
            first_fold[permutation[: n // 2]] = True
            learned_contributions = np.empty(n, dtype=float)
            for evaluation_mask in (first_fold, ~first_fold):
                training_mask = ~evaluation_mask
                fitted_model = fit_tabular_model(
                    states[training_mask],
                    actions[training_mask],
                    rewards[training_mask],
                    n_states=process.model.n_states,
                    n_actions=process.model.n_actions,
                    reward_prior_mean=0.5,
                    reward_prior_count=1.0,
                    transition_prior_count=1.0,
                )
                learned_contributions[evaluation_mask] = (
                    process.contrastive_augmented_samples(
                        states[evaluation_mask],
                        actions[evaluation_mask],
                        rewards[evaluation_mask],
                        fitted_model=fitted_model,
                    )
                )
            learned_estimate = float(np.mean(learned_contributions))

            pdis_errors.append((pdis_estimate - true_delta) ** 2)
            learned_errors.append((learned_estimate - true_delta) ** 2)
            oracle_errors.append((oracle_estimate - true_delta) ** 2)
            learned_signed_errors.append(learned_estimate - true_delta)

        pdis_array = np.asarray(pdis_errors)
        learned_array = np.asarray(learned_errors)
        oracle_array = np.asarray(oracle_errors)
        rows.append(
            {
                "n": n,
                "repetitions": repetitions,
                "true_difference": true_delta,
                "pdis_mse": float(np.mean(pdis_array)),
                "learned_augmented_mse": float(np.mean(learned_array)),
                "oracle_augmented_mse": float(np.mean(oracle_array)),
                "pdis_mse_se": float(np.std(pdis_array, ddof=1) / np.sqrt(repetitions)),
                "learned_augmented_mse_se": float(
                    np.std(learned_array, ddof=1) / np.sqrt(repetitions)
                ),
                "oracle_augmented_mse_se": float(
                    np.std(oracle_array, ddof=1) / np.sqrt(repetitions)
                ),
                "learned_augmented_bias": float(np.mean(learned_signed_errors)),
                "paired_mse_gain": float(np.mean(pdis_array - learned_array)),
                "paired_mse_gain_se": float(
                    np.std(pdis_array - learned_array, ddof=1) / np.sqrt(repetitions)
                ),
                "n_times_pdis_mse": float(n * np.mean(pdis_array)),
                "n_times_learned_augmented_mse": float(n * np.mean(learned_array)),
                "n_times_oracle_augmented_mse": float(n * np.mean(oracle_array)),
                "pdis_to_learned_mse_ratio": float(
                    np.mean(pdis_array) / np.mean(learned_array)
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(DATA_DIR / "stochastic_tabular_crossfit_mse.csv", index=False)

    fig, axis = plt.subplots(figsize=(5.2, 3.45))
    axis.plot(
        frame["n"], frame["n_times_pdis_mse"], marker="o", label="signed PDIS"
    )
    axis.plot(
        frame["n"],
        frame["n_times_learned_augmented_mse"],
        marker="s",
        label="2-fold tabular CA-PDIS",
    )
    axis.plot(
        frame["n"],
        frame["n_times_oracle_augmented_mse"],
        marker="^",
        label="oracle CA-PDIS",
    )
    axis.set_xscale("log")
    axis.set_xlabel("logged episodes $n$")
    axis.set_ylabel(r"$n \times$ mean squared error")
    axis.set_title("Stochastic MDP with a shared logger-null branch")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stochastic_tabular_crossfit_mse.pdf", bbox_inches="tight")
    fig.savefig(
        FIG_DIR / "stochastic_tabular_crossfit_mse.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)
    return frame
