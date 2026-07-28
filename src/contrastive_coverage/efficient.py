"""Efficient estimation of an identified sequential policy contrast.

The population contrast is represented by signed per-decision weights.  When
logging propensities are known, randomization of the logged actions is ancillary.
A contrastive Bellman recursion yields the projection that removes this action-
composition noise without ever estimating either policy value separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VarianceDecomposition:
    """Exact per-episode variance decomposition for a finite chain."""

    pdis_variance: float
    efficient_variance: float
    action_noise_variance: float
    decomposition_error: float
    true_difference: float

    @property
    def variance_ratio(self) -> float:
        if self.efficient_variance <= 0.0:
            return float("inf")
        return self.pdis_variance / self.efficient_variance


def _discount_vector(horizon: int, discount: float) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must lie in (0, 1]")
    return discount ** np.arange(horizon, dtype=float)


def signed_pdis_contributions(
    signed_weights: np.ndarray,
    rewards: np.ndarray,
    *,
    discount: float = 1.0,
) -> np.ndarray:
    """Return one signed-PDIS contribution per episode."""

    weights = np.asarray(signed_weights, dtype=float)
    reward_array = np.asarray(rewards, dtype=float)
    if weights.shape != reward_array.shape or weights.ndim != 2:
        raise ValueError("signed_weights and rewards must have shape (n, horizon)")
    discounts = _discount_vector(weights.shape[1], discount)
    return np.sum(discounts * weights * reward_array, axis=1)


def contrastive_augmented_contributions(
    signed_weights: np.ndarray,
    rewards: np.ndarray,
    q_values: np.ndarray,
    v_values: np.ndarray,
    *,
    discount: float = 1.0,
) -> np.ndarray:
    """Return contrastive augmented PDIS (CA-PDIS) contributions.

    ``q_values[:, t]`` is a fitted contrastive continuation ``q_{t+1}`` after
    the logged action. ``v_values[:, t]`` is its logging-policy action average
    before that action, and ``v_values[:, horizon]`` is the terminal value zero.
    If the nuisance functions are fitted on independent data and
    ``v_t(h)=E_beta[q_t(H_t,A_t)|H_t=h]`` is enforced, the sample mean is
    unbiased for the identified contrast for *any* fitted ``q``.
    """

    weights = np.asarray(signed_weights, dtype=float)
    reward_array = np.asarray(rewards, dtype=float)
    q_array = np.asarray(q_values, dtype=float)
    v_array = np.asarray(v_values, dtype=float)
    if weights.shape != reward_array.shape or weights.shape != q_array.shape:
        raise ValueError("weights, rewards, and q_values must align")
    if weights.ndim != 2:
        raise ValueError("inputs must have shape (n, horizon)")
    n, horizon = weights.shape
    if v_array.shape != (n, horizon + 1):
        raise ValueError("v_values must have shape (n, horizon + 1)")
    discounts = _discount_vector(horizon, discount)
    residuals = weights * reward_array + discount * v_array[:, 1:] - q_array
    return v_array[:, 0] + np.sum(discounts * residuals, axis=1)


def action_composition_noise(
    q_values: np.ndarray,
    v_values: np.ndarray,
    *,
    discount: float = 1.0,
) -> np.ndarray:
    """Return the action-randomization component removed by CA-PDIS."""

    q_array = np.asarray(q_values, dtype=float)
    v_array = np.asarray(v_values, dtype=float)
    if q_array.ndim != 2:
        raise ValueError("q_values must have shape (n, horizon)")
    n, horizon = q_array.shape
    if v_array.shape != (n, horizon + 1):
        raise ValueError("v_values must have shape (n, horizon + 1)")
    discounts = _discount_vector(horizon, discount)
    return np.sum(discounts * (q_array - v_array[:, :-1]), axis=1)


def verify_sample_decomposition(
    signed_weights: np.ndarray,
    rewards: np.ndarray,
    q_values: np.ndarray,
    v_values: np.ndarray,
    *,
    discount: float = 1.0,
) -> float:
    """Maximum absolute error in PDIS = CA-PDIS + action noise."""

    pdis = signed_pdis_contributions(signed_weights, rewards, discount=discount)
    augmented = contrastive_augmented_contributions(
        signed_weights, rewards, q_values, v_values, discount=discount
    )
    noise = action_composition_noise(q_values, v_values, discount=discount)
    return float(np.max(np.abs(pdis - augmented - noise)))


def empirical_stage_action_means(
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    n_actions: int = 2,
    prior_mean: float = 0.5,
    prior_count: float = 0.0,
) -> np.ndarray:
    """Estimate a separate reward mean for each time and action.

    A small optional pseudo-count makes the routine defined in very small
    samples. The experiments use ``prior_count=0`` because both logged actions
    have positive probability and training samples are large enough.
    """

    action_array = np.asarray(actions, dtype=int)
    reward_array = np.asarray(rewards, dtype=float)
    if action_array.shape != reward_array.shape or action_array.ndim != 2:
        raise ValueError("actions and rewards must have shape (n, horizon)")
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    if prior_count < 0.0 or not 0.0 <= prior_mean <= 1.0:
        raise ValueError("invalid prior")
    horizon = action_array.shape[1]
    estimates = np.empty((horizon, n_actions), dtype=float)
    for t in range(horizon):
        for action in range(n_actions):
            mask = action_array[:, t] == action
            count = int(np.sum(mask))
            if count == 0 and prior_count == 0.0:
                estimates[t, action] = prior_mean
            else:
                numerator = float(np.sum(reward_array[mask])) + prior_count * prior_mean
                estimates[t, action] = numerator / (count + prior_count)
    return np.clip(estimates, 0.0, 1.0)



def empirical_pooled_action_means(
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    n_actions: int = 2,
    prior_mean: float = 0.5,
    prior_count: float = 0.0,
) -> np.ndarray:
    """Estimate stationary action reward means and repeat them across stages.

    This is a deliberately small nuisance model for the chain experiments.  It
    pools all time points for each action, then broadcasts the estimates to every
    stage.  Cross-fitting this routine supplies a data-adaptive CA-PDIS estimator
    while preserving conditional unbiasedness on each evaluation fold.
    """

    action_array = np.asarray(actions, dtype=int)
    reward_array = np.asarray(rewards, dtype=float)
    if action_array.shape != reward_array.shape or action_array.ndim != 2:
        raise ValueError("actions and rewards must have shape (n, horizon)")
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    if prior_count < 0.0 or not 0.0 <= prior_mean <= 1.0:
        raise ValueError("invalid prior")
    pooled = np.empty(n_actions, dtype=float)
    for action in range(n_actions):
        mask = action_array == action
        count = int(np.sum(mask))
        if count == 0 and prior_count == 0.0:
            pooled[action] = prior_mean
        else:
            numerator = float(np.sum(reward_array[mask])) + prior_count * prior_mean
            pooled[action] = numerator / (count + prior_count)
    pooled = np.clip(pooled, 0.0, 1.0)
    return np.tile(pooled, (action_array.shape[1], 1))

def enumerate_binary_action_strings(horizon: int) -> np.ndarray:
    """Enumerate all binary logged action strings in lexicographic mask order."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    masks = np.arange(2**horizon, dtype=np.uint64)[:, None]
    shifts = np.arange(horizon, dtype=np.uint64)[None, :]
    return ((masks >> shifts) & 1).astype(int)


def affine_binary_reward_range(
    offsets: Iterable[float], coefficients: np.ndarray
) -> tuple[float, float]:
    """Exact global range of offset(a)+sum_t coefficient(a,t) R_t, R_t in {0,1}."""

    offset_array = np.asarray(list(offsets), dtype=float)
    coefficient_array = np.asarray(coefficients, dtype=float)
    if coefficient_array.ndim != 2 or coefficient_array.shape[0] != offset_array.size:
        raise ValueError("offsets and coefficients must align")
    lower_each = offset_array + np.sum(np.minimum(coefficient_array, 0.0), axis=1)
    upper_each = offset_array + np.sum(np.maximum(coefficient_array, 0.0), axis=1)
    return float(np.min(lower_each)), float(np.max(upper_each))
