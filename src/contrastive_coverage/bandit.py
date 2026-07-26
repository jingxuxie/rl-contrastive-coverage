"""Sharp identification and finite-sample intervals for bandit policy contrasts.

The key object is the signed policy difference d = pi - pi0.  When the
logging policy has zero probability on an action, only unmatched signed mass
on that action contributes to non-identification.  Common unsupported mass
cancels exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class IdentifiedInterval:
    """An interval together with its identified and unidentified pieces."""

    lower: float
    upper: float
    identified: float
    unidentified_lower: float
    unidentified_upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float, atol: float = 1e-12) -> bool:
        return self.lower - atol <= value <= self.upper + atol


def _as_probability_vector(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(arr < -1e-12):
        raise ValueError(f"{name} contains a negative probability")
    if not np.isclose(arr.sum(), 1.0, atol=1e-9):
        raise ValueError(f"{name} must sum to one; got {arr.sum():.12g}")
    return np.clip(arr, 0.0, 1.0)


def _validate_inputs(
    behavior: Iterable[float],
    target: Iterable[float],
    baseline: Iterable[float],
    reward_means: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta = _as_probability_vector(behavior, "behavior")
    pi = _as_probability_vector(target, "target")
    pi0 = _as_probability_vector(baseline, "baseline")
    mu = np.asarray(reward_means, dtype=float)
    if not (beta.shape == pi.shape == pi0.shape == mu.shape):
        raise ValueError("behavior, target, baseline, and reward_means must align")
    return beta, pi, pi0, mu


def bandit_difference(
    target: Iterable[float], baseline: Iterable[float], reward_means: Iterable[float]
) -> float:
    """Return the exact policy-value difference for a finite-action bandit."""

    pi = _as_probability_vector(target, "target")
    pi0 = _as_probability_vector(baseline, "baseline")
    mu = np.asarray(reward_means, dtype=float)
    if not (pi.shape == pi0.shape == mu.shape):
        raise ValueError("target, baseline, and reward_means must align")
    return float(np.dot(pi - pi0, mu))


def sharp_difference_interval(
    behavior: Iterable[float],
    target: Iterable[float],
    baseline: Iterable[float],
    reward_means: Iterable[float],
    *,
    reward_lower: float = 0.0,
    reward_upper: float = 1.0,
    support_tol: float = 1e-12,
) -> IdentifiedInterval:
    """Compute the sharp population identified set for V(target)-V(baseline).

    ``reward_means`` is used only on actions supported by ``behavior``.  Values
    supplied on unsupported actions are ignored.  With bounded rewards, the
    lower endpoint assigns the worst admissible mean independently to each
    unmatched signed action mass; the upper endpoint does the converse.
    """

    if reward_lower > reward_upper:
        raise ValueError("reward_lower must not exceed reward_upper")
    beta, pi, pi0, mu = _validate_inputs(behavior, target, baseline, reward_means)
    supported = beta > support_tol
    d = pi - pi0
    identified = float(np.dot(d[supported], mu[supported]))
    d_u = d[~supported]
    lower_u = float(
        np.sum(np.where(d_u >= 0.0, d_u * reward_lower, d_u * reward_upper))
    )
    upper_u = float(
        np.sum(np.where(d_u >= 0.0, d_u * reward_upper, d_u * reward_lower))
    )
    return IdentifiedInterval(
        lower=identified + lower_u,
        upper=identified + upper_u,
        identified=identified,
        unidentified_lower=lower_u,
        unidentified_upper=upper_u,
    )


def _single_policy_interval(
    behavior: np.ndarray,
    policy: np.ndarray,
    reward_means: np.ndarray,
    reward_lower: float,
    reward_upper: float,
    support_tol: float,
) -> tuple[float, float]:
    supported = behavior > support_tol
    identified = float(np.dot(policy[supported], reward_means[supported]))
    unsupported_mass = float(policy[~supported].sum())
    return (
        identified + unsupported_mass * reward_lower,
        identified + unsupported_mass * reward_upper,
    )


def separate_difference_interval(
    behavior: Iterable[float],
    target: Iterable[float],
    baseline: Iterable[float],
    reward_means: Iterable[float],
    *,
    reward_lower: float = 0.0,
    reward_upper: float = 1.0,
    support_tol: float = 1e-12,
) -> IdentifiedInterval:
    """Difference interval obtained by bounding the two values separately.

    This deliberately ignores that the two policies face the same unknown
    reward function.  It is a useful conservative baseline: common unsupported
    mass is charged twice instead of canceled.
    """

    beta, pi, pi0, mu = _validate_inputs(behavior, target, baseline, reward_means)
    lo_pi, hi_pi = _single_policy_interval(
        beta, pi, mu, reward_lower, reward_upper, support_tol
    )
    lo_0, hi_0 = _single_policy_interval(
        beta, pi0, mu, reward_lower, reward_upper, support_tol
    )
    supported = beta > support_tol
    identified = float(np.dot((pi - pi0)[supported], mu[supported]))
    lower = lo_pi - hi_0
    upper = hi_pi - lo_0
    return IdentifiedInterval(
        lower=lower,
        upper=upper,
        identified=identified,
        unidentified_lower=lower - identified,
        unidentified_upper=upper - identified,
    )


def sample_bandit(
    behavior: Iterable[float],
    reward_means: Iterable[float],
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample actions and Bernoulli rewards from a finite-action bandit."""

    if n <= 0:
        raise ValueError("n must be positive")
    beta = _as_probability_vector(behavior, "behavior")
    mu = np.asarray(reward_means, dtype=float)
    if beta.shape != mu.shape:
        raise ValueError("behavior and reward_means must align")
    if np.any((mu < 0.0) | (mu > 1.0)):
        raise ValueError("Bernoulli reward means must lie in [0, 1]")
    actions = rng.choice(beta.size, size=n, p=beta)
    rewards = rng.binomial(1, mu[actions]).astype(float)
    return actions, rewards


def signed_ips_samples(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior: Iterable[float],
    target: Iterable[float],
    baseline: Iterable[float],
) -> np.ndarray:
    """Trajectory contributions for the identified supported contrast."""

    beta = _as_probability_vector(behavior, "behavior")
    pi = _as_probability_vector(target, "target")
    pi0 = _as_probability_vector(baseline, "baseline")
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have the same shape")
    if np.any(beta[actions] <= 0.0):
        raise ValueError("observed action has zero behavior probability")
    return (pi[actions] - pi0[actions]) * rewards / beta[actions]


def empirical_bernstein_radius(
    samples: Iterable[float],
    *,
    delta: float,
    range_bound: float,
) -> float:
    """Two-sided empirical-Bernstein radius.

    The implementation uses the standard sample-variance form of the
    Maurer--Pontil bound. ``range_bound`` is a deterministic upper bound on
    max(samples)-min(samples), not a data-dependent observed range.
    """

    x = np.asarray(list(samples), dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("at least two scalar samples are required")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if range_bound <= 0.0:
        raise ValueError("range_bound must be positive")
    variance = float(np.var(x, ddof=1))
    log_term = float(np.log(4.0 / delta))
    return float(
        np.sqrt(2.0 * variance * log_term / x.size)
        + 7.0 * range_bound * log_term / (3.0 * (x.size - 1))
    )


def partial_difference_confidence_interval(
    actions: np.ndarray,
    rewards: np.ndarray,
    behavior: Iterable[float],
    target: Iterable[float],
    baseline: Iterable[float],
    *,
    delta: float = 0.05,
    reward_lower: float = 0.0,
    reward_upper: float = 1.0,
    support_tol: float = 1e-12,
) -> IdentifiedInterval:
    """Finite-sample confidence region plus sharp off-support uncertainty."""

    if reward_lower != 0.0 or reward_upper != 1.0:
        raise NotImplementedError(
            "finite-sample Bernoulli implementation currently assumes [0, 1] rewards"
        )
    beta = _as_probability_vector(behavior, "behavior")
    pi = _as_probability_vector(target, "target")
    pi0 = _as_probability_vector(baseline, "baseline")
    if not (beta.shape == pi.shape == pi0.shape):
        raise ValueError("behavior, target, and baseline must align")
    y = signed_ips_samples(actions, rewards, beta, pi, pi0)
    estimate = float(np.mean(y))
    supported = beta > support_tol
    d = pi - pi0
    if np.any(supported):
        abs_bound = float(np.max(np.abs(d[supported]) / beta[supported]))
    else:
        abs_bound = 0.0
    if abs_bound == 0.0:
        radius = 0.0
    else:
        radius = empirical_bernstein_radius(
            y, delta=delta, range_bound=2.0 * abs_bound
        )
    d_u = d[~supported]
    lower_u = float(np.sum(np.minimum(d_u, 0.0)))
    upper_u = float(np.sum(np.maximum(d_u, 0.0)))
    return IdentifiedInterval(
        lower=estimate - radius + lower_u,
        upper=estimate + radius + upper_u,
        identified=estimate,
        unidentified_lower=lower_u,
        unidentified_upper=upper_u,
    )


def signed_ips_population_variance(
    behavior: Iterable[float],
    target: Iterable[float],
    baseline: Iterable[float],
    reward_means: Iterable[float],
    *,
    support_tol: float = 1e-12,
) -> float:
    """Exact one-sample variance of signed IPS for Bernoulli rewards.

    Requires contrastive coverage on unsupported actions: target-baseline mass
    must be zero wherever the behavior probability is zero.
    """

    beta, pi, pi0, mu = _validate_inputs(behavior, target, baseline, reward_means)
    d = pi - pi0
    unsupported = beta <= support_tol
    if np.any(np.abs(d[unsupported]) > support_tol):
        raise ValueError("signed IPS variance requires contrastive coverage")
    supported = ~unsupported
    delta = float(np.dot(d[supported], mu[supported]))
    second = float(np.sum((d[supported] ** 2) * mu[supported] / beta[supported]))
    return max(0.0, second - delta**2)


def bandit_efficiency_bound(
    behavior: Iterable[float],
    target: Iterable[float],
    baseline: Iterable[float],
    reward_means: Iterable[float],
    *,
    support_tol: float = 1e-12,
) -> float:
    """Per-observation semiparametric efficiency bound for Bernoulli rewards."""

    beta, pi, pi0, mu = _validate_inputs(behavior, target, baseline, reward_means)
    d = pi - pi0
    unsupported = beta <= support_tol
    if np.any(np.abs(d[unsupported]) > support_tol):
        raise ValueError("efficiency bound requires contrastive coverage")
    supported = ~unsupported
    return float(
        np.sum(
            (d[supported] ** 2)
            * mu[supported]
            * (1.0 - mu[supported])
            / beta[supported]
        )
    )


def stratified_difference_estimate(
    actions: np.ndarray,
    rewards: np.ndarray,
    target: Iterable[float],
    baseline: Iterable[float],
    *,
    n_actions: int | None = None,
    support_tol: float = 1e-12,
) -> float:
    """Efficient finite-bandit plug-in estimate using within-action means.

    The estimate is defined when every action with nonzero target-baseline
    contrast appears at least once. Unsupported actions are harmless when their
    contrast is zero.
    """

    pi = _as_probability_vector(target, "target")
    pi0 = _as_probability_vector(baseline, "baseline")
    actions = np.asarray(actions, dtype=int)
    rewards = np.asarray(rewards, dtype=float)
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have the same shape")
    if n_actions is None:
        n_actions = pi.size
    if n_actions != pi.size or pi0.size != n_actions:
        raise ValueError("n_actions must match policy dimensions")
    d = pi - pi0
    estimate = 0.0
    for action in range(n_actions):
        if abs(d[action]) <= support_tol:
            continue
        mask = actions == action
        if not np.any(mask):
            raise ValueError(
                f"action {action} has nonzero contrast but was not observed"
            )
        estimate += d[action] * float(np.mean(rewards[mask]))
    return float(estimate)
