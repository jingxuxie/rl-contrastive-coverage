"""A small sequential shared-component model for contrastive coverage.

Two policies are episode-level mixtures of a common, potentially unsupported
controller and distinct covered controllers.  Equal mixture weights make the
unknown common value cancel exactly.  Unequal weights leave only the unmatched
coefficient times the common controller's bounded value unidentified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .bandit import IdentifiedInterval, empirical_bernstein_radius


@dataclass(frozen=True)
class MixtureChain:
    """Finite-horizon chain with a hidden common branch at the first action."""

    horizon: int = 5
    behavior_good_prob: float = 0.5
    reward_good_mean: float = 0.75
    reward_bad_mean: float = 0.25
    discount: float = 1.0
    hidden_return: float = 2.5

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        for name, value in (
            ("behavior_good_prob", self.behavior_good_prob),
            ("reward_good_mean", self.reward_good_mean),
            ("reward_bad_mean", self.reward_bad_mean),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")
        if not 0.0 <= self.hidden_return <= self.max_return:
            raise ValueError("hidden_return must lie in [0, max_return]")

    @property
    def discounts(self) -> np.ndarray:
        return self.discount ** np.arange(self.horizon, dtype=float)

    @property
    def max_return(self) -> float:
        return float(np.sum(self.discount ** np.arange(self.horizon)))

    def supported_value(self, good_action_prob: float) -> float:
        if not 0.0 <= good_action_prob <= 1.0:
            raise ValueError("good_action_prob must lie in [0, 1]")
        mean = (
            good_action_prob * self.reward_good_mean
            + (1.0 - good_action_prob) * self.reward_bad_mean
        )
        return float(mean * np.sum(self.discounts))

    def mixture_value(self, common_weight: float, good_action_prob: float) -> float:
        if not 0.0 <= common_weight <= 1.0:
            raise ValueError("common_weight must lie in [0, 1]")
        return float(
            common_weight * self.hidden_return
            + (1.0 - common_weight) * self.supported_value(good_action_prob)
        )

    def exact_difference(
        self,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
    ) -> float:
        return self.mixture_value(
            target_common_weight, target_good_prob
        ) - self.mixture_value(baseline_common_weight, baseline_good_prob)

    def identified_supported_difference(
        self,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
    ) -> float:
        return float(
            (1.0 - target_common_weight)
            * self.supported_value(target_good_prob)
            - (1.0 - baseline_common_weight)
            * self.supported_value(baseline_good_prob)
        )

    def sample_behavior(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            raise ValueError("n must be positive")
        actions = rng.binomial(
            1, self.behavior_good_prob, size=(n, self.horizon)
        ).astype(int)
        means = np.where(
            actions == 1, self.reward_good_mean, self.reward_bad_mean
        )
        rewards = rng.binomial(1, means).astype(float)
        return actions, rewards

    @staticmethod
    def _prefix_likelihood(actions: np.ndarray, good_prob: float) -> np.ndarray:
        if not 0.0 <= good_prob <= 1.0:
            raise ValueError("good_prob must lie in [0, 1]")
        action_prob = np.where(actions == 1, good_prob, 1.0 - good_prob)
        return np.cumprod(action_prob, axis=1)

    def signed_pdis_samples(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        *,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=int)
        rewards = np.asarray(rewards, dtype=float)
        if actions.shape != rewards.shape:
            raise ValueError("actions and rewards must have matching shapes")
        if actions.ndim != 2 or actions.shape[1] != self.horizon:
            raise ValueError("actions must have shape (n, horizon)")
        if np.any((actions != 0) & (actions != 1)):
            raise ValueError("logged behavior actions must be 0/1")
        for q in (target_common_weight, baseline_common_weight):
            if not 0.0 <= q <= 1.0:
                raise ValueError("common weights must lie in [0, 1]")

        l_target = (1.0 - target_common_weight) * self._prefix_likelihood(
            actions, target_good_prob
        )
        l_baseline = (1.0 - baseline_common_weight) * self._prefix_likelihood(
            actions, baseline_good_prob
        )
        l_behavior = self._prefix_likelihood(actions, self.behavior_good_prob)
        signed_weights = (l_target - l_baseline) / l_behavior
        return np.sum(self.discounts * signed_weights * rewards, axis=1)

    def deterministic_sample_bound(
        self,
        *,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
    ) -> float:
        """Enumerate the 2^H logged action strings to bound |PDIS sample|."""

        strings = np.array(
            [
                [(mask >> j) & 1 for j in range(self.horizon)]
                for mask in range(2**self.horizon)
            ],
            dtype=int,
        )
        ones = np.ones_like(strings, dtype=float)
        samples = self.signed_pdis_samples(
            strings,
            ones,
            target_common_weight=target_common_weight,
            target_good_prob=target_good_prob,
            baseline_common_weight=baseline_common_weight,
            baseline_good_prob=baseline_good_prob,
        )
        return float(np.max(np.abs(samples)))


def contrastive_mixture_interval(
    identified_supported_difference: float,
    target_common_weight: float,
    baseline_common_weight: float,
    *,
    common_value_lower: float,
    common_value_upper: float,
) -> IdentifiedInterval:
    """Sharp interval when the same unsupported component appears in both policies."""

    if common_value_lower > common_value_upper:
        raise ValueError("invalid common value range")
    coefficient = target_common_weight - baseline_common_weight
    lower_u = min(
        coefficient * common_value_lower, coefficient * common_value_upper
    )
    upper_u = max(
        coefficient * common_value_lower, coefficient * common_value_upper
    )
    return IdentifiedInterval(
        lower=identified_supported_difference + lower_u,
        upper=identified_supported_difference + upper_u,
        identified=identified_supported_difference,
        unidentified_lower=lower_u,
        unidentified_upper=upper_u,
    )


def separate_mixture_interval(
    target_supported_contribution: float,
    baseline_supported_contribution: float,
    target_common_weight: float,
    baseline_common_weight: float,
    *,
    common_value_lower: float,
    common_value_upper: float,
) -> IdentifiedInterval:
    """Conservative interval from bounding policy values independently."""

    target_lo = target_supported_contribution + target_common_weight * common_value_lower
    target_hi = target_supported_contribution + target_common_weight * common_value_upper
    base_lo = baseline_supported_contribution + baseline_common_weight * common_value_lower
    base_hi = baseline_supported_contribution + baseline_common_weight * common_value_upper
    identified = target_supported_contribution - baseline_supported_contribution
    lower = target_lo - base_hi
    upper = target_hi - base_lo
    return IdentifiedInterval(
        lower=lower,
        upper=upper,
        identified=identified,
        unidentified_lower=lower - identified,
        unidentified_upper=upper - identified,
    )


def sequential_confidence_interval(
    samples: Iterable[float],
    *,
    target_common_weight: float,
    baseline_common_weight: float,
    common_value_lower: float,
    common_value_upper: float,
    sample_abs_bound: float,
    delta: float = 0.05,
) -> IdentifiedInterval:
    """Empirical-Bernstein CI plus shared-component partial identification."""

    x = np.asarray(list(samples), dtype=float)
    estimate = float(np.mean(x))
    radius = empirical_bernstein_radius(
        x, delta=delta, range_bound=2.0 * sample_abs_bound
    )
    population = contrastive_mixture_interval(
        estimate,
        target_common_weight,
        baseline_common_weight,
        common_value_lower=common_value_lower,
        common_value_upper=common_value_upper,
    )
    return IdentifiedInterval(
        lower=population.lower - radius,
        upper=population.upper + radius,
        identified=estimate,
        unidentified_lower=population.unidentified_lower,
        unidentified_upper=population.unidentified_upper,
    )
