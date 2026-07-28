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
from .efficient import (
    VarianceDecomposition,
    affine_binary_reward_range,
    contrastive_augmented_contributions,
    enumerate_binary_action_strings,
    signed_pdis_contributions,
)


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

        signed_weights = self.signed_weight_matrix(
            actions,
            target_common_weight=target_common_weight,
            target_good_prob=target_good_prob,
            baseline_common_weight=baseline_common_weight,
            baseline_good_prob=baseline_good_prob,
        )
        return signed_pdis_contributions(
            signed_weights, rewards, discount=self.discount
        )

    def component_ratio_matrices(
        self,
        actions: np.ndarray,
        *,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cumulative supported-component ratios for target and baseline.

        The common controller is selected before the logged action sequence and
        is never observed under the logger.  On a logger-supported trajectory,
        the target path mass is therefore ``(1-q)`` times the distinct
        controller's action likelihood.
        """

        actions = np.asarray(actions, dtype=int)
        if actions.ndim != 2 or actions.shape[1] != self.horizon:
            raise ValueError("actions must have shape (n, horizon)")
        if np.any((actions != 0) & (actions != 1)):
            raise ValueError("logged behavior actions must be 0/1")
        for name, probability in (
            ("target_common_weight", target_common_weight),
            ("baseline_common_weight", baseline_common_weight),
            ("target_good_prob", target_good_prob),
            ("baseline_good_prob", baseline_good_prob),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        behavior_likelihood = self._prefix_likelihood(
            actions, self.behavior_good_prob
        )
        target_likelihood = (1.0 - target_common_weight) * self._prefix_likelihood(
            actions, target_good_prob
        )
        baseline_likelihood = (
            1.0 - baseline_common_weight
        ) * self._prefix_likelihood(actions, baseline_good_prob)
        return (
            target_likelihood / behavior_likelihood,
            baseline_likelihood / behavior_likelihood,
        )

    def signed_weight_matrix(
        self,
        actions: np.ndarray,
        *,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
    ) -> np.ndarray:
        """Return signed cumulative path ratios on logged trajectories."""

        target_ratio, baseline_ratio = self.component_ratio_matrices(
            actions,
            target_common_weight=target_common_weight,
            target_good_prob=target_good_prob,
            baseline_common_weight=baseline_common_weight,
            baseline_good_prob=baseline_good_prob,
        )
        return target_ratio - baseline_ratio

    def _controller_values(
        self, good_prob: float, stage_action_means: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pre-action values and after-action Q values for a covered controller."""

        means = np.asarray(stage_action_means, dtype=float)
        if means.shape != (self.horizon, 2):
            raise ValueError("stage_action_means must have shape (horizon, 2)")
        if np.any((means < 0.0) | (means > 1.0)):
            raise ValueError("reward means must lie in [0, 1]")
        if not 0.0 <= good_prob <= 1.0:
            raise ValueError("good_prob must lie in [0, 1]")
        values = np.zeros(self.horizon + 1, dtype=float)
        q_values = np.zeros((self.horizon, 2), dtype=float)
        action_probs = np.array([1.0 - good_prob, good_prob], dtype=float)
        for t in range(self.horizon - 1, -1, -1):
            q_values[t] = means[t] + self.discount * values[t + 1]
            values[t] = float(np.dot(action_probs, q_values[t]))
        return values, q_values

    def contrastive_continuations(
        self,
        actions: np.ndarray,
        *,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
        stage_action_means: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute signed weights and contrastive continuation functions.

        The returned arrays are ``(weights, q, v)``.  ``q[:, t]`` is the
        contrastive post-action continuation and ``v[:, t]`` is its behavior-
        action average before stage ``t``.  The final column ``v[:, H]`` is
        terminal zero.  Supplying estimated stage-action reward means gives a
        fitted nuisance model; omitting them uses the true simulator means.
        """

        action_array = np.asarray(actions, dtype=int)
        if stage_action_means is None:
            means = np.column_stack(
                [
                    np.full(self.horizon, self.reward_bad_mean, dtype=float),
                    np.full(self.horizon, self.reward_good_mean, dtype=float),
                ]
            )
        else:
            means = np.asarray(stage_action_means, dtype=float)
        target_values, target_q = self._controller_values(target_good_prob, means)
        baseline_values, baseline_q = self._controller_values(
            baseline_good_prob, means
        )
        target_ratio, baseline_ratio = self.component_ratio_matrices(
            action_array,
            target_common_weight=target_common_weight,
            target_good_prob=target_good_prob,
            baseline_common_weight=baseline_common_weight,
            baseline_good_prob=baseline_good_prob,
        )
        n = action_array.shape[0]
        target_prior = np.empty_like(target_ratio)
        baseline_prior = np.empty_like(baseline_ratio)
        target_prior[:, 0] = 1.0 - target_common_weight
        baseline_prior[:, 0] = 1.0 - baseline_common_weight
        if self.horizon > 1:
            target_prior[:, 1:] = target_ratio[:, :-1]
            baseline_prior[:, 1:] = baseline_ratio[:, :-1]

        stages = np.arange(self.horizon)[None, :]
        q = target_ratio * target_q[stages, action_array] - baseline_ratio * baseline_q[
            stages, action_array
        ]
        v = np.zeros((n, self.horizon + 1), dtype=float)
        v[:, :-1] = (
            target_prior * target_values[:-1]
            - baseline_prior * baseline_values[:-1]
        )
        return target_ratio - baseline_ratio, q, v

    def contrastive_augmented_samples(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        *,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
        stage_action_means: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return CA-PDIS samples using true or fitted continuation functions."""

        weights, q, v = self.contrastive_continuations(
            actions,
            target_common_weight=target_common_weight,
            target_good_prob=target_good_prob,
            baseline_common_weight=baseline_common_weight,
            baseline_good_prob=baseline_good_prob,
            stage_action_means=stage_action_means,
        )
        return contrastive_augmented_contributions(
            weights, rewards, q, v, discount=self.discount
        )

    def exact_variance_decomposition(
        self,
        *,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
    ) -> VarianceDecomposition:
        """Enumerate all logged action strings and integrate Bernoulli reward noise."""

        actions = enumerate_binary_action_strings(self.horizon)
        behavior_action_prob = np.where(
            actions == 1, self.behavior_good_prob, 1.0 - self.behavior_good_prob
        )
        path_prob = np.prod(behavior_action_prob, axis=1)
        weights, q, v = self.contrastive_continuations(
            actions,
            target_common_weight=target_common_weight,
            target_good_prob=target_good_prob,
            baseline_common_weight=baseline_common_weight,
            baseline_good_prob=baseline_good_prob,
        )
        reward_means = np.where(
            actions == 1, self.reward_good_mean, self.reward_bad_mean
        )
        coefficients = self.discounts * weights
        pdis_offsets = np.zeros(actions.shape[0], dtype=float)
        augmented_offsets = v[:, 0] + np.sum(
            self.discounts * (self.discount * v[:, 1:] - q), axis=1
        )

        def total_variance(offsets: np.ndarray) -> tuple[float, float]:
            conditional_mean = offsets + np.sum(coefficients * reward_means, axis=1)
            conditional_variance = np.sum(
                coefficients**2 * reward_means * (1.0 - reward_means), axis=1
            )
            mean = float(np.dot(path_prob, conditional_mean))
            second = float(
                np.dot(path_prob, conditional_variance + conditional_mean**2)
            )
            return second - mean**2, mean

        pdis_variance, pdis_mean = total_variance(pdis_offsets)
        efficient_variance, augmented_mean = total_variance(augmented_offsets)
        action_noise = np.sum(self.discounts * (q - v[:, :-1]), axis=1)
        action_noise_mean = float(np.dot(path_prob, action_noise))
        action_noise_variance = float(
            np.dot(path_prob, action_noise**2) - action_noise_mean**2
        )
        true_difference = self.exact_difference(
            target_common_weight,
            target_good_prob,
            baseline_common_weight,
            baseline_good_prob,
        )
        if not np.isclose(pdis_mean, true_difference, atol=1e-10):
            raise AssertionError("signed PDIS mean does not equal the true contrast")
        if not np.isclose(augmented_mean, true_difference, atol=1e-10):
            raise AssertionError("augmented mean does not equal the true contrast")
        return VarianceDecomposition(
            pdis_variance=float(pdis_variance),
            efficient_variance=float(efficient_variance),
            action_noise_variance=float(action_noise_variance),
            decomposition_error=float(
                abs(pdis_variance - efficient_variance - action_noise_variance)
            ),
            true_difference=float(true_difference),
        )

    def exact_contribution_range(
        self,
        *,
        estimator: str,
        target_common_weight: float,
        target_good_prob: float,
        baseline_common_weight: float,
        baseline_good_prob: float,
        stage_action_means: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Exact deterministic range over binary actions and rewards."""

        actions = enumerate_binary_action_strings(self.horizon)
        weights, q, v = self.contrastive_continuations(
            actions,
            target_common_weight=target_common_weight,
            target_good_prob=target_good_prob,
            baseline_common_weight=baseline_common_weight,
            baseline_good_prob=baseline_good_prob,
            stage_action_means=stage_action_means,
        )
        coefficients = self.discounts * weights
        if estimator == "pdis":
            offsets = np.zeros(actions.shape[0], dtype=float)
        elif estimator == "augmented":
            offsets = v[:, 0] + np.sum(
                self.discounts * (self.discount * v[:, 1:] - q), axis=1
            )
        else:
            raise ValueError("estimator must be 'pdis' or 'augmented'")
        return affine_binary_reward_range(offsets, coefficients)

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
