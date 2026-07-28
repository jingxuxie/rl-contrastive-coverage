"""Small stochastic tabular processes for contrastive off-policy estimation.

The logger never selects a common initial branch that both target policies use
with the same probability.  That branch makes the two absolute policy values
unidentified, but cancels from their contrast.  On the logged branch we fit an
ordinary tabular transition/reward model and combine its two policy value
functions into a *direct* contrastive augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .efficient import contrastive_augmented_contributions, signed_pdis_contributions


@dataclass(frozen=True)
class TabularModel:
    """Finite-horizon reward and transition model."""

    reward_means: np.ndarray  # (H, S, A)
    transitions: np.ndarray  # (H, S, A, S)

    def __post_init__(self) -> None:
        rewards = np.asarray(self.reward_means, dtype=float)
        transitions = np.asarray(self.transitions, dtype=float)
        if rewards.ndim != 3:
            raise ValueError("reward_means must have shape (H, S, A)")
        horizon, n_states, n_actions = rewards.shape
        if transitions.shape != (horizon, n_states, n_actions, n_states):
            raise ValueError("transitions must have shape (H, S, A, S)")
        if np.any((rewards < 0.0) | (rewards > 1.0)):
            raise ValueError("reward means must lie in [0, 1]")
        if np.any(transitions < 0.0):
            raise ValueError("transition probabilities must be nonnegative")
        if not np.allclose(np.sum(transitions, axis=-1), 1.0, atol=1e-10):
            raise ValueError("transition rows must sum to one")

    @property
    def horizon(self) -> int:
        return int(self.reward_means.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.reward_means.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.reward_means.shape[2])


@dataclass(frozen=True)
class SharedBranchTabularProcess:
    """Stochastic covered MDP plus a shared, logger-null initial branch."""

    model: TabularModel
    behavior: np.ndarray  # (H, S, A), conditional on the logged branch
    target: np.ndarray  # (H, S, A), conditional on the logged branch
    baseline: np.ndarray  # (H, S, A), conditional on the logged branch
    initial_distribution: np.ndarray  # (S,)
    common_weight: float = 0.25
    discount: float = 1.0

    def __post_init__(self) -> None:
        shape = (
            self.model.horizon,
            self.model.n_states,
            self.model.n_actions,
        )
        for name, policy in (
            ("behavior", self.behavior),
            ("target", self.target),
            ("baseline", self.baseline),
        ):
            array = np.asarray(policy, dtype=float)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if np.any(array <= 0.0):
                raise ValueError(f"{name} must be strictly positive on logged actions")
            if not np.allclose(np.sum(array, axis=-1), 1.0, atol=1e-10):
                raise ValueError(f"{name} rows must sum to one")
        initial = np.asarray(self.initial_distribution, dtype=float)
        if initial.shape != (self.model.n_states,):
            raise ValueError("initial_distribution has the wrong shape")
        if np.any(initial < 0.0) or not np.isclose(np.sum(initial), 1.0):
            raise ValueError("initial_distribution must be a probability vector")
        if not 0.0 <= self.common_weight < 1.0:
            raise ValueError("common_weight must lie in [0, 1)")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")

    @property
    def horizon(self) -> int:
        return self.model.horizon

    @property
    def discounts(self) -> np.ndarray:
        return self.discount ** np.arange(self.horizon, dtype=float)

    def policy_values(
        self,
        policy: np.ndarray,
        *,
        model: TabularModel | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(V, Q)`` for a conditional policy on the logged branch."""

        working = self.model if model is None else model
        policy_array = np.asarray(policy, dtype=float)
        expected_shape = (working.horizon, working.n_states, working.n_actions)
        if policy_array.shape != expected_shape:
            raise ValueError("policy has the wrong shape")
        values = np.zeros((working.horizon + 1, working.n_states), dtype=float)
        q_values = np.zeros(expected_shape, dtype=float)
        for t in range(working.horizon - 1, -1, -1):
            continuation = np.einsum(
                "sak,k->sa", working.transitions[t], values[t + 1]
            )
            q_values[t] = working.reward_means[t] + self.discount * continuation
            values[t] = np.sum(policy_array[t] * q_values[t], axis=-1)
        return values, q_values

    def true_difference(self) -> float:
        """Value difference; the unknown common branch cancels exactly."""

        target_values, _ = self.policy_values(self.target)
        baseline_values, _ = self.policy_values(self.baseline)
        supported = float(
            np.dot(
                self.initial_distribution,
                target_values[0] - baseline_values[0],
            )
        )
        return (1.0 - self.common_weight) * supported

    def sample_behavior(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample logged states, actions, and Bernoulli rewards."""

        if n <= 0:
            raise ValueError("n must be positive")
        states = np.empty((n, self.horizon + 1), dtype=int)
        actions = np.empty((n, self.horizon), dtype=int)
        rewards = np.empty((n, self.horizon), dtype=float)
        states[:, 0] = rng.choice(
            self.model.n_states, size=n, p=self.initial_distribution
        )
        for t in range(self.horizon):
            state = states[:, t]
            action_probabilities = self.behavior[t, state]
            action_uniforms = rng.random(n)
            actions[:, t] = np.sum(
                action_uniforms[:, None]
                > np.cumsum(action_probabilities, axis=1),
                axis=1,
            )
            action = actions[:, t]
            reward_means = self.model.reward_means[t, state, action]
            rewards[:, t] = rng.binomial(1, reward_means)
            transition_probabilities = self.model.transitions[t, state, action]
            transition_uniforms = rng.random(n)
            states[:, t + 1] = np.sum(
                transition_uniforms[:, None]
                > np.cumsum(transition_probabilities, axis=1),
                axis=1,
            )
        return states, actions, rewards

    def component_ratio_matrices(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cumulative target/logger and baseline/logger path ratios."""

        state_array = np.asarray(states, dtype=int)
        action_array = np.asarray(actions, dtype=int)
        if state_array.shape != (action_array.shape[0], self.horizon + 1):
            raise ValueError("states and actions have incompatible shapes")
        target_ratio = np.empty_like(action_array, dtype=float)
        baseline_ratio = np.empty_like(action_array, dtype=float)
        target_running = np.full(
            action_array.shape[0], 1.0 - self.common_weight, dtype=float
        )
        baseline_running = np.full(
            action_array.shape[0], 1.0 - self.common_weight, dtype=float
        )
        for t in range(self.horizon):
            state = state_array[:, t]
            action = action_array[:, t]
            behavior_prob = self.behavior[t, state, action]
            target_running *= self.target[t, state, action] / behavior_prob
            baseline_running *= self.baseline[t, state, action] / behavior_prob
            target_ratio[:, t] = target_running
            baseline_ratio[:, t] = baseline_running
        return target_ratio, baseline_ratio

    def signed_pdis_samples(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        target_ratio, baseline_ratio = self.component_ratio_matrices(states, actions)
        return signed_pdis_contributions(
            target_ratio - baseline_ratio,
            rewards,
            discount=self.discount,
        )

    def contrastive_continuations(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        *,
        fitted_model: TabularModel,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate model-based direct contrastive continuations on episodes."""

        state_array = np.asarray(states, dtype=int)
        action_array = np.asarray(actions, dtype=int)
        target_values, target_q = self.policy_values(self.target, model=fitted_model)
        baseline_values, baseline_q = self.policy_values(
            self.baseline, model=fitted_model
        )
        target_ratio, baseline_ratio = self.component_ratio_matrices(
            state_array, action_array
        )
        n = action_array.shape[0]
        prior_target = np.empty_like(target_ratio)
        prior_baseline = np.empty_like(baseline_ratio)
        prior_target[:, 0] = 1.0 - self.common_weight
        prior_baseline[:, 0] = 1.0 - self.common_weight
        if self.horizon > 1:
            prior_target[:, 1:] = target_ratio[:, :-1]
            prior_baseline[:, 1:] = baseline_ratio[:, :-1]

        rows = np.arange(n)[:, None]
        times = np.arange(self.horizon)[None, :]
        current_states = state_array[:, :-1]
        q = (
            target_ratio
            * target_q[times, current_states, action_array]
            - baseline_ratio
            * baseline_q[times, current_states, action_array]
        )
        v = np.zeros((n, self.horizon + 1), dtype=float)
        v[:, :-1] = (
            prior_target * target_values[times, current_states]
            - prior_baseline * baseline_values[times, current_states]
        )
        return target_ratio - baseline_ratio, q, v

    def contrastive_augmented_samples(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        *,
        fitted_model: TabularModel,
    ) -> np.ndarray:
        weights, q, v = self.contrastive_continuations(
            states, actions, fitted_model=fitted_model
        )
        return contrastive_augmented_contributions(
            weights, rewards, q, v, discount=self.discount
        )


def fit_tabular_model(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    n_states: int,
    n_actions: int,
    reward_prior_mean: float = 0.5,
    reward_prior_count: float = 1.0,
    transition_prior_count: float = 1.0,
) -> TabularModel:
    """Fit a smoothed finite-horizon tabular model from logged episodes."""

    state_array = np.asarray(states, dtype=int)
    action_array = np.asarray(actions, dtype=int)
    reward_array = np.asarray(rewards, dtype=float)
    if action_array.shape != reward_array.shape or action_array.ndim != 2:
        raise ValueError("actions and rewards must have shape (n, H)")
    if state_array.shape != (action_array.shape[0], action_array.shape[1] + 1):
        raise ValueError("states must have shape (n, H+1)")
    if reward_prior_count < 0.0 or transition_prior_count < 0.0:
        raise ValueError("prior counts must be nonnegative")
    horizon = action_array.shape[1]
    rewards_hat = np.empty((horizon, n_states, n_actions), dtype=float)
    transitions_hat = np.empty(
        (horizon, n_states, n_actions, n_states), dtype=float
    )
    for t in range(horizon):
        for state in range(n_states):
            for action in range(n_actions):
                mask = (state_array[:, t] == state) & (action_array[:, t] == action)
                count = int(np.sum(mask))
                reward_numerator = (
                    float(np.sum(reward_array[mask, t]))
                    + reward_prior_count * reward_prior_mean
                )
                reward_denominator = count + reward_prior_count
                rewards_hat[t, state, action] = (
                    reward_prior_mean
                    if reward_denominator == 0.0
                    else reward_numerator / reward_denominator
                )
                counts = np.bincount(
                    state_array[mask, t + 1], minlength=n_states
                ).astype(float)
                counts += transition_prior_count / n_states
                total = float(np.sum(counts))
                transitions_hat[t, state, action] = (
                    np.full(n_states, 1.0 / n_states)
                    if total == 0.0
                    else counts / total
                )
    return TabularModel(reward_means=rewards_hat, transitions=transitions_hat)


def make_stochastic_validation_process(
    *,
    horizon: int = 6,
    n_states: int = 5,
    common_weight: float = 0.25,
    discount: float = 0.98,
    seed: int = 20260726,
) -> SharedBranchTabularProcess:
    """Create a reproducible stochastic process with a beneficial target policy."""

    if horizon <= 0 or n_states <= 1:
        raise ValueError("invalid dimensions")
    rng = np.random.default_rng(seed)
    n_actions = 2
    reward_means = np.empty((horizon, n_states, n_actions), dtype=float)
    transitions = np.empty((horizon, n_states, n_actions, n_states), dtype=float)
    target = np.empty((horizon, n_states, n_actions), dtype=float)
    baseline = np.empty_like(target)
    behavior = np.full_like(target, 0.5)
    for t in range(horizon):
        for state in range(n_states):
            preferred = (state + t) % 2
            reward_means[t, state, preferred] = 0.72 + 0.04 * np.sin(state + t)
            reward_means[t, state, 1 - preferred] = 0.28 + 0.04 * np.cos(state - t)
            target[t, state] = np.array([0.2, 0.2])
            target[t, state, preferred] = 0.8
            baseline[t, state] = np.array([0.8, 0.8])
            baseline[t, state, preferred] = 0.2
            for action in range(n_actions):
                concentration = np.full(n_states, 0.8)
                favored_state = (state + 1 + action + t) % n_states
                concentration[favored_state] += 3.0
                transitions[t, state, action] = rng.dirichlet(concentration)
    reward_means = np.clip(reward_means, 0.05, 0.95)
    initial = np.zeros(n_states, dtype=float)
    initial[0] = 1.0
    return SharedBranchTabularProcess(
        model=TabularModel(reward_means=reward_means, transitions=transitions),
        behavior=behavior,
        target=target,
        baseline=baseline,
        initial_distribution=initial,
        common_weight=common_weight,
        discount=discount,
    )
