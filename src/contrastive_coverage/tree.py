"""Exact policy-tree calculations for contrastive coverage.

A deterministic history tree is the smallest sequential model in which the
support question is completely transparent: a state at depth t is the action
history up to depth t-1, and the next state appends the chosen action.  Thus a
policy's probability of a state-action prefix is exactly its policy-path
likelihood.  The module enumerates these prefixes, checks contrastive coverage,
and computes sharp population identified sets when off-support reward means are
only known to lie in a bounded interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .bandit import IdentifiedInterval

History = tuple[int, ...]
Policy = Callable[[History], Sequence[float]]


@dataclass(frozen=True)
class PrefixRecord:
    """Policy likelihoods for one action prefix."""

    prefix: History
    behavior_likelihood: float
    target_likelihood: float
    baseline_likelihood: float

    @property
    def contrast(self) -> float:
        return self.target_likelihood - self.baseline_likelihood


def _probabilities(policy: Policy, history: History, n_actions: int) -> np.ndarray:
    probs = np.asarray(policy(history), dtype=float)
    if probs.shape != (n_actions,):
        raise ValueError(
            f"policy returned shape {probs.shape} at history {history}; "
            f"expected {(n_actions,)}"
        )
    if np.any(probs < -1e-12) or not np.isclose(probs.sum(), 1.0, atol=1e-9):
        raise ValueError(f"invalid policy probabilities at history {history}: {probs}")
    return np.clip(probs, 0.0, 1.0)


def prefix_likelihood(policy: Policy, prefix: History, n_actions: int) -> float:
    """Return product_t policy(a_t | a_1,...,a_{t-1})."""

    likelihood = 1.0
    for t, action in enumerate(prefix):
        if action < 0 or action >= n_actions:
            raise ValueError(f"action {action} is outside [0, {n_actions})")
        likelihood *= float(_probabilities(policy, prefix[:t], n_actions)[action])
    return likelihood


def enumerate_prefix_records(
    behavior: Policy,
    target: Policy,
    baseline: Policy,
    *,
    horizon: int,
    n_actions: int,
) -> list[PrefixRecord]:
    """Enumerate all nonempty action prefixes through ``horizon``."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_actions <= 1:
        raise ValueError("n_actions must be at least two")
    records: list[PrefixRecord] = []
    for depth in range(1, horizon + 1):
        for prefix in product(range(n_actions), repeat=depth):
            records.append(
                PrefixRecord(
                    prefix=prefix,
                    behavior_likelihood=prefix_likelihood(
                        behavior, prefix, n_actions
                    ),
                    target_likelihood=prefix_likelihood(target, prefix, n_actions),
                    baseline_likelihood=prefix_likelihood(
                        baseline, prefix, n_actions
                    ),
                )
            )
    return records


def contrastive_coverage_violation(
    records: Iterable[PrefixRecord], *, support_tol: float = 1e-12
) -> float:
    """Largest unmatched policy-path mass on a behavior-null prefix."""

    values = [
        abs(record.contrast)
        for record in records
        if record.behavior_likelihood <= support_tol
    ]
    return max(values, default=0.0)


def individual_coverage_violation(
    records: Iterable[PrefixRecord],
    *,
    which: str,
    support_tol: float = 1e-12,
) -> float:
    """Largest target or baseline path mass on a behavior-null prefix."""

    if which not in {"target", "baseline"}:
        raise ValueError("which must be 'target' or 'baseline'")
    values = []
    for record in records:
        if record.behavior_likelihood <= support_tol:
            likelihood = (
                record.target_likelihood
                if which == "target"
                else record.baseline_likelihood
            )
            values.append(likelihood)
    return max(values, default=0.0)


def exact_tree_difference(
    records: Iterable[PrefixRecord],
    reward_means: Mapping[History, float],
    *,
    discount: float = 1.0,
) -> float:
    """Evaluate the exact target-minus-baseline value on a policy tree."""

    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must lie in (0, 1]")
    total = 0.0
    for record in records:
        if record.prefix not in reward_means:
            raise KeyError(f"missing reward mean for prefix {record.prefix}")
        total += (
            discount ** (len(record.prefix) - 1)
            * record.contrast
            * float(reward_means[record.prefix])
        )
    return float(total)


def sharp_tree_difference_interval(
    records: Iterable[PrefixRecord],
    observed_reward_means: Mapping[History, float],
    *,
    reward_lower: float = 0.0,
    reward_upper: float = 1.0,
    discount: float = 1.0,
    support_tol: float = 1e-12,
) -> IdentifiedInterval:
    """Sharp identified set for a deterministic history tree.

    Reward means on behavior-supported prefixes must be supplied in
    ``observed_reward_means``.  Reward means on behavior-null prefixes are
    unrestricted within ``[reward_lower, reward_upper]``.  Because each prefix
    has its own free reward kernel, assigning an endpoint according to the sign
    of the target-minus-baseline path mass attains both interval endpoints.
    """

    if reward_lower > reward_upper:
        raise ValueError("reward_lower must not exceed reward_upper")
    if not 0.0 < discount <= 1.0:
        raise ValueError("discount must lie in (0, 1]")

    identified = 0.0
    lower_u = 0.0
    upper_u = 0.0
    for record in records:
        coefficient = discount ** (len(record.prefix) - 1) * record.contrast
        if record.behavior_likelihood > support_tol:
            if record.prefix not in observed_reward_means:
                raise KeyError(
                    f"missing observed reward mean for supported prefix {record.prefix}"
                )
            identified += coefficient * float(observed_reward_means[record.prefix])
        else:
            lower_u += min(coefficient * reward_lower, coefficient * reward_upper)
            upper_u += max(coefficient * reward_lower, coefficient * reward_upper)

    return IdentifiedInterval(
        lower=float(identified + lower_u),
        upper=float(identified + upper_u),
        identified=float(identified),
        unidentified_lower=float(lower_u),
        unidentified_upper=float(upper_u),
    )


def unmatched_discounted_mass(
    records: Iterable[PrefixRecord],
    *,
    discount: float = 1.0,
    support_tol: float = 1e-12,
) -> float:
    """Sum of discounted absolute contrast on behavior-null prefixes."""

    return float(
        sum(
            discount ** (len(record.prefix) - 1) * abs(record.contrast)
            for record in records
            if record.behavior_likelihood <= support_tol
        )
    )



def off_support_signatures(
    behavior: Policy,
    policies: Mapping[str, Policy],
    *,
    horizon: int,
    n_actions: int,
    support_tol: float = 1e-12,
) -> dict[str, tuple[float, ...]]:
    """Compute each policy's likelihood signature on behavior-null prefixes.

    The signatures use a common depth-first prefix ordering.  Two policies
    satisfy uniform contrastive coverage exactly when their returned vectors
    agree (up to the caller's numerical tolerance).  The traversal propagates
    all likelihoods jointly, so its cost is linear in the explicit tree size
    times the number of policies.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if n_actions <= 1:
        raise ValueError("n_actions must be at least two")
    if not policies:
        return {}

    names = tuple(policies)
    values: dict[str, list[float]] = {name: [] for name in names}

    def visit(
        history: History,
        behavior_likelihood: float,
        policy_likelihoods: dict[str, float],
    ) -> None:
        if len(history) >= horizon:
            return
        behavior_probs = _probabilities(behavior, history, n_actions)
        policy_probs = {
            name: _probabilities(policies[name], history, n_actions) for name in names
        }
        for action in range(n_actions):
            child = history + (action,)
            child_behavior = behavior_likelihood * float(behavior_probs[action])
            child_policies = {
                name: policy_likelihoods[name] * float(policy_probs[name][action])
                for name in names
            }
            if child_behavior <= support_tol:
                for name in names:
                    values[name].append(child_policies[name])
            visit(child, child_behavior, child_policies)

    visit((), 1.0, {name: 1.0 for name in names})
    return {name: tuple(values[name]) for name in names}


def contrastive_equivalence_classes(
    behavior: Policy,
    policies: Mapping[str, Policy],
    *,
    horizon: int,
    n_actions: int,
    support_tol: float = 1e-12,
    decimals: int = 12,
) -> list[tuple[str, ...]]:
    """Group policies by their off-support path-likelihood signatures.

    Rounding is only a numerical convenience for floating-point policies; the
    mathematical equivalence relation uses exact equality of signatures.
    """

    if decimals < 0:
        raise ValueError("decimals must be nonnegative")
    signatures = off_support_signatures(
        behavior,
        policies,
        horizon=horizon,
        n_actions=n_actions,
        support_tol=support_tol,
    )
    groups: dict[tuple[float, ...], list[str]] = {}
    for name, signature in signatures.items():
        key = tuple(round(value, decimals) for value in signature)
        groups.setdefault(key, []).append(name)
    return [tuple(names) for names in groups.values()]

def table_policy(
    table: Mapping[History, Sequence[float]], *, default: Sequence[float]
) -> Policy:
    """Create a history-dependent policy from a probability table."""

    default_arr = np.asarray(default, dtype=float)

    def policy(history: History) -> Sequence[float]:
        return table.get(history, default_arr)

    return policy
