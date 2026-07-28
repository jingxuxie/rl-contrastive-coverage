import unittest

import numpy as np

from contrastive_coverage.bandit import (
    augmented_difference_range,
    augmented_difference_samples,
    bandit_difference,
    sharp_difference_interval,
    separate_difference_interval,
)
from contrastive_coverage.sequential import (
    MixtureChain,
    contrastive_mixture_interval,
    separate_mixture_interval,
)


class BanditTheoryTests(unittest.TestCase):
    def test_common_unsupported_mass_cancels(self):
        beta = np.array([0.5, 0.5, 0.0])
        pi = np.array([0.4, 0.1, 0.5])
        pi0 = np.array([0.1, 0.4, 0.5])
        mu = np.array([0.8, 0.2, 0.99])
        interval = sharp_difference_interval(beta, pi, pi0, mu)
        self.assertAlmostEqual(interval.width, 0.0)
        self.assertAlmostEqual(interval.lower, bandit_difference(pi, pi0, mu))
        separate = separate_difference_interval(beta, pi, pi0, mu)
        self.assertAlmostEqual(separate.width, 1.0)

    def test_unmatched_mass_equals_width(self):
        beta = np.array([0.5, 0.5, 0.0])
        pi = np.array([0.48, 0.12, 0.40])
        pi0 = np.array([0.12, 0.53, 0.35])
        mu = np.array([0.8, 0.2, 0.6])
        interval = sharp_difference_interval(beta, pi, pi0, mu)
        self.assertAlmostEqual(interval.width, 0.05)
        self.assertTrue(interval.contains(bandit_difference(pi, pi0, mu)))

    def test_augmented_bandit_contribution_is_model_robust(self):
        beta = np.array([0.5, 0.5, 0.0])
        pi = np.array([0.4, 0.1, 0.5])
        pi0 = np.array([0.1, 0.4, 0.5])
        mu = np.array([0.8, 0.2, 0.9])
        deliberately_wrong = np.array([0.1, 0.9, 0.4])

        # Enumerate the two supported actions and both reward endpoints, then
        # average with the correct Bernoulli probabilities.
        expectation = 0.0
        values = []
        probabilities = []
        for action in (0, 1):
            for reward in (0.0, 1.0):
                sample = augmented_difference_samples(
                    np.array([action]),
                    np.array([reward]),
                    beta,
                    pi,
                    pi0,
                    deliberately_wrong,
                )[0]
                probability = beta[action] * (
                    mu[action] if reward == 1.0 else 1.0 - mu[action]
                )
                expectation += probability * sample
                values.append(sample)
                probabilities.append(probability)

        self.assertAlmostEqual(expectation, bandit_difference(pi, pi0, mu))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(
            augmented_difference_range(
                beta, pi, pi0, deliberately_wrong
            ),
            max(values) - min(values),
        )


class SequentialTheoryTests(unittest.TestCase):
    def test_equal_common_weights_cancel(self):
        env = MixtureChain(hidden_return=3.0)
        identified = env.identified_supported_difference(0.4, 0.8, 0.4, 0.2)
        interval = contrastive_mixture_interval(
            identified,
            0.4,
            0.4,
            common_value_lower=0.0,
            common_value_upper=env.max_return,
        )
        self.assertAlmostEqual(interval.width, 0.0)
        self.assertAlmostEqual(interval.lower, env.exact_difference(0.4, 0.8, 0.4, 0.2))

    def test_mismatch_width(self):
        env = MixtureChain(hidden_return=3.0)
        identified = env.identified_supported_difference(0.45, 0.8, 0.35, 0.2)
        contrastive = contrastive_mixture_interval(
            identified,
            0.45,
            0.35,
            common_value_lower=0.0,
            common_value_upper=env.max_return,
        )
        self.assertAlmostEqual(contrastive.width, 0.10 * env.max_return)
        target_supported = (1.0 - 0.45) * env.supported_value(0.8)
        base_supported = (1.0 - 0.35) * env.supported_value(0.2)
        separate = separate_mixture_interval(
            target_supported,
            base_supported,
            0.45,
            0.35,
            common_value_lower=0.0,
            common_value_upper=env.max_return,
        )
        self.assertGreater(separate.width, contrastive.width)

    def test_contrastive_augmented_sample_decomposition(self):
        from contrastive_coverage.efficient import verify_sample_decomposition

        env = MixtureChain(horizon=6, hidden_return=3.0)
        rng = np.random.default_rng(11)
        actions, rewards = env.sample_behavior(512, rng)
        weights, q_values, v_values = env.contrastive_continuations(
            actions,
            target_common_weight=0.4,
            target_good_prob=0.7,
            baseline_common_weight=0.4,
            baseline_good_prob=0.3,
        )
        error = verify_sample_decomposition(
            weights,
            rewards,
            q_values,
            v_values,
            discount=env.discount,
        )
        self.assertLess(error, 1e-12)

    def test_sequential_efficiency_variance_decomposition(self):
        env = MixtureChain(horizon=7, hidden_return=3.0)
        result = env.exact_variance_decomposition(
            target_common_weight=0.35,
            target_good_prob=0.65,
            baseline_common_weight=0.35,
            baseline_good_prob=0.35,
        )
        self.assertLess(result.decomposition_error, 1e-11)
        self.assertGreater(result.pdis_variance, result.efficient_variance)
        self.assertGreater(result.action_noise_variance, 0.0)

    def test_pooled_action_mean_nuisance(self):
        from contrastive_coverage.efficient import empirical_pooled_action_means

        actions = np.array([[0, 1, 0], [1, 1, 0]], dtype=int)
        rewards = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
        means = empirical_pooled_action_means(actions, rewards)
        self.assertEqual(means.shape, (3, 2))
        np.testing.assert_allclose(means[:, 0], 1.0 / 3.0)
        np.testing.assert_allclose(means[:, 1], 2.0 / 3.0)

    def test_augmented_range_is_exact_on_binary_chain(self):
        from contrastive_coverage.efficient import enumerate_binary_action_strings

        env = MixtureChain(horizon=3, hidden_return=1.5)
        action_strings = enumerate_binary_action_strings(env.horizon)
        reward_strings = enumerate_binary_action_strings(env.horizon).astype(float)
        actions = np.repeat(action_strings, reward_strings.shape[0], axis=0)
        rewards = np.tile(reward_strings, (action_strings.shape[0], 1))
        samples = env.contrastive_augmented_samples(
            actions,
            rewards,
            target_common_weight=0.35,
            target_good_prob=0.65,
            baseline_common_weight=0.35,
            baseline_good_prob=0.35,
        )
        lower, upper = env.exact_contribution_range(
            estimator="augmented",
            target_common_weight=0.35,
            target_good_prob=0.65,
            baseline_common_weight=0.35,
            baseline_good_prob=0.35,
        )
        self.assertAlmostEqual(lower, float(np.min(samples)), places=12)
        self.assertAlmostEqual(upper, float(np.max(samples)), places=12)

    def test_augmented_expectation_is_robust_to_misspecified_continuation(self):
        env = MixtureChain(horizon=5, hidden_return=2.0)
        from contrastive_coverage.efficient import enumerate_binary_action_strings

        actions = enumerate_binary_action_strings(env.horizon)
        behavior_prob = np.prod(
            np.where(
                actions == 1,
                env.behavior_good_prob,
                1.0 - env.behavior_good_prob,
            ),
            axis=1,
        )
        deliberately_wrong_means = np.tile(np.array([0.1, 0.9]), (env.horizon, 1))
        weights, q_values, v_values = env.contrastive_continuations(
            actions,
            target_common_weight=0.4,
            target_good_prob=0.7,
            baseline_common_weight=0.4,
            baseline_good_prob=0.3,
            stage_action_means=deliberately_wrong_means,
        )
        true_means = np.where(
            actions == 1, env.reward_good_mean, env.reward_bad_mean
        )
        offsets = v_values[:, 0] + np.sum(
            env.discounts * (env.discount * v_values[:, 1:] - q_values), axis=1
        )
        conditional_means = offsets + np.sum(
            env.discounts * weights * true_means, axis=1
        )
        exact_mean = float(np.dot(behavior_prob, conditional_means))
        self.assertAlmostEqual(
            exact_mean,
            env.exact_difference(0.4, 0.7, 0.4, 0.3),
            places=12,
        )


class TreeTheoryTests(unittest.TestCase):
    def test_shared_unsupported_subtree_satisfies_contrastive_not_individual_coverage(self):
        from contrastive_coverage.tree import (
            contrastive_coverage_violation,
            enumerate_prefix_records,
            individual_coverage_violation,
            table_policy,
        )

        behavior = table_policy({(): [0.5, 0.5, 0.0]}, default=[1 / 3, 1 / 3, 1 / 3])
        target = table_policy({(): [0.4, 0.1, 0.5]}, default=[0.2, 0.3, 0.5])
        baseline = table_policy({(): [0.1, 0.4, 0.5]}, default=[0.2, 0.3, 0.5])
        records = enumerate_prefix_records(
            behavior, target, baseline, horizon=3, n_actions=3
        )
        self.assertAlmostEqual(contrastive_coverage_violation(records), 0.0)
        self.assertGreater(individual_coverage_violation(records, which="target"), 0.0)
        self.assertGreater(individual_coverage_violation(records, which="baseline"), 0.0)

    def test_policy_library_partitions_by_off_support_signature(self):
        from contrastive_coverage.tree import (
            contrastive_equivalence_classes,
            off_support_signatures,
            table_policy,
        )

        behavior = table_policy({(): [0.5, 0.5, 0.0]}, default=[1 / 3, 1 / 3, 1 / 3])
        target = table_policy({(): [0.4, 0.1, 0.5]}, default=[0.2, 0.3, 0.5])
        baseline = table_policy({(): [0.1, 0.4, 0.5]}, default=[0.2, 0.3, 0.5])
        covered = table_policy({(): [0.7, 0.3, 0.0]}, default=[1 / 3, 1 / 3, 1 / 3])
        different = table_policy({(): [0.2, 0.4, 0.4]}, default=[0.2, 0.3, 0.5])
        policies = {
            "behavior": behavior,
            "target": target,
            "baseline": baseline,
            "covered": covered,
            "different": different,
        }
        signatures = off_support_signatures(
            behavior, policies, horizon=3, n_actions=3
        )
        self.assertEqual(signatures["target"], signatures["baseline"])
        self.assertEqual(signatures["behavior"], signatures["covered"])
        self.assertNotEqual(signatures["target"], signatures["different"])
        classes = {frozenset(group) for group in contrastive_equivalence_classes(
            behavior, policies, horizon=3, n_actions=3
        )}
        self.assertIn(frozenset({"target", "baseline"}), classes)
        self.assertIn(frozenset({"behavior", "covered"}), classes)
        self.assertIn(frozenset({"different"}), classes)

    def test_tree_width_identity(self):
        from contrastive_coverage.tree import (
            enumerate_prefix_records,
            sharp_tree_difference_interval,
            table_policy,
            unmatched_discounted_mass,
        )

        behavior = table_policy({(): [0.5, 0.5, 0.0]}, default=[0.5, 0.5, 0.0])
        target = table_policy({(): [0.36, 0.09, 0.55]}, default=[0.2, 0.3, 0.5])
        baseline = table_policy({(): [0.11, 0.44, 0.45]}, default=[0.2, 0.3, 0.5])
        records = enumerate_prefix_records(
            behavior, target, baseline, horizon=3, n_actions=3
        )
        observed = {
            record.prefix: 0.5
            for record in records
            if record.behavior_likelihood > 1e-12
        }
        interval = sharp_tree_difference_interval(records, observed, discount=0.9)
        self.assertAlmostEqual(
            interval.width, unmatched_discounted_mass(records, discount=0.9)
        )


class BanditEfficiencyTests(unittest.TestCase):
    def test_efficiency_bound_is_below_signed_ips_variance(self) -> None:
        beta = np.array([0.5, 0.5, 0.0])
        target = np.array([0.44, 0.11, 0.45])
        baseline = np.array([0.11, 0.44, 0.45])
        mu = np.array([0.8, 0.2, 0.55])
        from contrastive_coverage.bandit import (
            bandit_efficiency_bound,
            signed_ips_population_variance,
        )

        efficient = bandit_efficiency_bound(beta, target, baseline, mu)
        ips = signed_ips_population_variance(beta, target, baseline, mu)
        self.assertLess(efficient, ips)
        self.assertAlmostEqual(ips / efficient, 2.5625)


class StochasticTabularTests(unittest.TestCase):
    def test_shared_hidden_branch_cancels_in_stochastic_mdp(self):
        from contrastive_coverage.tabular import make_stochastic_validation_process

        process = make_stochastic_validation_process(horizon=4, n_states=4)
        target_values, _ = process.policy_values(process.target)
        baseline_values, _ = process.policy_values(process.baseline)
        supported = float(
            np.dot(
                process.initial_distribution,
                target_values[0] - baseline_values[0],
            )
        )
        self.assertAlmostEqual(
            process.true_difference(),
            (1.0 - process.common_weight) * supported,
            places=12,
        )
        self.assertGreater(process.true_difference(), 0.0)

    def test_tabular_augmented_sample_decomposition(self):
        from contrastive_coverage.efficient import verify_sample_decomposition
        from contrastive_coverage.tabular import make_stochastic_validation_process

        process = make_stochastic_validation_process(horizon=4, n_states=4)
        rng = np.random.default_rng(23)
        states, actions, rewards = process.sample_behavior(2048, rng)
        weights, q_values, v_values = process.contrastive_continuations(
            states, actions, fitted_model=process.model
        )
        error = verify_sample_decomposition(
            weights, rewards, q_values, v_values, discount=process.discount
        )
        self.assertLess(error, 1e-11)

    def test_tabular_crossfit_model_is_nearly_unbiased(self):
        from contrastive_coverage.tabular import (
            fit_tabular_model,
            make_stochastic_validation_process,
        )

        process = make_stochastic_validation_process(horizon=4, n_states=4)
        rng = np.random.default_rng(29)
        train_states, train_actions, train_rewards = process.sample_behavior(3000, rng)
        fitted = fit_tabular_model(
            train_states,
            train_actions,
            train_rewards,
            n_states=process.model.n_states,
            n_actions=process.model.n_actions,
        )
        states, actions, rewards = process.sample_behavior(100000, rng)
        estimate = float(
            np.mean(
                process.contrastive_augmented_samples(
                    states, actions, rewards, fitted_model=fitted
                )
            )
        )
        self.assertAlmostEqual(estimate, process.true_difference(), delta=0.02)


if __name__ == "__main__":
    unittest.main()
