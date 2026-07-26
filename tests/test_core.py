import unittest

import numpy as np

from contrastive_coverage.bandit import (
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


if __name__ == "__main__":
    unittest.main()
