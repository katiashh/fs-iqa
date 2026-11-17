import math
import unittest

import numpy as np
from scipy.stats import binom, norm

from fs_iqa.statistics import interval_from_scores, quadratic_radius, quantile_indices


class StatisticsTests(unittest.TestCase):
    def test_binomial_tail_probabilities_and_tight_indices(self):
        for n, radius in ((50, 0.0), (1000, 0.1), (2000, 0.2)):
            lo, hi = quantile_indices(n, 0.1, radius, 0.999)
            beta = 0.0005
            self.assertIsNotNone(lo)
            self.assertIsNotNone(hi)
            self.assertLessEqual(binom.cdf(lo, n, norm.cdf(-radius / 0.1)), beta)
            self.assertGreater(binom.cdf(lo + 1, n, norm.cdf(-radius / 0.1)), beta)
            self.assertLessEqual(binom.sf(hi, n, norm.cdf(radius / 0.1)), beta)
            self.assertGreater(binom.sf(hi - 1, n, norm.cdf(radius / 0.1)), beta)

    def test_flatten_and_sort_sample_dimension(self):
        values = np.random.default_rng(0).normal(size=(1000, 1))
        result = interval_from_scores(values, 0.1, 0.1)
        self.assertLess(result["lower_b"], result["median"])
        self.assertLess(result["median"], result["upper_b"])
        self.assertEqual(result, interval_from_scores(values[::-1], 0.1, 0.1))

    def test_joint_coverage_on_known_gaussian_distribution(self):
        # Exact target feature quantiles for the identity scorer with sigma=1.
        n, trials = 200, 4000
        lo, hi = quantile_indices(n, 1.0, 0.5, 0.95)
        samples = np.random.default_rng(123).normal(size=(trials, n))
        samples.sort(axis=1)
        covered = (samples[:, lo] <= -0.5) & (samples[:, hi] >= 0.5)
        self.assertGreater(covered.mean(), 0.94)

    def test_insufficient_samples_abstain(self):
        result = interval_from_scores([0.1, 0.2], 0.1, 1.0)
        self.assertEqual(result["interval_status"], "abstained")
        self.assertIsNone(result["lower_b"])
        self.assertIsNone(result["upper_b"])

    def test_quadratic_equation_and_near_linear_stability(self):
        for j, curvature, target in ((2, 0, 0.1), (2, 5, 0.1), (1e8, 1e-12, 0.1)):
            r = quadratic_radius(j, curvature, target)
            self.assertTrue(math.isclose(j*r + curvature*r*r/2, target, rel_tol=1e-12))
            self.assertGreater(r, 0)

    def test_validation(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                interval_from_scores(values, 0.1, 0.1)
        with self.assertRaises(ValueError):
            quantile_indices(1000, 0, 0.1)
        with self.assertRaises(ValueError):
            quadratic_radius(0, 1, 1)
