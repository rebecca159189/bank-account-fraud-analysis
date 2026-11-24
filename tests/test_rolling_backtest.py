"""Regression tests for the bootstrap's tie-aware weighted AP calculation."""
import unittest
import numpy as np
from sklearn.metrics import average_precision_score
from src.rolling_backtest import weighted_ap_setup


class WeightedAPTests(unittest.TestCase):
    def test_ties_and_unobserved_high_score_group(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        s = np.array([.9, .9, .8, .8, .4, .4, .1, .1])
        weights = np.array([0, 0, 4, 2, 1, 3, 0, 1])
        actual = weighted_ap_setup(y, s)(weights)
        expanded = np.repeat(np.arange(len(y)), weights)
        self.assertAlmostEqual(actual, average_precision_score(y[expanded], s[expanded]), places=14)

    def test_many_random_bootstrap_multiplicities(self):
        rng = np.random.default_rng(42)
        y = np.r_[np.zeros(950, dtype=int), np.ones(50, dtype=int)]
        s = rng.integers(0, 15, len(y))/15
        calculate = weighted_ap_setup(y, s)
        for _ in range(20):
            w = rng.integers(0, 5, len(y))
            self.assertAlmostEqual(calculate(w), average_precision_score(y, s, sample_weight=w), places=13)

    def test_constant_scores_equal_weighted_prevalence(self):
        y = np.array([0, 0, 0, 1, 1])
        w = np.array([7, 1, 2, 3, 1])
        self.assertAlmostEqual(weighted_ap_setup(y, np.ones(5))(w), 4/14)


if __name__ == '__main__':
    unittest.main()
