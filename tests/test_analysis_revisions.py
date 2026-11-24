"""Regression checks for drift measurement and validation-only rule selection."""
import unittest

import numpy as np
import pandas as pd

from src.eda import calculate_psi, fit_psi_bins
from src.extensions import select_rule_size
from src.rolling_verify import (independent_average_precision, independent_fpr_threshold,
                                independent_top_k, independent_binary_metrics, independent_paired_points,
                                independent_roc_auc)


class PsiRegressionTests(unittest.TestCase):
    def test_binary_shift_is_not_collapsed_into_one_bin(self):
        reference = pd.Series([0]*83+[1]*17)
        current = pd.Series([0]*17+[1]*83, dtype=float)
        value, bins = calculate_psi(reference, current, False)
        p = np.array([83.5, 17.5, .5, .5])/102
        q = np.array([17.5, 83.5, .5, .5])/102
        expected = float(np.sum((q-p)*np.log(q/p)))
        self.assertEqual(bins, 4)
        self.assertAlmostEqual(value, expected, places=14)
        self.assertGreater(value, 1)

    def test_training_defines_discrete_mode_even_with_many_new_current_values(self):
        reference = pd.Series([0, 1, 2]*30)
        current = pd.Series(np.arange(90))
        _, bins = calculate_psi(reference, current, False)
        self.assertEqual(fit_psi_bins(reference)['kind'], 'discrete')
        self.assertEqual(bins, 5)  # three reference values, unseen, missing

    def test_new_category_variety_cannot_change_the_bin_definition(self):
        reference = pd.Series(['A']*9+['B'])
        all_c = pd.Series(['C']*10)
        mixed_unseen = pd.Series(['C']*5+['D']*5)
        value_a, bins_a = calculate_psi(reference, all_c, True)
        value_b, bins_b = calculate_psi(reference, mixed_unseen, True)
        self.assertEqual(bins_a, 4)
        self.assertEqual(bins_a, bins_b)
        self.assertEqual(value_a, value_b)

    def test_missing_is_distinct_from_a_real_category_or_unseen_value(self):
        reference = pd.Series(['<MISSING>']*50+[None]*50)
        unchanged, _ = calculate_psi(reference, reference.copy(), True)
        shifted, bins = calculate_psi(reference, pd.Series(['<MISSING>']*100), True)
        self.assertEqual(unchanged, 0)
        self.assertEqual(bins, 3)
        self.assertGreater(shifted, 1)

    def test_continuous_deciles_are_fitted_to_training_only(self):
        reference = pd.Series(np.arange(1000, dtype=float))
        specification = fit_psi_bins(reference)
        self.assertEqual(specification['kind'], 'continuous')
        np.testing.assert_allclose(specification['edges'], np.r_[-np.inf, np.quantile(reference, np.arange(.1, 1, .1)), np.inf], rtol=0, atol=1e-12)
        for current in (pd.Series([-1e9, 1e9]), pd.Series(np.linspace(-1e9, 1e9, 300))):
            value, bins = calculate_psi(reference, current, False)
            self.assertTrue(np.isfinite(value))
            self.assertEqual(bins, 11)

    def test_constant_training_feature_detects_new_value(self):
        value, bins = calculate_psi(pd.Series([0]*100), pd.Series([1]*100), False)
        self.assertEqual(bins, 3)
        self.assertGreater(value, 5)


class RuleSelectionRegressionTests(unittest.TestCase):
    def test_unmet_target_is_reported_and_fallback_ignores_test_performance(self):
        table = pd.DataFrame({'leaves': [5, 8, 12], 'val_whole_recall': [.1, .3, .3],
                              'test_whole_recall': [.99, .1, .9]})
        chosen, metadata = select_rule_size(table, teacher_validation_recall=.5)
        self.assertEqual(chosen, 8)
        self.assertFalse(metadata['target_met'])
        self.assertTrue(metadata['fallback_used'])
        self.assertTrue(metadata['fallback_reason'])
        self.assertAlmostEqual(metadata['target_validation_recall'], .45)
        self.assertAlmostEqual(metadata['achieved_fraction_of_teacher'], .6)
        self.assertFalse(metadata['test_used_for_selection'])

    def test_fewest_eligible_leaves_are_chosen_instead_of_highest_recall(self):
        table = pd.DataFrame({'leaves': [5, 8, 12], 'val_whole_recall': [.2, .46, .55]})
        chosen, metadata = select_rule_size(table, teacher_validation_recall=.5)
        self.assertEqual(chosen, 8)
        self.assertTrue(metadata['target_met'])
        self.assertFalse(metadata['fallback_used'])
        self.assertIsNone(metadata['fallback_reason'])


class IndependentVerificationTests(unittest.TestCase):
    def test_ap_treats_an_equal_score_as_a_whole_group(self):
        value = independent_average_precision([1, 0, 1, 0], [.9, .9, .8, .2])
        self.assertAlmostEqual(value, 7/12, places=14)

    def test_roc_auc_assigns_half_credit_to_tied_positive_negative_pairs(self):
        self.assertEqual(independent_roc_auc([1, 0, 1, 0], [.9, .9, .8, .2]), .625)

    def test_float32_validation_tie_is_excluded_as_a_whole(self):
        y = np.array([0]*20+[1, 1])
        scores = np.array([.9, .9]+[.1]*18+[.95, .85], dtype=np.float32)
        threshold = independent_fpr_threshold(y, scores)
        result = independent_binary_metrics(y, scores.astype(np.float64) >= threshold)
        self.assertEqual(result['fp'], 0)
        self.assertEqual(result['tp'], 1)

    def test_capacity_ranking_is_row_id_based_and_permutation_invariant(self):
        ids = np.array([10, 20, 30, 40, 50, 60])
        scores = np.ones(6)
        selected = independent_top_k(scores, 3, ids)
        permutation = np.array([5, 1, 4, 2, 0, 3])
        shuffled = independent_top_k(scores[permutation], 3, ids[permutation])
        self.assertEqual(int(selected.sum()), 3)
        self.assertEqual(set(ids[selected]), set(ids[permutation][shuffled]))

    def test_paired_recall_point_uses_the_same_positive_cases(self):
        result = independent_paired_points([1, 0, 1, 0], [.9, .8, .7, .1], [.6, .5, .4, .3],
                                            [1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(result['recall_difference'], -.5)
        self.assertAlmostEqual(result['ap_difference'], 0., places=14)


if __name__ == '__main__':
    unittest.main()
