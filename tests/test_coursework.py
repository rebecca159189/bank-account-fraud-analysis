"""Checks for the course-method supplement helpers (src/coursework.py)."""
import unittest

import numpy as np
import pandas as pd

from src.coursework import (FenceClipper, build_prep, continuous_columns, fence_mask, outlier_table,
                            skew_kurtosis_table, smote, vif_table, contract_check)


def toy_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({'normal_field': a, 'lognormal_field': np.exp(rng.normal(size=n)),
                         'collinear_field': a*2+rng.normal(scale=.05, size=n), 'few_values': rng.integers(0, 3, n).astype(float),
                         'flag__missing': (rng.random(n) < .1).astype(float),
                         'payment_type': pd.Categorical(rng.choice(['AA', 'AB'], n)), 'device_os': pd.Categorical(rng.choice(['w', 'l'], n)),
                         'housing_status': pd.Categorical(rng.choice(['BA', 'BB'], n)), 'employment_status': pd.Categorical(rng.choice(['CA', 'CB'], n)),
                         'source': pd.Categorical(rng.choice(['I', 'T'], n))})


class ShapeTests(unittest.TestCase):
    def test_continuous_columns_skip_flags_categories_and_few_values(self):
        self.assertEqual(sorted(continuous_columns(toy_frame())), ['collinear_field', 'lognormal_field', 'normal_field'])

    def test_skewness_flags_only_the_lognormal_field(self):
        x = toy_frame(); t = skew_kurtosis_table(x, continuous_columns(x)).set_index('field')
        self.assertTrue(t.loc['lognormal_field', 'highly_skewed']); self.assertFalse(t.loc['normal_field', 'highly_skewed'])
        self.assertTrue(t.loc['lognormal_field', 'log1p_ok'])
        self.assertAlmostEqual(t.loc['normal_field', 'excess_kurtosis'], 0, delta=.3)

    def test_each_preprocessor_runs_and_has_no_nan(self):
        x = toy_frame(); x.loc[::7, 'normal_field'] = np.nan
        cont = continuous_columns(x)
        for kind in ['minmax', 'robust', 'yeo', 'log1p', 'clip']:
            z = build_prep(x, kind, cont, ['lognormal_field']).fit_transform(x)
            self.assertEqual(len(z), len(x)); self.assertFalse(np.isnan(z).any(), kind)

    def test_log_transform_reduces_skew(self):
        x = toy_frame(); before = abs(x.lognormal_field.skew()); after = abs(np.log1p(x.lognormal_field).skew())
        self.assertLess(after, before/2)


class OutlierTests(unittest.TestCase):
    def test_clipper_uses_training_fences_and_keeps_nan(self):
        tr = np.r_[np.arange(100.), [1e6]].reshape(-1, 1); c = FenceClipper(1.5).fit(tr)
        out = c.transform(np.array([[1e9], [np.nan], [50.]]))
        self.assertLess(out[0, 0], 1e3); self.assertTrue(np.isnan(out[1, 0])); self.assertEqual(out[2, 0], 50.)

    def test_outlier_lift_detects_planted_signal(self):
        rng = np.random.default_rng(1); v = rng.normal(size=20000); y = np.zeros(20000, int)
        v[:200] = 12; y[:200] = rng.random(200) < .8; y[200:] = rng.random(19800) < .01
        x = pd.DataFrame({'f': v}); t, any3 = outlier_table(x, y, ['f'])
        row = t[t.rule == 'iqr3'].iloc[0]
        self.assertGreater(row.lift, 20); self.assertTrue(any3[:200].all()); self.assertEqual(row.flagged_rows, any3.sum())

    def test_fence_mask_ignores_nan(self):
        x = pd.DataFrame({'f': [1., 2, 3, 4, 5, np.nan, 1000]})
        self.assertEqual(fence_mask(x, ['f']).tolist(), [False]*5+[False, True])


class SmoteAndVifTests(unittest.TestCase):
    def test_smote_balances_and_stays_inside_minority_hull(self):
        rng = np.random.default_rng(0)
        z = np.vstack([rng.normal(0, 1, (900, 3)), rng.normal(5, .1, (30, 3))]).astype('float32'); y = np.r_[np.zeros(900, int), np.ones(30, int)]
        z2, y2 = smote(z, y, ratio=1.0)
        self.assertEqual(int((y2 == 1).sum()), 900); self.assertTrue(np.array_equal(z2[:930], z)); self.assertEqual(z2.dtype, z.dtype)
        synth = z2[930:]; self.assertTrue((synth.min(0) >= z[y == 1].min(0)-1e-5).all() and (synth.max(0) <= z[y == 1].max(0)+1e-5).all())

    def test_smote_is_deterministic_and_noop_when_already_balanced(self):
        z = np.random.default_rng(0).normal(size=(40, 2)).astype('float32'); y = np.r_[np.zeros(20, int), np.ones(20, int)]
        self.assertEqual(smote(z, y)[0].shape, z.shape)
        y2 = np.r_[np.zeros(30, int), np.ones(10, int)]
        np.testing.assert_array_equal(smote(z, y2)[0], smote(z, y2)[0])

    def test_vif_flags_collinear_pair(self):
        x = toy_frame(); v = vif_table(x, ['normal_field', 'lognormal_field', 'collinear_field']).set_index('field').vif
        self.assertGreater(v['collinear_field'], 100); self.assertLess(v['lognormal_field'], 2)


class ContractTests(unittest.TestCase):
    def test_contract_counts_violations(self):
        from src.coursework import RANGES, BINARY
        row = {c: (lo+hi)/2 for c, (lo, hi) in RANGES.items()}; row.update({c: 1 for c in BINARY})
        df = pd.DataFrame([row, row]); df.loc[1, 'income'] = 5; df.loc[0, 'email_is_free'] = 3
        t = contract_check(df).set_index('field')
        self.assertEqual(t.loc['income', 'violations'], 1); self.assertEqual(t.loc['email_is_free', 'violations'], 1)
        self.assertEqual(t.loc['month', 'violations'], 0)


if __name__ == '__main__':
    unittest.main()
