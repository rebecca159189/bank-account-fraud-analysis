import unittest
import numpy as np
import pandas as pd
from src.feature_study import (fit_bins, assign_bins, bin_labels, woe_table, psi_from_bins, oof_target_encoding,
                               fit_encoders, add_derived, boot_diff, iv_level, GROUPS)
from src.common import CATEGORICAL


def toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({
        'income': rng.choice([.1, .3, .5, .7, .9], n), 'name_email_similarity': rng.random(n),
        'prev_address_months_count': np.where(rng.random(n) < .3, np.nan, rng.integers(1, 300, n)).astype(float),
        'current_address_months_count': rng.integers(1, 300, n).astype(float), 'customer_age': rng.choice([20, 30, 40, 50, 60], n),
        'velocity_6h': rng.normal(5000, 2000, n), 'velocity_24h': rng.normal(5000, 1000, n), 'velocity_4w': rng.normal(5000, 500, n),
        'date_of_birth_distinct_emails_4w': rng.integers(0, 30, n), 'email_is_free': rng.integers(0, 2, n),
        'phone_home_valid': rng.integers(0, 2, n), 'phone_mobile_valid': rng.integers(0, 2, n),
        'bank_months_count': np.where(rng.random(n) < .2, np.nan, rng.integers(1, 32, n)).astype(float),
        'proposed_credit_limit': rng.choice([200, 500, 1000, 1500], n).astype(float),
        'device_distinct_emails_8w': rng.integers(0, 3, n).astype(float)})
    for c in CATEGORICAL: x[c] = pd.Categorical(rng.choice(list('ABCD'), n))
    x['prev_address_months_count__missing'] = x.prev_address_months_count.isna().astype('float32')
    x['bank_months_count__missing'] = x.bank_months_count.isna().astype('float32')
    y = (rng.random(n) < .05 + .1*(x.payment_type == 'A')).astype(int).to_numpy()
    return x, y


class TestFeatureStudy(unittest.TestCase):
    def test_bins_continuous_and_missing(self):
        x, _ = toy()
        spec = fit_bins(x.prev_address_months_count)
        self.assertEqual(spec['kind'], 'edges')
        b = assign_bins(x.prev_address_months_count, spec)
        self.assertTrue(set(b) <= set(bin_labels(spec)))
        self.assertEqual((b == '缺失').sum(), x.prev_address_months_count.isna().sum())

    def test_bins_discrete_and_unseen(self):
        spec = fit_bins(pd.Series([1., 2., 2., 3.]))
        self.assertEqual(spec['kind'], 'values')
        self.assertEqual(list(assign_bins(pd.Series([2., 9., np.nan]), spec)), ['2', '其他', '缺失'])
        cspec = fit_bins(pd.Series(pd.Categorical(['a', 'b'])))
        self.assertEqual(list(assign_bins(pd.Series(pd.Categorical(['a', 'z'])), cspec)), ['a', '其他'])

    def test_iv_hand_computed(self):
        bins = np.array(['p']*50 + ['q']*50); y = np.array([1]*10 + [0]*40 + [1]*2 + [0]*48)
        tab, iv = woe_table(bins, y)
        bad = np.array([10.5, 2.5])/13; good = np.array([40.5, 48.5])/89
        self.assertAlmostEqual(iv, float(((bad-good)*np.log(bad/good)).sum()), places=12)
        self.assertGreater(tab.set_index('bin').woe['p'], 0)

    def test_iv_zero_signal_small_and_levels(self):
        rng = np.random.default_rng(1); b = rng.choice(list('abc'), 20000); y = rng.integers(0, 2, 20000)
        self.assertLess(woe_table(b, y)[1], .01)
        self.assertEqual(iv_level(.01), '无预测力'); self.assertEqual(iv_level(.2), '中')

    def test_psi_identical_is_zero(self):
        b = np.array(list('aabbc'))
        self.assertAlmostEqual(psi_from_bins(b, b), 0.0, places=12)
        self.assertGreater(psi_from_bins(b, np.array(list('ccccc'))), .5)

    def test_oof_target_encoding_hides_own_label(self):
        x, y = toy(2000)
        x['payment_type'] = pd.Categorical(['U'] + ['A']*1999)      # a category seen once
        te = oof_target_encoding(x, y)
        full = fit_encoders(x, y)
        self.assertNotAlmostEqual(float(te.te_payment_type.iloc[0]), float(full['te']['payment_type']['U']), places=9)

    def test_add_derived_columns(self):
        x, y = toy(); enc = fit_encoders(x, y)
        d = add_derived(x, enc)
        new = [c for c in d if c not in x]
        self.assertEqual(sorted(new), sorted(sum(GROUPS.values(), [])))
        self.assertTrue(np.isfinite(d.v_ratio_6h_24h).all())
        self.assertEqual(int(d.missing_count.max()), 2)

    def test_boot_diff_identical_scores(self):
        rng = np.random.default_rng(0); y = (rng.random(3000) < .1).astype(int); s = rng.random(3000)
        d = boot_diff(y, s, s, 150, np.arange(3000), reps=20)
        self.assertEqual(d['ap_diff'], 0.0); self.assertEqual(d['ap_diff_ci95'], [0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
