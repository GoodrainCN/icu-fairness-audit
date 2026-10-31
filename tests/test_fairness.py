"""验证频数 bootstrap 与逐人抽样等价，并保留低支持量保护。"""
import unittest

import numpy as np

from src.fairness import bootstrap_rates, interval


class BootstrapChecks(unittest.TestCase):
    def test_count_bootstrap_agrees_with_explicit_patient_resampling(self):
        counts = [35, 5, 6, 14]
        frequency_rates = bootstrap_rates(counts, np.random.default_rng(17), 30000)
        patients = np.repeat(np.arange(4), counts)
        samples = np.random.default_rng(18).choice(patients, size=(30000, len(patients)))
        with np.errstate(divide="ignore", invalid="ignore"):
            patient_fnr = (samples == 2).sum(axis=1) / ((samples == 2) | (samples == 3)).sum(axis=1)
            patient_fpr = (samples == 1).sum(axis=1) / ((samples == 0) | (samples == 1)).sum(axis=1)
        for metric, direct in [("FNR", patient_fnr), ("FPR", patient_fpr)]:
            self.assertAlmostEqual(float(np.nanmean(frequency_rates[metric])), float(np.nanmean(direct)), delta=0.004)
            self.assertAlmostEqual(float(np.nanstd(frequency_rates[metric])), float(np.nanstd(direct)), delta=0.004)

    def test_undefined_rates_do_not_become_zero_or_false_certainty(self):
        rates = bootstrap_rates([20, 0, 0, 0], np.random.default_rng(4), 1000)
        self.assertTrue(np.isnan(rates["FNR"]).all())
        low, high, valid, status = interval(rates["FNR"], supported=True)
        self.assertTrue(np.isnan(low) and np.isnan(high))
        self.assertEqual(valid, 0)
        self.assertEqual(status, "too_many_undefined_resamples")
        self.assertEqual(interval(np.ones(100), supported=False)[3], "low_support")


if __name__ == "__main__":
    unittest.main()
