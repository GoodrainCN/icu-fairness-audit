"""Checks for leakage boundaries and clinically relevant error definitions."""
import unittest

import numpy as np

from src.baseline import Preprocessor, metrics, patient_split, synthetic_data, validate_data


class BaselineChecks(unittest.TestCase):
    def test_repeated_patients_never_cross_partitions(self):
        df = synthetic_data()
        validate_data(df)
        parts = patient_split(df)
        self.assertEqual(sum(map(len, parts)), len(df))
        for i in range(3):
            for j in range(i):
                self.assertFalse(set(parts[i].patient_id) & set(parts[j].patient_id))

    def test_unseen_category_and_test_values_cannot_refit_preprocessing(self):
        train, valid, _ = patient_split(synthetic_data())
        prep = Preprocessor().fit(train, 420)
        means = prep.imputer.initial_imputer_.statistics_.copy()
        categories = list(prep.categories["ethnicity"])
        valid["ethnicity"] = "UNSEEN_ONLY_IN_TEST"
        valid["weight"] = 10000
        transformed = prep.transform(valid)
        self.assertTrue(transformed.ethnicity.isna().all())
        self.assertEqual(prep.categories["ethnicity"], categories)
        np.testing.assert_array_equal(prep.imputer.initial_imputer_.statistics_, means)
        for column in ["patient_id", "encounter_id", "hospital_id", "gender", "icu_id"]:
            self.assertNotIn(column, transformed)

    def test_missed_deaths_and_false_alarms_have_correct_denominators(self):
        result = metrics([1, 1, 0, 0, 0], [0.9, 0.1, 0.8, 0.2, 0.1], 0.5)
        self.assertEqual(result["fnr"], 0.5)
        self.assertAlmostEqual(result["fpr"], 1 / 3)
        self.assertEqual(result["precision"], 0.5)
        empty_positive = metrics([0, 0], [0.1, 0.2], 0.5)
        self.assertIsNone(empty_positive["roc_auc"])
        self.assertIsNone(empty_positive["recall"])
        self.assertIsNone(empty_positive["precision"])


if __name__ == "__main__":
    unittest.main()
