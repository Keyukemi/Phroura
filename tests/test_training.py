import unittest

import pandas as pd

from src.features import FEATURE_NAMES
from src.training import build_feature_matrix, clean_url_dataset, make_train_test_split


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": [
                "https://www.rmit.edu.au/",
                "https://www.latrobe.edu.au/",
                "https://www.monash.edu/",
                "https://www.deakin.edu.au/",
                "http://192.168.0.10:8080/verify/account",
                "secure-login.example-security.co.uk/reset_password",
                "http://example.com@192.168.0.10/login",
                "http://bank-confirm.example.com/update/password",
            ],
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            "unused_precomputed_feature": [99] * 8,
        }
    )


class TrainingDataTests(unittest.TestCase):
    def test_clean_url_dataset_keeps_only_url_and_label(self) -> None:
        dataset = pd.DataFrame(
            {
                "url": [" https://example.org ", "", "https://example.org", "http://example.test/login"],
                "label": [0, 0, 0, 1],
                "extra": [1, 2, 3, 4],
            }
        )

        cleaned = clean_url_dataset(dataset)

        self.assertEqual(list(cleaned.columns), ["url", "label"])
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned.loc[0, "url"], "https://example.org")

    def test_clean_url_dataset_requires_url_and_label_columns(self) -> None:
        with self.assertRaises(ValueError):
            clean_url_dataset(pd.DataFrame({"url": ["https://example.org"]}))

    def test_build_feature_matrix_uses_custom_feature_names(self) -> None:
        features, labels = build_feature_matrix(_sample_dataset())

        self.assertEqual(tuple(features.columns), FEATURE_NAMES)
        self.assertEqual(len(features), 8)
        self.assertEqual(labels.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_train_test_split_is_reproducible_and_stratified(self) -> None:
        dataset = _sample_dataset()

        first_split = make_train_test_split(dataset, test_size=0.25, random_state=7)
        second_split = make_train_test_split(dataset, test_size=0.25, random_state=7)

        self.assertTrue(first_split.X_train.equals(second_split.X_train))
        self.assertTrue(first_split.X_test.equals(second_split.X_test))
        self.assertEqual(first_split.y_train.tolist(), second_split.y_train.tolist())
        self.assertEqual(first_split.y_test.tolist(), second_split.y_test.tolist())
        self.assertEqual(first_split.y_test.value_counts().to_dict(), {0: 1, 1: 1})


if __name__ == "__main__":
    unittest.main()
