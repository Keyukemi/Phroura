import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.inference import (
    DEFAULT_PHISHING_THRESHOLD,
    build_model_artifact,
    load_model_artifact,
    predict_url,
    save_model_artifact,
    train_and_save_model,
)
from src.model_training import train_random_forest
from src.training import make_train_test_split


def _sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "url": [
                "https://www.rmit.edu.au/",
                "https://www.latrobe.edu.au/",
                "https://www.monash.edu/",
                "https://www.deakin.edu.au/",
                "https://www.unsw.edu.au/",
                "https://www.sydney.edu.au/",
                "http://192.168.0.10:8080/verify/account",
                "secure-login.example-security.co.uk/reset_password",
                "http://example.com@192.168.0.10/login",
                "http://bank-confirm.example.com/update/password",
                "http://10.0.0.5/secure/login/verify",
                "http://account-update.example-login.test/password",
            ],
            "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )


class InferenceTests(unittest.TestCase):
    def test_save_and_load_model_artifact(self) -> None:
        split = make_train_test_split(_sample_dataset(), test_size=0.33, random_state=3)
        model = train_random_forest(split, random_state=3)
        artifact = build_model_artifact(model)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.joblib"

            save_model_artifact(artifact, model_path)
            loaded_artifact = load_model_artifact(model_path)

            self.assertEqual(loaded_artifact["model_name"], "random_forest")
            self.assertEqual(loaded_artifact["feature_names"], artifact["feature_names"])

    def test_predict_url_returns_app_ready_result(self) -> None:
        split = make_train_test_split(_sample_dataset(), test_size=0.33, random_state=3)
        model = train_random_forest(split, random_state=3)
        artifact = build_model_artifact(model)

        result = predict_url("http://192.168.0.10:8080/verify/account", artifact=artifact)

        self.assertIn(result["label"], ["benign", "phishing"])
        self.assertGreaterEqual(result["phishing_probability"], 0.0)
        self.assertLessEqual(result["phishing_probability"], 1.0)
        self.assertEqual(result["threshold"], DEFAULT_PHISHING_THRESHOLD)
        self.assertIn("heuristic_reasons", result)
        self.assertIn("features", result)
        self.assertGreater(len(result["top_feature_signals"]), 0)

    def test_train_and_save_model_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            model_path = Path(temp_dir) / "model.joblib"
            _sample_dataset().to_csv(dataset_path, index=False)

            artifact = train_and_save_model(
                dataset_path=dataset_path,
                model_path=model_path,
                random_state=5,
            )

            self.assertTrue(model_path.exists())
            self.assertEqual(artifact["model_name"], "random_forest")

    def test_load_model_artifact_requires_existing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_model_artifact("missing-model.joblib")


if __name__ == "__main__":
    unittest.main()
