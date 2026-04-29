import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.model_training import (
    build_logistic_regression_model,
    evaluate_classifier,
    save_metrics,
    train_and_evaluate_logistic_regression,
    train_logistic_regression,
)
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


class ModelTrainingTests(unittest.TestCase):
    def test_logistic_regression_model_has_expected_steps(self) -> None:
        model = build_logistic_regression_model()

        self.assertEqual(list(model.named_steps), ["scaler", "classifier"])
        self.assertEqual(model.named_steps["classifier"].class_weight, "balanced")

    def test_train_and_evaluate_logistic_regression_returns_metrics(self) -> None:
        split = make_train_test_split(_sample_dataset(), test_size=0.33, random_state=3)
        model = train_logistic_regression(split, random_state=3)

        metrics = evaluate_classifier(model, split)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["test_rows"], len(split.y_test))

    def test_save_metrics_writes_json_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"

            save_metrics({"accuracy": 1.0}, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), '{\n  "accuracy": 1.0\n}')

    def test_full_training_workflow_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            metrics_path = Path(temp_dir) / "metrics.json"
            _sample_dataset().to_csv(dataset_path, index=False)

            metrics = train_and_evaluate_logistic_regression(
                dataset_path=dataset_path,
                metrics_path=metrics_path,
                random_state=5,
            )

            self.assertTrue(metrics_path.exists())
            self.assertEqual(metrics["test_rows"], 3)


if __name__ == "__main__":
    unittest.main()
