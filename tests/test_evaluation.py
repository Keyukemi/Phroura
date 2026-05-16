import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation import (
    build_cross_validation_results,
    build_model_comparison_rows,
    build_error_analysis_rows,
    build_random_forest_tuning_results,
    extract_random_forest_feature_importance,
    make_url_feature_split,
    save_cross_validation_results,
    save_model_comparison,
    save_error_summary,
    save_random_forest_tuning_results,
    save_random_forest_feature_importance,
    save_random_forest_error_analysis,
    save_random_forest_error_summary,
    summarize_error_analysis,
)


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


class EvaluationTests(unittest.TestCase):
    def test_make_url_feature_split_preserves_test_urls(self) -> None:
        split = make_url_feature_split(_sample_dataset(), test_size=0.33, random_state=3)

        self.assertEqual(len(split.url_test), len(split.features.y_test))
        self.assertEqual(len(split.features.X_test), len(split.features.y_test))

    def test_build_error_analysis_rows_keeps_only_mistakes(self) -> None:
        split = make_url_feature_split(_sample_dataset(), test_size=0.33, random_state=3)
        predictions = split.features.y_test.tolist()
        predictions[0] = 1 - predictions[0]
        phishing_scores = [0.75] * len(predictions)

        error_rows = build_error_analysis_rows(split, predictions, phishing_scores)

        self.assertEqual(len(error_rows), 1)
        self.assertIn(error_rows.loc[0, "error_type"], ["false_positive", "false_negative"])
        self.assertIn("url", error_rows.columns)
        self.assertIn("phishing_score", error_rows.columns)
        self.assertIn("entropy", error_rows.columns)

    def test_save_random_forest_error_analysis_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            output_path = Path(temp_dir) / "errors.csv"
            _sample_dataset().to_csv(dataset_path, index=False)

            error_rows = save_random_forest_error_analysis(
                dataset_path=dataset_path,
                output_path=output_path,
                random_state=5,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(list(pd.read_csv(output_path).columns), list(error_rows.columns))

    def test_summarize_error_analysis_groups_by_error_type(self) -> None:
        error_rows = pd.DataFrame(
            {
                "error_type": ["false_positive", "false_positive", "false_negative"],
                "phishing_score": [0.8, 0.6, 0.3],
                "url_length": [20, 30, 40],
                "hostname_length": [10, 15, 20],
                "path_length": [1, 2, 3],
                "query_length": [0, 0, 10],
                "path_depth": [0, 1, 2],
                "query_parameter_count": [0, 0, 1],
                "subdomain_count": [1, 1, 0],
                "uses_https": [1, 1, 0],
                "has_ip_host": [0, 0, 1],
                "has_port": [0, 0, 1],
                "has_query": [0, 0, 1],
                "digit_count": [0, 1, 2],
                "special_char_count": [5, 6, 7],
                "digit_ratio": [0.0, 0.1, 0.2],
                "special_char_ratio": [0.2, 0.3, 0.4],
                "hostname_digit_ratio": [0.0, 0.0, 0.1],
                "entropy": [3.5, 4.0, 4.5],
                "suspicious_keyword_count": [0, 0, 2],
            }
        )

        summary = summarize_error_analysis(error_rows)

        self.assertEqual(summary["total_errors"], 3)
        self.assertEqual(summary["error_types"]["false_positive"]["count"], 2)
        self.assertEqual(summary["error_types"]["false_negative"]["count"], 1)
        self.assertEqual(summary["error_types"]["false_positive"]["mean_features"]["url_length"], 25.0)

    def test_save_random_forest_error_summary_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            error_path = Path(temp_dir) / "errors.csv"
            summary_path = Path(temp_dir) / "summary.json"
            error_rows = pd.DataFrame(
                {
                    "error_type": ["false_positive"],
                    "phishing_score": [0.8],
                    "url_length": [20],
                    "hostname_length": [10],
                    "path_length": [1],
                    "query_length": [0],
                    "path_depth": [0],
                    "query_parameter_count": [0],
                    "subdomain_count": [1],
                    "uses_https": [1],
                    "has_ip_host": [0],
                    "has_port": [0],
                    "has_query": [0],
                    "digit_count": [0],
                    "special_char_count": [5],
                    "digit_ratio": [0.0],
                    "special_char_ratio": [0.2],
                    "hostname_digit_ratio": [0.0],
                    "entropy": [3.5],
                    "suspicious_keyword_count": [0],
                }
            )
            error_rows.to_csv(error_path, index=False)

            summary = save_random_forest_error_summary(error_path, summary_path)

            self.assertTrue(summary_path.exists())
            self.assertEqual(summary["total_errors"], 1)

    def test_save_error_summary_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.json"

            save_error_summary({"total_errors": 0, "error_types": {}}, summary_path)

            self.assertTrue(summary_path.exists())

    def test_extract_random_forest_feature_importance_returns_sorted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            _sample_dataset().to_csv(dataset_path, index=False)

            rows = extract_random_forest_feature_importance(dataset_path=dataset_path, random_state=5)

            self.assertGreater(len(rows), 0)
            self.assertIn("feature", rows[0])
            self.assertIn("importance", rows[0])
            self.assertGreaterEqual(rows[0]["importance"], rows[-1]["importance"])

    def test_save_random_forest_feature_importance_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            output_path = Path(temp_dir) / "importance.json"
            _sample_dataset().to_csv(dataset_path, index=False)

            rows = save_random_forest_feature_importance(
                dataset_path=dataset_path,
                output_path=output_path,
                random_state=5,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(len(rows), 0)

    def test_build_model_comparison_rows_sorts_by_f1(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            weak_path = Path(temp_dir) / "weak.json"
            strong_path = Path(temp_dir) / "strong.json"
            weak_path.write_text(
                """
{
  "accuracy": 0.7,
  "precision": 0.6,
  "recall": 0.5,
  "f1": 0.55,
  "roc_auc": 0.8,
  "confusion_matrix": [[7, 2], [3, 5]],
  "test_rows": 17
}
""".strip(),
                encoding="utf-8",
            )
            strong_path.write_text(
                """
{
  "accuracy": 0.9,
  "precision": 0.8,
  "recall": 0.85,
  "f1": 0.82,
  "roc_auc": 0.95,
  "confusion_matrix": [[9, 1], [1, 8]],
  "test_rows": 19
}
""".strip(),
                encoding="utf-8",
            )

            comparison = build_model_comparison_rows({"weak": weak_path, "strong": strong_path})

            self.assertEqual(comparison.loc[0, "model"], "strong")
            self.assertEqual(comparison.loc[0, "false_positives"], 1)
            self.assertEqual(comparison.loc[0, "false_negatives"], 1)

    def test_save_model_comparison_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            output_path = Path(temp_dir) / "comparison.csv"
            metrics_path.write_text(
                """
{
  "accuracy": 0.9,
  "precision": 0.8,
  "recall": 0.85,
  "f1": 0.82,
  "roc_auc": 0.95,
  "confusion_matrix": [[9, 1], [1, 8]],
  "test_rows": 19
}
""".strip(),
                encoding="utf-8",
            )

            comparison = save_model_comparison(output_path, {"model": metrics_path})

            self.assertTrue(output_path.exists())
            self.assertEqual(len(comparison), 1)

    def test_build_cross_validation_results_returns_model_metric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            _sample_dataset().to_csv(dataset_path, index=False)

            results = build_cross_validation_results(dataset_path=dataset_path, folds=3, random_state=5)

            self.assertEqual(set(results["model"]), {"logistic_regression", "random_forest", "svm"})
            self.assertEqual(set(results["folds"]), {3})
            self.assertEqual(set(results["rows"]), {12})
            for metric_name in ["precision", "recall", "f1", "roc_auc"]:
                self.assertIn(f"{metric_name}_mean", results.columns)
                self.assertIn(f"{metric_name}_std", results.columns)

    def test_save_cross_validation_results_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            output_path = Path(temp_dir) / "cross_validation.csv"
            _sample_dataset().to_csv(dataset_path, index=False)

            results = save_cross_validation_results(
                dataset_path=dataset_path,
                output_path=output_path,
                folds=3,
                random_state=5,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(list(pd.read_csv(output_path).columns), list(results.columns))

    def test_build_random_forest_tuning_results_returns_search_rows_and_best_params(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            _sample_dataset().to_csv(dataset_path, index=False)

            results, best_params = build_random_forest_tuning_results(
                dataset_path=dataset_path,
                folds=3,
                n_iter=2,
                random_state=5,
            )

            self.assertEqual(len(results), 2)
            self.assertIn("rank_test_score", results.columns)
            self.assertIn("param_n_estimators", results.columns)
            self.assertIn("best_params", best_params)
            self.assertIn("best_score", best_params)
            self.assertEqual(best_params["folds"], 3)
            self.assertEqual(best_params["n_iter"], 2)

    def test_save_random_forest_tuning_results_writes_csv_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "dataset.csv"
            results_path = Path(temp_dir) / "tuning.csv"
            best_params_path = Path(temp_dir) / "best_params.json"
            _sample_dataset().to_csv(dataset_path, index=False)

            results, best_params = save_random_forest_tuning_results(
                dataset_path=dataset_path,
                results_output_path=results_path,
                best_params_output_path=best_params_path,
                folds=3,
                n_iter=2,
                random_state=5,
            )

            self.assertTrue(results_path.exists())
            self.assertTrue(best_params_path.exists())
            self.assertEqual(list(pd.read_csv(results_path).columns), list(results.columns))
            self.assertEqual(best_params["n_iter"], 2)


if __name__ == "__main__":
    unittest.main()
