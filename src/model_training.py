"""Model training and evaluation for Phroura."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.training import DATASET_PATH, DEFAULT_RANDOM_STATE, FeatureSplit, load_url_dataset, make_train_test_split


METRICS_OUTPUT_PATH = Path("models/logistic_regression_metrics.json")


def build_logistic_regression_model(random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    """Create the first Sprint 3 machine learning model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_logistic_regression(split: FeatureSplit, random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    """Train Logistic Regression on a prepared feature split."""

    model = build_logistic_regression_model(random_state=random_state)
    model.fit(split.X_train, split.y_train)
    return model


def evaluate_classifier(model: Pipeline, split: FeatureSplit) -> dict[str, Any]:
    """Evaluate a trained classifier on the test split."""

    predictions = model.predict(split.X_test)
    probabilities = model.predict_proba(split.X_test)[:, 1]

    return {
        "accuracy": accuracy_score(split.y_test, predictions),
        "precision": precision_score(split.y_test, predictions, zero_division=0),
        "recall": recall_score(split.y_test, predictions, zero_division=0),
        "f1": f1_score(split.y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(split.y_test, probabilities),
        "confusion_matrix": confusion_matrix(split.y_test, predictions).tolist(),
        "test_rows": int(len(split.y_test)),
    }


def save_metrics(metrics: dict[str, Any], path: str | Path = METRICS_OUTPUT_PATH) -> None:
    """Save evaluation metrics as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_and_evaluate_logistic_regression(
    dataset_path: str | Path = DATASET_PATH,
    metrics_path: str | Path = METRICS_OUTPUT_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Run the full Logistic Regression training and evaluation workflow."""

    dataset = load_url_dataset(dataset_path)
    split = make_train_test_split(dataset, random_state=random_state)
    model = train_logistic_regression(split, random_state=random_state)
    metrics = evaluate_classifier(model, split)
    save_metrics(metrics, metrics_path)
    return metrics


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Phroura's first Logistic Regression model.")
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(METRICS_OUTPUT_PATH),
        help="Path where evaluation metrics JSON should be written.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for the reproducible split and model.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    metrics = train_and_evaluate_logistic_regression(
        dataset_path=args.dataset,
        metrics_path=args.metrics_output,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
