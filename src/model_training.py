"""Model training and evaluation for Phroura."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.training import DATASET_PATH, DEFAULT_RANDOM_STATE, FeatureSplit, load_url_dataset, make_train_test_split


LOGISTIC_REGRESSION_METRICS_PATH = Path("models/logistic_regression_metrics.json")
RANDOM_FOREST_METRICS_PATH = Path("models/random_forest_metrics.json")
SVM_METRICS_PATH = Path("models/svm_metrics.json")
DEFAULT_MODEL_NAME = "logistic_regression"
MODEL_METRICS_PATHS = {
    "logistic_regression": LOGISTIC_REGRESSION_METRICS_PATH,
    "random_forest": RANDOM_FOREST_METRICS_PATH,
    "svm": SVM_METRICS_PATH,
}


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


def build_random_forest_model(random_state: int = DEFAULT_RANDOM_STATE) -> RandomForestClassifier:
    """Create the second Sprint 3 comparison model."""

    return RandomForestClassifier(
        class_weight="balanced",
        max_depth=12,
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
    )


def train_random_forest(split: FeatureSplit, random_state: int = DEFAULT_RANDOM_STATE) -> RandomForestClassifier:
    """Train Random Forest on a prepared feature split."""

    model = build_random_forest_model(random_state=random_state)
    model.fit(split.X_train, split.y_train)
    return model


def build_svm_model(random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    """Create the third Sprint 3 comparison model."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_svm(split: FeatureSplit, random_state: int = DEFAULT_RANDOM_STATE) -> Pipeline:
    """Train a linear SVM on a prepared feature split."""

    model = build_svm_model(random_state=random_state)
    model.fit(split.X_train, split.y_train)
    return model


def evaluate_classifier(model: Any, split: FeatureSplit) -> dict[str, Any]:
    """Evaluate a trained classifier on the test split."""

    predictions = model.predict(split.X_test)
    if hasattr(model, "predict_proba"):
        prediction_scores = model.predict_proba(split.X_test)[:, 1]
    else:
        prediction_scores = model.decision_function(split.X_test)

    return {
        "accuracy": accuracy_score(split.y_test, predictions),
        "precision": precision_score(split.y_test, predictions, zero_division=0),
        "recall": recall_score(split.y_test, predictions, zero_division=0),
        "f1": f1_score(split.y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(split.y_test, prediction_scores),
        "confusion_matrix": confusion_matrix(split.y_test, predictions).tolist(),
        "test_rows": int(len(split.y_test)),
    }


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Save evaluation metrics as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def train_and_evaluate_logistic_regression(
    dataset_path: str | Path = DATASET_PATH,
    metrics_path: str | Path = LOGISTIC_REGRESSION_METRICS_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Run the full Logistic Regression training and evaluation workflow."""

    dataset = load_url_dataset(dataset_path)
    split = make_train_test_split(dataset, random_state=random_state)
    model = train_logistic_regression(split, random_state=random_state)
    metrics = evaluate_classifier(model, split)
    save_metrics(metrics, metrics_path)
    return metrics


def train_and_evaluate_random_forest(
    dataset_path: str | Path = DATASET_PATH,
    metrics_path: str | Path = RANDOM_FOREST_METRICS_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Run the full Random Forest training and evaluation workflow."""

    dataset = load_url_dataset(dataset_path)
    split = make_train_test_split(dataset, random_state=random_state)
    model = train_random_forest(split, random_state=random_state)
    metrics = evaluate_classifier(model, split)
    save_metrics(metrics, metrics_path)
    return metrics


def train_and_evaluate_svm(
    dataset_path: str | Path = DATASET_PATH,
    metrics_path: str | Path = SVM_METRICS_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Run the full linear SVM training and evaluation workflow."""

    dataset = load_url_dataset(dataset_path)
    split = make_train_test_split(dataset, random_state=random_state)
    model = train_svm(split, random_state=random_state)
    metrics = evaluate_classifier(model, split)
    save_metrics(metrics, metrics_path)
    return metrics


def train_and_evaluate_model(
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_path: str | Path = DATASET_PATH,
    metrics_path: str | Path | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train and evaluate one of the supported Sprint 3 models."""

    if model_name == "logistic_regression":
        output_path = metrics_path or LOGISTIC_REGRESSION_METRICS_PATH
        return train_and_evaluate_logistic_regression(dataset_path, output_path, random_state)

    if model_name == "random_forest":
        output_path = metrics_path or RANDOM_FOREST_METRICS_PATH
        return train_and_evaluate_random_forest(dataset_path, output_path, random_state)

    if model_name == "svm":
        output_path = metrics_path or SVM_METRICS_PATH
        return train_and_evaluate_svm(dataset_path, output_path, random_state)

    supported_models = ", ".join(sorted(MODEL_METRICS_PATHS))
    raise ValueError(f"Unsupported model '{model_name}'. Choose one of: {supported_models}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Phroura Sprint 3 models.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_METRICS_PATHS),
        default=DEFAULT_MODEL_NAME,
        help="Model to train and evaluate.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Path where evaluation metrics JSON should be written. Defaults depend on --model.",
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
    metrics = train_and_evaluate_model(
        model_name=args.model,
        dataset_path=args.dataset,
        metrics_path=args.metrics_output,
        random_state=args.random_state,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
