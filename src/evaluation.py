"""Sprint 4 evaluation and error-analysis helpers for Phroura."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, extract_feature_rows
from src.model_training import train_random_forest
from src.training import DATASET_PATH, DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE, FeatureSplit, clean_url_dataset, load_url_dataset


RANDOM_FOREST_ERROR_ANALYSIS_PATH = Path("models/random_forest_error_analysis.csv")
RANDOM_FOREST_ERROR_SUMMARY_PATH = Path("models/random_forest_error_summary.json")
RANDOM_FOREST_FEATURE_IMPORTANCE_PATH = Path("models/random_forest_feature_importance.json")
ERROR_ANALYSIS_FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "path_depth",
    "query_parameter_count",
    "subdomain_count",
    "uses_https",
    "has_ip_host",
    "has_port",
    "has_query",
    "digit_count",
    "special_char_count",
    "digit_ratio",
    "special_char_ratio",
    "hostname_digit_ratio",
    "entropy",
    "suspicious_keyword_count",
]
ERROR_SUMMARY_NUMERIC_COLUMNS = [
    "phishing_score",
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "path_depth",
    "query_parameter_count",
    "subdomain_count",
    "uses_https",
    "has_ip_host",
    "has_port",
    "has_query",
    "digit_count",
    "special_char_count",
    "digit_ratio",
    "special_char_ratio",
    "hostname_digit_ratio",
    "entropy",
    "suspicious_keyword_count",
]


@dataclass(frozen=True)
class URLFeatureSplit:
    """Train/test split that also keeps the original URL strings."""

    features: FeatureSplit
    url_train: pd.Series
    url_test: pd.Series


def make_url_feature_split(
    dataset: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> URLFeatureSplit:
    """Create the reproducible split while preserving raw test URLs."""

    cleaned = clean_url_dataset(dataset)
    feature_rows = extract_feature_rows(cleaned["url"])
    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    labels = cleaned["label"]
    urls = cleaned["url"]

    X_train, X_test, y_train, y_test, url_train, url_test = train_test_split(
        features,
        labels,
        urls,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return URLFeatureSplit(
        features=FeatureSplit(
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
        ),
        url_train=url_train.reset_index(drop=True),
        url_test=url_test.reset_index(drop=True),
    )


def build_error_analysis_rows(
    split: URLFeatureSplit,
    predictions: list[int],
    phishing_scores: list[float],
) -> pd.DataFrame:
    """Build a table of false positives and false negatives."""

    analysis_rows = split.features.X_test.copy()
    analysis_rows.insert(0, "url", split.url_test)
    analysis_rows.insert(1, "actual_label", split.features.y_test)
    analysis_rows.insert(2, "predicted_label", predictions)
    analysis_rows.insert(3, "phishing_score", phishing_scores)
    analysis_rows.insert(
        4,
        "error_type",
        [
            _error_type(actual, predicted)
            for actual, predicted in zip(split.features.y_test, predictions, strict=True)
        ],
    )

    error_rows = analysis_rows[analysis_rows["error_type"] != ""].copy()
    selected_columns = [
        "url",
        "actual_label",
        "predicted_label",
        "error_type",
        "phishing_score",
        *ERROR_ANALYSIS_FEATURE_COLUMNS,
    ]
    return error_rows.loc[:, selected_columns].reset_index(drop=True)


def save_random_forest_error_analysis(
    dataset_path: str | Path = DATASET_PATH,
    output_path: str | Path = RANDOM_FOREST_ERROR_ANALYSIS_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Train Random Forest and save its false positives/negatives for inspection."""

    dataset = load_url_dataset(dataset_path)
    split = make_url_feature_split(dataset, random_state=random_state)
    model = train_random_forest(split.features, random_state=random_state)
    predictions = model.predict(split.features.X_test).tolist()
    phishing_scores = model.predict_proba(split.features.X_test)[:, 1].tolist()
    error_rows = build_error_analysis_rows(split, predictions, phishing_scores)

    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    error_rows.to_csv(csv_path, index=False)
    return error_rows


def summarize_error_analysis(error_rows: pd.DataFrame) -> dict[str, Any]:
    """Summarize false positives and false negatives for report analysis."""

    summary: dict[str, Any] = {
        "total_errors": int(len(error_rows)),
        "error_types": {},
    }

    for error_type, rows in error_rows.groupby("error_type"):
        numeric_means = rows[ERROR_SUMMARY_NUMERIC_COLUMNS].mean().round(4)
        summary["error_types"][error_type] = {
            "count": int(len(rows)),
            "mean_features": numeric_means.to_dict(),
        }

    return summary


def save_error_summary(summary: dict[str, Any], output_path: str | Path = RANDOM_FOREST_ERROR_SUMMARY_PATH) -> None:
    """Save an error-analysis summary as JSON."""

    summary_path = Path(output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_random_forest_error_summary(
    error_analysis_path: str | Path = RANDOM_FOREST_ERROR_ANALYSIS_PATH,
    output_path: str | Path = RANDOM_FOREST_ERROR_SUMMARY_PATH,
) -> dict[str, Any]:
    """Read the Random Forest error CSV and save a compact JSON summary."""

    error_rows = pd.read_csv(error_analysis_path)
    summary = summarize_error_analysis(error_rows)
    save_error_summary(summary, output_path)
    return summary


def extract_random_forest_feature_importance(
    dataset_path: str | Path = DATASET_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, float | str]]:
    """Train Random Forest and return sorted feature importances."""

    dataset = load_url_dataset(dataset_path)
    split = make_url_feature_split(dataset, random_state=random_state)
    model = train_random_forest(split.features, random_state=random_state)

    importance_rows = [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in zip(split.features.X_train.columns, model.feature_importances_, strict=True)
    ]
    return sorted(importance_rows, key=lambda row: row["importance"], reverse=True)


def save_random_forest_feature_importance(
    dataset_path: str | Path = DATASET_PATH,
    output_path: str | Path = RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, float | str]]:
    """Save Random Forest feature importances as JSON."""

    importance_rows = extract_random_forest_feature_importance(
        dataset_path=dataset_path,
        random_state=random_state,
    )
    importance_path = Path(output_path)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_path.write_text(json.dumps(importance_rows, indent=2), encoding="utf-8")
    return importance_rows


def _error_type(actual_label: int, predicted_label: int) -> str:
    if actual_label == 0 and predicted_label == 1:
        return "false_positive"
    if actual_label == 1 and predicted_label == 0:
        return "false_negative"
    return ""


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Sprint 4 error-analysis artifacts.")
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(RANDOM_FOREST_ERROR_ANALYSIS_PATH),
        help="Path where the Random Forest error-analysis CSV should be written.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(RANDOM_FOREST_ERROR_SUMMARY_PATH),
        help="Path where the Random Forest error-summary JSON should be written.",
    )
    parser.add_argument(
        "--feature-importance-output",
        default=str(RANDOM_FOREST_FEATURE_IMPORTANCE_PATH),
        help="Path where the Random Forest feature-importance JSON should be written.",
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
    error_rows = save_random_forest_error_analysis(
        dataset_path=args.dataset,
        output_path=args.output,
        random_state=args.random_state,
    )
    summary = summarize_error_analysis(error_rows)
    save_error_summary(summary, args.summary_output)
    importance_rows = save_random_forest_feature_importance(
        dataset_path=args.dataset,
        output_path=args.feature_importance_output,
        random_state=args.random_state,
    )
    print(f"Saved {len(error_rows)} Random Forest errors to {args.output}")
    print(f"Saved Random Forest error summary to {args.summary_output}")
    print(f"Saved {len(importance_rows)} Random Forest feature importances to {args.feature_importance_output}")


if __name__ == "__main__":
    main()
