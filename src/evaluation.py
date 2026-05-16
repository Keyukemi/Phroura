"""Sprint 4 evaluation and error-analysis helpers for Phroura."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split

from src.features import FEATURE_NAMES, extract_feature_rows
from src.model_training import (
    MODEL_METRICS_PATHS,
    build_logistic_regression_model,
    build_random_forest_model,
    build_svm_model,
    train_random_forest,
)
from src.training import (
    DATASET_PATH,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FeatureSplit,
    build_feature_matrix,
    clean_url_dataset,
    load_url_dataset,
)


RANDOM_FOREST_ERROR_ANALYSIS_PATH = Path("models/random_forest_error_analysis.csv")
RANDOM_FOREST_ERROR_SUMMARY_PATH = Path("models/random_forest_error_summary.json")
RANDOM_FOREST_FEATURE_IMPORTANCE_PATH = Path("models/random_forest_feature_importance.json")
MODEL_COMPARISON_PATH = Path("models/model_comparison.csv")
CROSS_VALIDATION_RESULTS_PATH = Path("models/cross_validation_results.csv")
RANDOM_FOREST_TUNING_RESULTS_PATH = Path("models/random_forest_tuning_results.csv")
RANDOM_FOREST_BEST_PARAMS_PATH = Path("models/random_forest_best_params.json")
ABLATION_RESULTS_PATH = Path("models/ablation_results.csv")
FEATURE_IMPORTANCE_TABLE_PATH = Path("models/feature_importance_table.csv")
CV_SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
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
FEATURE_GROUPS = {
    "length_features": [
        "url_length",
        "hostname_length",
        "path_length",
        "query_length",
    ],
    "character_composition": [
        "digit_count",
        "letter_count",
        "special_char_count",
        "dot_count",
        "hyphen_count",
        "underscore_count",
        "slash_count",
        "question_mark_count",
        "equals_count",
        "ampersand_count",
        "at_symbol_count",
        "digit_ratio",
        "special_char_ratio",
        "hostname_digit_ratio",
    ],
    "structural_features": [
        "path_depth",
        "query_parameter_count",
        "subdomain_count",
    ],
    "binary_indicators": [
        "uses_https",
        "has_ip_host",
        "has_port",
        "has_query",
        "has_fragment",
    ],
    "keyword_features": [
        "suspicious_keyword_count",
    ],
    "complexity_features": [
        "entropy",
    ],
}
FEATURE_INTERPRETATIONS = {
    "uses_https": (
        "Indicates whether the URL uses HTTPS.",
        "Dataset-dependent signal; HTTPS should not be interpreted as proof of safety or danger.",
    ),
    "entropy": (
        "Measures character randomness and possible obfuscation.",
        "High entropy can also appear in legitimate generated URLs.",
    ),
    "digit_ratio": (
        "Measures how much of the URL is numeric.",
        "Numeric-heavy URLs can be suspicious but are not always malicious.",
    ),
    "digit_count": (
        "Counts numeric characters in the URL.",
        "Useful for generated-looking URLs, but some benign URLs contain IDs.",
    ),
    "url_length": (
        "Measures total URL length.",
        "Long URLs may indicate tracking or obfuscation but are not inherently phishing.",
    ),
    "path_length": (
        "Measures length of the URL path.",
        "Long paths can appear in both phishing and legitimate web applications.",
    ),
    "special_char_ratio": (
        "Measures the proportion of non-alphanumeric characters.",
        "Special characters can indicate obfuscation but also normal query/path syntax.",
    ),
    "letter_count": (
        "Counts alphabetic characters in the URL.",
        "Acts as a general lexical composition signal.",
    ),
    "hyphen_count": (
        "Counts hyphens in the URL.",
        "Hyphens can be used in deceptive domains but are common in benign domains too.",
    ),
    "hostname_length": (
        "Measures domain and subdomain length.",
        "Long hostnames can indicate suspicious subdomain structures.",
    ),
    "dot_count": (
        "Counts dot characters in the URL.",
        "Many dots may indicate deep subdomain nesting.",
    ),
    "path_depth": (
        "Counts path segments.",
        "Deep paths may reflect complex routing or phishing landing pages.",
    ),
    "subdomain_count": (
        "Counts subdomain levels before the registered domain.",
        "Useful for detecting deceptive nested subdomains.",
    ),
    "slash_count": (
        "Counts slash characters.",
        "Correlates with URL structure and path depth.",
    ),
    "special_char_count": (
        "Counts non-alphanumeric characters.",
        "Captures punctuation-heavy or encoded-looking URLs.",
    ),
    "hostname_digit_ratio": (
        "Measures numeric density in the hostname.",
        "May indicate generated domains, but some benign services use digits.",
    ),
    "suspicious_keyword_count": (
        "Counts security or account-related words.",
        "Useful supporting signal, but easy for attackers to avoid.",
    ),
    "query_length": (
        "Measures query string length.",
        "Long queries may reflect tracking, forms, or obfuscation.",
    ),
    "equals_count": (
        "Counts equals signs in query parameters.",
        "Mostly reflects query string structure.",
    ),
    "at_symbol_count": (
        "Counts at symbols.",
        "Can indicate URL obfuscation, though rare in this dataset.",
    ),
    "query_parameter_count": (
        "Counts query parameters.",
        "Captures URL parameter complexity.",
    ),
    "has_query": (
        "Indicates whether a query string is present.",
        "Low-level structure signal.",
    ),
    "underscore_count": (
        "Counts underscores.",
        "Minor lexical composition signal.",
    ),
    "ampersand_count": (
        "Counts ampersands in query strings.",
        "Mostly reflects multiple query parameters.",
    ),
    "question_mark_count": (
        "Counts question marks.",
        "Mostly indicates query string presence.",
    ),
    "has_fragment": (
        "Indicates whether a URL fragment is present.",
        "Low-importance structure signal in this dataset.",
    ),
    "has_ip_host": (
        "Indicates whether the hostname is an IP address.",
        "Important cybersecurity signal, but rare in this dataset.",
    ),
    "has_port": (
        "Indicates whether a custom port is present.",
        "Potentially suspicious but rare in this dataset.",
    ),
}


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


def build_feature_importance_table(
    importance_path: str | Path = RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
) -> pd.DataFrame:
    """Build a report-ready Random Forest feature importance table."""

    importance_rows = json.loads(Path(importance_path).read_text(encoding="utf-8"))
    rows = []
    for rank, row in enumerate(importance_rows, start=1):
        feature = row["feature"]
        interpretation, caution = FEATURE_INTERPRETATIONS.get(
            feature,
            ("Lexical URL signal used by the model.", "Interpret alongside ablation and error analysis."),
        )
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "importance": float(row["importance"]),
                "feature_group": _feature_group_for_feature(feature),
                "interpretation": interpretation,
                "caution": caution,
            }
        )
    return pd.DataFrame(rows)


def save_feature_importance_table(
    importance_path: str | Path = RANDOM_FOREST_FEATURE_IMPORTANCE_PATH,
    output_path: str | Path = FEATURE_IMPORTANCE_TABLE_PATH,
) -> pd.DataFrame:
    """Save a report-ready feature importance table as CSV."""

    table = build_feature_importance_table(importance_path)
    table_path = Path(output_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    return table


def build_model_comparison_rows(
    metric_paths: dict[str, str | Path] = MODEL_METRICS_PATHS,
) -> pd.DataFrame:
    """Build a comparison table from saved model metric JSON files."""

    rows = []
    for model_name, metric_path in metric_paths.items():
        metrics = json.loads(Path(metric_path).read_text(encoding="utf-8"))
        confusion = metrics["confusion_matrix"]
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "true_negatives": confusion[0][0],
                "false_positives": confusion[0][1],
                "false_negatives": confusion[1][0],
                "true_positives": confusion[1][1],
                "test_rows": metrics["test_rows"],
            }
        )

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def save_model_comparison(
    output_path: str | Path = MODEL_COMPARISON_PATH,
    metric_paths: dict[str, str | Path] = MODEL_METRICS_PATHS,
) -> pd.DataFrame:
    """Save a model comparison table as CSV."""

    comparison = build_model_comparison_rows(metric_paths)
    comparison_path = Path(output_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    return comparison


def build_cross_validation_results(
    dataset_path: str | Path = DATASET_PATH,
    folds: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate supported ML models with stratified k-fold cross-validation."""

    dataset = load_url_dataset(dataset_path)
    features, labels = build_feature_matrix(dataset)
    cross_validator = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    models = {
        "logistic_regression": build_logistic_regression_model(random_state=random_state),
        "random_forest": build_random_forest_model(random_state=random_state),
        "svm": build_svm_model(random_state=random_state),
    }

    rows = []
    for model_name, model in models.items():
        scores = cross_validate(
            model,
            features,
            labels,
            cv=cross_validator,
            scoring=CV_SCORING,
            error_score="raise",
        )
        row: dict[str, Any] = {
            "model": model_name,
            "folds": folds,
            "rows": int(len(labels)),
        }
        for metric_name in CV_SCORING:
            metric_scores = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = float(metric_scores.mean())
            row[f"{metric_name}_std"] = float(metric_scores.std())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def save_cross_validation_results(
    dataset_path: str | Path = DATASET_PATH,
    output_path: str | Path = CROSS_VALIDATION_RESULTS_PATH,
    folds: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Save stratified k-fold cross-validation results as a report-ready CSV."""

    results = build_cross_validation_results(
        dataset_path=dataset_path,
        folds=folds,
        random_state=random_state,
    )
    results_path = Path(output_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    return results


def build_random_forest_tuning_results(
    dataset_path: str | Path = DATASET_PATH,
    folds: int = 3,
    n_iter: int = 12,
    random_state: int = DEFAULT_RANDOM_STATE,
    scoring: str = "f1",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune Random Forest hyperparameters with randomized cross-validation search."""

    dataset = load_url_dataset(dataset_path)
    features, labels = build_feature_matrix(dataset)
    cross_validator = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=build_random_forest_model(random_state=random_state),
        param_distributions={
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 12, 16, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        n_iter=n_iter,
        scoring=scoring,
        cv=cross_validator,
        n_jobs=n_jobs,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(features, labels)

    results = pd.DataFrame(search.cv_results_)
    selected_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_class_weight",
    ]
    tuning_results = results.loc[:, selected_columns].sort_values("rank_test_score").reset_index(drop=True)
    best_params = {
        "best_params": _json_ready(search.best_params_),
        "best_score": float(search.best_score_),
        "folds": folds,
        "n_iter": n_iter,
        "scoring": scoring,
        "rows": int(len(labels)),
        "n_jobs": n_jobs,
    }
    return tuning_results, best_params


def save_random_forest_tuning_results(
    dataset_path: str | Path = DATASET_PATH,
    results_output_path: str | Path = RANDOM_FOREST_TUNING_RESULTS_PATH,
    best_params_output_path: str | Path = RANDOM_FOREST_BEST_PARAMS_PATH,
    folds: int = 3,
    n_iter: int = 12,
    random_state: int = DEFAULT_RANDOM_STATE,
    scoring: str = "f1",
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Save Random Forest tuning results and the best parameter set."""

    tuning_results, best_params = build_random_forest_tuning_results(
        dataset_path=dataset_path,
        folds=folds,
        n_iter=n_iter,
        random_state=random_state,
        scoring=scoring,
        n_jobs=n_jobs,
    )
    results_path = Path(results_output_path)
    params_path = Path(best_params_output_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(results_path, index=False)
    params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    return tuning_results, best_params


def build_feature_ablation_results(
    dataset_path: str | Path = DATASET_PATH,
    folds: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate Random Forest after removing each lexical feature group."""

    dataset = load_url_dataset(dataset_path)
    features, labels = build_feature_matrix(dataset)
    baseline_scores = _cross_validate_random_forest_features(
        features=features,
        labels=labels,
        folds=folds,
        random_state=random_state,
    )
    rows = [
        {
            "experiment": "all_features",
            "removed_group": "",
            "removed_feature_count": 0,
            "remaining_feature_count": int(len(features.columns)),
            "removed_features": "",
            **baseline_scores,
        }
    ]

    for group_name, removed_features in FEATURE_GROUPS.items():
        remaining_features = features.drop(columns=removed_features)
        scores = _cross_validate_random_forest_features(
            features=remaining_features,
            labels=labels,
            folds=folds,
            random_state=random_state,
        )
        rows.append(
            {
                "experiment": f"without_{group_name}",
                "removed_group": group_name,
                "removed_feature_count": len(removed_features),
                "remaining_feature_count": int(len(remaining_features.columns)),
                "removed_features": ", ".join(removed_features),
                **scores,
            }
        )

    results = pd.DataFrame(rows)
    results["f1_delta_from_all_features"] = results["f1_mean"] - baseline_scores["f1_mean"]
    results["recall_delta_from_all_features"] = results["recall_mean"] - baseline_scores["recall_mean"]
    results["roc_auc_delta_from_all_features"] = results["roc_auc_mean"] - baseline_scores["roc_auc_mean"]
    return results.sort_values("f1_mean", ascending=False).reset_index(drop=True)


def save_feature_ablation_results(
    dataset_path: str | Path = DATASET_PATH,
    output_path: str | Path = ABLATION_RESULTS_PATH,
    folds: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Save Random Forest feature ablation results as a report-ready CSV."""

    results = build_feature_ablation_results(
        dataset_path=dataset_path,
        folds=folds,
        random_state=random_state,
    )
    results_path = Path(output_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    return results


def _error_type(actual_label: int, predicted_label: int) -> str:
    if actual_label == 0 and predicted_label == 1:
        return "false_positive"
    if actual_label == 1 and predicted_label == 0:
        return "false_negative"
    return ""


def _cross_validate_random_forest_features(
    features: pd.DataFrame,
    labels: pd.Series,
    folds: int,
    random_state: int,
) -> dict[str, float | int]:
    cross_validator = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_validate(
        build_random_forest_model(random_state=random_state),
        features,
        labels,
        cv=cross_validator,
        scoring=CV_SCORING,
        error_score="raise",
    )
    row: dict[str, float | int] = {
        "folds": folds,
        "rows": int(len(labels)),
    }
    for metric_name in CV_SCORING:
        metric_scores = scores[f"test_{metric_name}"]
        row[f"{metric_name}_mean"] = float(metric_scores.mean())
        row[f"{metric_name}_std"] = float(metric_scores.std())
    return row


def _feature_group_for_feature(feature: str) -> str:
    for group_name, features in FEATURE_GROUPS.items():
        if feature in features:
            return group_name
    return "unassigned"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


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
        "--comparison-output",
        default=str(MODEL_COMPARISON_PATH),
        help="Path where the generated model-comparison CSV should be written.",
    )
    parser.add_argument(
        "--cross-validation-output",
        default=str(CROSS_VALIDATION_RESULTS_PATH),
        help="Path where the cross-validation results CSV should be written.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for stratified cross-validation.",
    )
    parser.add_argument(
        "--tune-random-forest",
        action="store_true",
        help="Run Random Forest hyperparameter tuning and save tuning artifacts.",
    )
    parser.add_argument(
        "--tuning-output",
        default=str(RANDOM_FOREST_TUNING_RESULTS_PATH),
        help="Path where Random Forest tuning results should be written.",
    )
    parser.add_argument(
        "--best-params-output",
        default=str(RANDOM_FOREST_BEST_PARAMS_PATH),
        help="Path where the best Random Forest parameters should be written.",
    )
    parser.add_argument(
        "--tuning-folds",
        type=int,
        default=3,
        help="Number of folds for Random Forest hyperparameter tuning.",
    )
    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=12,
        help="Number of sampled parameter combinations for Random Forest tuning.",
    )
    parser.add_argument(
        "--tuning-scoring",
        default="f1",
        help="Scoring metric used for Random Forest hyperparameter tuning.",
    )
    parser.add_argument(
        "--tuning-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Random Forest hyperparameter tuning.",
    )
    parser.add_argument(
        "--ablation-output",
        default=str(ABLATION_RESULTS_PATH),
        help="Path where feature ablation results should be written.",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run Random Forest feature ablation study and save the results.",
    )
    parser.add_argument(
        "--feature-importance-table-output",
        default=str(FEATURE_IMPORTANCE_TABLE_PATH),
        help="Path where the report-ready feature importance CSV should be written.",
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
    comparison = save_model_comparison(args.comparison_output)
    cross_validation = save_cross_validation_results(
        dataset_path=args.dataset,
        output_path=args.cross_validation_output,
        folds=args.folds,
        random_state=args.random_state,
    )
    print(f"Saved {len(error_rows)} Random Forest errors to {args.output}")
    print(f"Saved Random Forest error summary to {args.summary_output}")
    print(f"Saved {len(importance_rows)} Random Forest feature importances to {args.feature_importance_output}")
    print(f"Saved {len(comparison)} model comparison rows to {args.comparison_output}")
    print(f"Saved {len(cross_validation)} cross-validation rows to {args.cross_validation_output}")
    feature_importance_table = save_feature_importance_table(
        importance_path=args.feature_importance_output,
        output_path=args.feature_importance_table_output,
    )
    print(f"Saved {len(feature_importance_table)} feature importance table rows to {args.feature_importance_table_output}")
    if args.tune_random_forest:
        tuning_results, best_params = save_random_forest_tuning_results(
            dataset_path=args.dataset,
            results_output_path=args.tuning_output,
            best_params_output_path=args.best_params_output,
            folds=args.tuning_folds,
            n_iter=args.tuning_iterations,
            random_state=args.random_state,
            scoring=args.tuning_scoring,
            n_jobs=args.tuning_jobs,
        )
        print(f"Saved {len(tuning_results)} Random Forest tuning rows to {args.tuning_output}")
        print(f"Saved best Random Forest parameters to {args.best_params_output}")
        print(f"Best Random Forest {args.tuning_scoring}: {best_params['best_score']:.4f}")
    if args.run_ablation:
        ablation_results = save_feature_ablation_results(
            dataset_path=args.dataset,
            output_path=args.ablation_output,
            folds=args.folds,
            random_state=args.random_state,
        )
        print(f"Saved {len(ablation_results)} feature ablation rows to {args.ablation_output}")


if __name__ == "__main__":
    main()
