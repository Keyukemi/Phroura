"""Inference helpers for the Phroura phishing detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict

import joblib
import pandas as pd

from src.baseline import score_features
from src.features import FEATURE_NAMES, extract_url_features
from src.model_training import train_random_forest
from src.training import DATASET_PATH, DEFAULT_RANDOM_STATE, load_url_dataset, make_train_test_split


DEFAULT_MODEL_PATH = Path("models/random_forest_model.joblib")
DEFAULT_PHISHING_THRESHOLD = 0.5
DEPLOYMENT_MODEL_NAME = "random_forest"


class FeatureSignal(TypedDict):
    """A feature value exposed for lightweight prediction explanation."""

    feature: str
    value: int | float
    importance: float | None


class PredictionResult(TypedDict):
    """Prediction output for a single submitted URL."""

    url: str
    prediction: int
    label: str
    phishing_probability: float
    threshold: float
    heuristic_score: int
    heuristic_reasons: list[str]
    top_feature_signals: list[FeatureSignal]
    features: dict[str, int | float]


def build_model_artifact(model: Any) -> dict[str, Any]:
    """Package a trained model with inference metadata."""

    return {
        "model_name": DEPLOYMENT_MODEL_NAME,
        "feature_names": list(FEATURE_NAMES),
        "model": model,
    }


def train_and_save_model(
    dataset_path: str | Path = DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train the selected deployment model and save it for inference."""

    dataset = load_url_dataset(dataset_path)
    split = make_train_test_split(dataset, random_state=random_state)
    model = train_random_forest(split, random_state=random_state)
    artifact = build_model_artifact(model)
    save_model_artifact(artifact, model_path)
    return artifact


def save_model_artifact(artifact: dict[str, Any], model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
    """Persist a model artifact to disk."""

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and validate a saved model artifact."""

    artifact_path = Path(model_path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. "
            "Run train_and_save_model() or `python3 -m src.inference --save-model` first."
        )

    artifact = joblib.load(artifact_path)
    _validate_model_artifact(artifact)
    return artifact


def predict_url(
    url: str,
    artifact: dict[str, Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_PHISHING_THRESHOLD,
    top_n: int = 5,
) -> PredictionResult:
    """Predict whether a submitted URL is phishing."""

    model_artifact = artifact if artifact is not None else load_model_artifact(model_path)
    _validate_model_artifact(model_artifact)

    features = extract_url_features(url)
    feature_names = model_artifact["feature_names"]
    feature_frame = pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)

    probability = _phishing_probability(model_artifact["model"], feature_frame)
    prediction = int(probability >= threshold)
    heuristic_score, heuristic_reasons = score_features(features)

    return {
        "url": url,
        "prediction": prediction,
        "label": "phishing" if prediction else "benign",
        "phishing_probability": probability,
        "threshold": threshold,
        "heuristic_score": heuristic_score,
        "heuristic_reasons": heuristic_reasons,
        "top_feature_signals": _top_feature_signals(model_artifact["model"], features, top_n=top_n),
        "features": features,
    }


def _validate_model_artifact(artifact: Any) -> None:
    if not isinstance(artifact, dict):
        raise ValueError("Model artifact must be a dictionary.")

    required_keys = {"model_name", "feature_names", "model"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Model artifact is missing required keys: {missing}")

    if tuple(artifact["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model artifact feature names do not match the current feature extractor.")


def _phishing_probability(model: Any, feature_frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(feature_frame)[0][1])

    prediction = int(model.predict(feature_frame)[0])
    return float(prediction)


def _top_feature_signals(
    model: Any,
    features: dict[str, int | float],
    top_n: int = 5,
) -> list[FeatureSignal]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return [
            {"feature": name, "value": features[name], "importance": None}
            for name in FEATURE_NAMES
            if features[name] not in (0, 0.0)
        ][:top_n]

    ranked_signals = sorted(
        (
            {
                "feature": feature_name,
                "value": features[feature_name],
                "importance": float(importance),
            }
            for feature_name, importance in zip(FEATURE_NAMES, importances, strict=True)
        ),
        key=lambda signal: signal["importance"],
        reverse=True,
    )
    return ranked_signals[:top_n]


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phroura URL inference.")
    parser.add_argument("--url", help="URL to classify.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the saved model artifact.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the dataset CSV used when saving a model.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Train and save the deployment model before optional prediction.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PHISHING_THRESHOLD,
        help="Minimum phishing probability required to predict phishing.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for prediction output.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    artifact = None
    if args.save_model:
        artifact = train_and_save_model(dataset_path=args.dataset, model_path=args.model_path)
        print(f"Saved {artifact['model_name']} artifact to {args.model_path}")

    if args.url:
        result = predict_url(
            args.url,
            artifact=artifact,
            model_path=args.model_path,
            threshold=args.threshold,
        )
        print(json.dumps(result, indent=args.indent))

    if not args.save_model and not args.url:
        parser.error("Pass --save-model, --url, or both.")


if __name__ == "__main__":
    main()
