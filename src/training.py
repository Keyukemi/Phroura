"""Training data preparation helpers for Phroura.

Sprint 3 starts by turning raw dataset URLs into a reproducible feature split.
The helpers in this module intentionally use only the raw ``url`` and ``label``
columns so model training stays aligned with the custom feature extractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, extract_feature_rows


DATASET_PATH = Path("data/Dataset.csv")
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class FeatureSplit:
    """Container for a reproducible train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_url_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the project dataset using only raw URL and label columns."""

    dataset = pd.read_csv(path, usecols=["url", "label"])
    return clean_url_dataset(dataset)


def clean_url_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable rows before feature extraction."""

    required_columns = {"url", "label"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    cleaned = dataset.loc[:, ["url", "label"]].copy()
    cleaned["url"] = cleaned["url"].astype(str).str.strip()
    cleaned = cleaned[cleaned["url"] != ""]
    cleaned = cleaned.dropna(subset=["label"])
    cleaned = cleaned.drop_duplicates(subset=["url"])
    cleaned["label"] = cleaned["label"].astype(int)
    return cleaned.reset_index(drop=True)


def build_feature_matrix(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert cleaned raw URLs into model-ready features and labels."""

    cleaned = clean_url_dataset(dataset)
    feature_rows = extract_feature_rows(cleaned["url"])
    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    labels = cleaned["label"]
    return features, labels


def make_train_test_split(
    dataset: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> FeatureSplit:
    """Create a reproducible stratified split from raw URL data."""

    features, labels = build_feature_matrix(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    return FeatureSplit(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
