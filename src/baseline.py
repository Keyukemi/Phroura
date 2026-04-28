"""Heuristic phishing baseline for Phroura.

This module implements the first Sprint 3 baseline comparator. It uses the
lexical features from ``src.features`` and applies a small set of transparent
rules before any machine learning model is trained.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterable, TypedDict

from src.features import extract_url_features


class HeuristicResult(TypedDict):
    """Prediction output from the heuristic baseline."""

    prediction: int
    label: str
    score: int
    threshold: int
    reasons: list[str]


HEURISTIC_THRESHOLD = 4


def score_features(features: dict[str, int | float]) -> tuple[int, list[str]]:
    """Score extracted URL features using simple phishing-oriented rules."""

    score = 0
    reasons: list[str] = []

    if features["has_ip_host"]:
        score += 3
        reasons.append("hostname is an IP address")

    if features["suspicious_keyword_count"] >= 2:
        score += 2
        reasons.append("multiple suspicious keywords are present")
    elif features["suspicious_keyword_count"] == 1:
        score += 1
        reasons.append("one suspicious keyword is present")

    if features["url_length"] >= 75:
        score += 2
        reasons.append("URL is unusually long")
    elif features["url_length"] >= 50:
        score += 1
        reasons.append("URL is moderately long")

    if features["subdomain_count"] >= 3:
        score += 2
        reasons.append("hostname has many subdomains")
    elif features["subdomain_count"] == 2:
        score += 1
        reasons.append("hostname has extra subdomains")

    if features["path_depth"] >= 4:
        score += 1
        reasons.append("path is deeply nested")

    if features["entropy"] >= 4.7:
        score += 2
        reasons.append("URL has high character entropy")
    elif features["entropy"] >= 4.2:
        score += 1
        reasons.append("URL has moderately high character entropy")

    if features["has_query"] and features["query_parameter_count"] >= 3:
        score += 1
        reasons.append("query string has multiple parameters")

    if features["special_char_ratio"] >= 0.25:
        score += 1
        reasons.append("URL has a high special-character ratio")

    if features["at_symbol_count"] > 0:
        score += 2
        reasons.append("URL contains an @ symbol")

    return score, reasons


def classify_url(url: str, threshold: int = HEURISTIC_THRESHOLD) -> HeuristicResult:
    """Classify a URL with the heuristic baseline."""

    features = extract_url_features(url)
    score, reasons = score_features(features)
    prediction = int(score >= threshold)

    return {
        "prediction": prediction,
        "label": "phishing" if prediction else "benign",
        "score": score,
        "threshold": threshold,
        "reasons": reasons,
    }


def classify_urls(urls: Iterable[str], threshold: int = HEURISTIC_THRESHOLD) -> list[HeuristicResult]:
    """Classify a batch of URLs with the heuristic baseline."""

    return [classify_url(url, threshold=threshold) for url in urls]


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the heuristic phishing baseline on raw URLs.")
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        required=True,
        help="A URL to classify. Pass --url multiple times for multiple results.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=HEURISTIC_THRESHOLD,
        help="Minimum score required to predict phishing.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level for the output.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    results = classify_urls(args.urls, threshold=args.threshold)
    print(json.dumps(results, indent=args.indent))


if __name__ == "__main__":
    main()
