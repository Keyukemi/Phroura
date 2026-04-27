"""Lexical URL feature extraction for Phroura.

The first Sprint 2 feature set is intentionally small and explainable.
Every feature is derived from the raw URL string at runtime rather than
from the precomputed numeric columns included in the source dataset.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
from collections import Counter
from typing import Iterable
from urllib.parse import urlsplit

COMMON_SECOND_LEVEL_SUFFIXES = {
    "ac",
    "co",
    "com",
    "edu",
    "gov",
    "mil",
    "net",
    "org",
}

SUSPICIOUS_KEYWORDS = (
    "account",
    "bank",
    "confirm",
    "login",
    "password",
    "secure",
    "signin",
    "update",
    "verify",
    "webscr",
)

FEATURE_NAMES = (
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
    "has_fragment",
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
    "entropy",
    "suspicious_keyword_count",
)


def _normalized_url(url: str) -> str:
    normalized = url.strip()
    if not normalized:
        raise ValueError("URL must not be empty.")
    return normalized


def _urlsplit_with_fallback(url: str):
    if "://" in url:
        return urlsplit(url)
    return urlsplit(f"http://{url}")


def _safe_port(parts) -> int | None:
    try:
        return parts.port
    except ValueError:
        return None


def _is_ip_host(hostname: str) -> bool:
    if not hostname:
        return False
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def _subdomain_count(hostname: str) -> int:
    if not hostname or _is_ip_host(hostname):
        return 0

    labels = [label for label in hostname.split(".") if label]
    if len(labels) < 3:
        return 0

    root_label_count = 2
    if len(labels[-1]) == 2 and labels[-2] in COMMON_SECOND_LEVEL_SUFFIXES and len(labels) >= 3:
        root_label_count = 3

    return max(len(labels) - root_label_count, 0)


def _character_entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    if length == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)
    return entropy


def _count_special_characters(value: str) -> int:
    return sum(1 for char in value if not char.isalnum())


def extract_url_features(url: str) -> dict[str, int | float]:
    """Extract the v1 lexical feature set from a raw URL string."""

    normalized_url = _normalized_url(url)
    parts = _urlsplit_with_fallback(normalized_url)
    hostname = parts.hostname or ""
    query = parts.query or ""
    path = parts.path or ""
    lowered_url = normalized_url.lower()

    digit_count = sum(char.isdigit() for char in normalized_url)
    letter_count = sum(char.isalpha() for char in normalized_url)
    special_char_count = _count_special_characters(normalized_url)
    hostname_digit_count = sum(char.isdigit() for char in hostname)
    url_length = len(normalized_url)
    hostname_length = len(hostname)

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": len(path),
        "query_length": len(query),
        "path_depth": len([segment for segment in path.split("/") if segment]),
        "query_parameter_count": len([item for item in query.split("&") if item]),
        "subdomain_count": _subdomain_count(hostname),
        "uses_https": int(parts.scheme.lower() == "https"),
        "has_ip_host": int(_is_ip_host(hostname)),
        "has_port": int(_safe_port(parts) is not None),
        "has_query": int(bool(query)),
        "has_fragment": int(bool(parts.fragment)),
        "digit_count": digit_count,
        "letter_count": letter_count,
        "special_char_count": special_char_count,
        "dot_count": normalized_url.count("."),
        "hyphen_count": normalized_url.count("-"),
        "underscore_count": normalized_url.count("_"),
        "slash_count": normalized_url.count("/"),
        "question_mark_count": normalized_url.count("?"),
        "equals_count": normalized_url.count("="),
        "ampersand_count": normalized_url.count("&"),
        "at_symbol_count": normalized_url.count("@"),
        "digit_ratio": digit_count / url_length,
        "special_char_ratio": special_char_count / url_length,
        "hostname_digit_ratio": hostname_digit_count / hostname_length if hostname_length else 0.0,
        "entropy": _character_entropy(normalized_url),
        "suspicious_keyword_count": sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered_url),
    }


def extract_feature_rows(urls: Iterable[str], include_url: bool = False) -> list[dict[str, int | float | str]]:
    """Extract features for a batch of URLs."""

    rows: list[dict[str, int | float | str]] = []
    for url in urls:
        feature_row: dict[str, int | float | str] = extract_url_features(url)
        if include_url:
            feature_row = {"url": url, **feature_row}
        rows.append(feature_row)
    return rows


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract lexical features from raw URLs.")
    parser.add_argument(
        "--url",
        dest="urls",
        action="append",
        required=True,
        help="A URL to analyze. Pass --url multiple times for multiple rows.",
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
    rows = extract_feature_rows(args.urls, include_url=True)
    print(json.dumps(rows, indent=args.indent))


if __name__ == "__main__":
    main()
