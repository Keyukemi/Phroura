import unittest

from src.baseline import HEURISTIC_THRESHOLD, classify_url, score_features
from src.features import extract_url_features


class HeuristicBaselineTests(unittest.TestCase):
    def test_clear_benign_url_scores_below_threshold(self) -> None:
        result = classify_url("https://www.rmit.edu.au/")

        self.assertEqual(result["prediction"], 0)
        self.assertEqual(result["label"], "benign")
        self.assertLess(result["score"], result["threshold"])
        self.assertEqual(result["threshold"], HEURISTIC_THRESHOLD)

    def test_clear_suspicious_url_scores_as_phishing(self) -> None:
        result = classify_url("http://192.168.0.10:8080/verify/account")

        self.assertEqual(result["prediction"], 1)
        self.assertEqual(result["label"], "phishing")
        self.assertGreaterEqual(result["score"], result["threshold"])
        self.assertIn("hostname is an IP address", result["reasons"])
        self.assertIn("multiple suspicious keywords are present", result["reasons"])

    def test_borderline_url_can_score_without_crossing_threshold(self) -> None:
        result = classify_url("https://www.service.rmit.edu.au/path")

        self.assertEqual(result["prediction"], 0)
        self.assertEqual(result["label"], "benign")
        self.assertGreater(result["score"], 0)
        self.assertLess(result["score"], result["threshold"])

    def test_at_symbol_adds_specific_reason(self) -> None:
        result = classify_url("http://example.com@192.168.0.10/login")

        self.assertEqual(result["prediction"], 1)
        self.assertIn("URL contains an @ symbol", result["reasons"])

    def test_score_features_returns_numeric_score_and_reasons(self) -> None:
        features = extract_url_features("secure-login.example-security.co.uk/reset_password")

        score, reasons = score_features(features)

        self.assertIsInstance(score, int)
        self.assertIsInstance(reasons, list)
        self.assertGreaterEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
