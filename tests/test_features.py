import math
import unittest

from src.features import FEATURE_NAMES, extract_feature_rows, extract_url_features


class ExtractURLFeaturesTests(unittest.TestCase):
    def test_https_url_with_query_and_fragment(self) -> None:
        url = "https://example.com/login?next=dashboard&lang=en#reset"

        features = extract_url_features(url)

        self.assertEqual(features["uses_https"], 1)
        self.assertEqual(features["has_query"], 1)
        self.assertEqual(features["has_fragment"], 1)
        self.assertEqual(features["path_depth"], 1)
        self.assertEqual(features["query_parameter_count"], 2)
        self.assertEqual(features["question_mark_count"], 1)
        self.assertEqual(features["equals_count"], 2)
        self.assertEqual(features["ampersand_count"], 1)
        self.assertGreaterEqual(features["suspicious_keyword_count"], 1)
        self.assertTrue(math.isclose(features["digit_ratio"], 0.0, abs_tol=1e-12))

    def test_ip_host_and_port_are_detected(self) -> None:
        url = "http://192.168.0.10:8080/verify/account"

        features = extract_url_features(url)

        self.assertEqual(features["has_ip_host"], 1)
        self.assertEqual(features["has_port"], 1)
        self.assertEqual(features["subdomain_count"], 0)
        self.assertGreaterEqual(features["digit_count"], 10)
        self.assertGreaterEqual(features["suspicious_keyword_count"], 2)

    def test_multilevel_public_suffix_subdomain_count_uses_heuristic(self) -> None:
        url = "https://www.service.rmit.edu.au/path"

        features = extract_url_features(url)

        self.assertEqual(features["subdomain_count"], 2)
        self.assertEqual(features["hostname_length"], len("www.service.rmit.edu.au"))

    def test_url_without_scheme_is_still_parsed(self) -> None:
        url = "secure-login.example-security.co.uk/reset_password"

        features = extract_url_features(url)

        self.assertEqual(features["uses_https"], 0)
        self.assertEqual(features["subdomain_count"], 1)
        self.assertEqual(features["underscore_count"], 1)
        self.assertEqual(features["hyphen_count"], 2)
        self.assertGreaterEqual(features["suspicious_keyword_count"], 3)

    def test_empty_url_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            extract_url_features("   ")

    def test_batch_helper_can_include_input_url(self) -> None:
        rows = extract_feature_rows(
            ["https://example.org", "http://10.0.0.1/login"],
            include_url=True,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["url"], "https://example.org")
        self.assertEqual(rows[1]["has_ip_host"], 1)

    def test_feature_name_list_matches_output_keys(self) -> None:
        features = extract_url_features("https://example.org")
        self.assertEqual(tuple(features.keys()), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
