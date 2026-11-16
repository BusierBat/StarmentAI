"""Unit tests for deterministic eval scoring."""

import unittest

from eval.score import (
    CHECK_TYPES,
    run_check,
    score_item,
    summarize,
    VIETNAMESE_PATTERN,
)


class TestChecks(unittest.TestCase):
    def test_contains(self):
        passed, _ = run_check("Tôi là StarmentAI", {
            "type": "contains",
            "value": "StarmentAI",
        })

        self.assertTrue(passed)

    def test_contains_case_insensitive(self):
        passed, _ = run_check("Hà Nội", {
            "type": "contains",
            "value": "hà nội",
        })

        self.assertTrue(passed)

    def test_contains_any(self):
        passed, _ = run_check("Tôi thích trà đào", {
            "type": "contains",
            "value": ["cà phê", "trà đào"],
        })

        self.assertTrue(passed)

    def test_contains_all(self):
        passed, _ = run_check("Python và SQL", {
            "type": "contains_all",
            "value": ["python", "sql"],
        })

        self.assertTrue(passed)

    def test_not_contains(self):
        passed, _ = run_check("Món chay ngon", {
            "type": "not_contains",
            "value": ["thịt", "cá"],
        })

        self.assertTrue(passed)

    def test_regex(self):
        passed, _ = run_check("def add(a, b):", {
            "type": "regex",
            "value": r"def\s+\w+\s*\(",
        })

        self.assertTrue(passed)

    def test_regex_flags(self):
        passed, _ = run_check("select * from users", {
            "type": "regex",
            "value": "select",
            "flags": "i",
        })

        self.assertTrue(passed)

    def test_regex_any(self):
        passed, _ = run_check("Dùng requests nhé", {
            "type": "regex_any",
            "value": ["urllib", "requests"],
        })

        self.assertTrue(passed)

    def test_not_regex(self):
        passed, _ = run_check("Xin lỗi, tôi không thể giúp.", {
            "type": "not_regex",
            "value": r"import\s+\w+",
        })

        self.assertTrue(passed)

    def test_max_length(self):
        passed, _ = run_check("ngắn", {"type": "max_length", "value": 10})
        self.assertTrue(passed)

        passed, _ = run_check("quá dài rồi", {
            "type": "max_length",
            "value": 3,
        })
        self.assertFalse(passed)

    def test_min_length(self):
        passed, _ = run_check("ngắn", {"type": "min_length", "value": 10})
        self.assertFalse(passed)

    def test_max_words(self):
        passed, _ = run_check("một hai ba", {
            "type": "max_words",
            "value": 3,
        })

        self.assertTrue(passed)

    def test_min_words(self):
        passed, _ = run_check("một", {"type": "min_words", "value": 3})
        self.assertFalse(passed)

    def test_max_sentences(self):
        passed, _ = run_check("Một. Hai.", {
            "type": "max_sentences",
            "value": 1,
        })

        self.assertFalse(passed)

    def test_min_list_items(self):
        text = "- một\n- hai\n- ba"

        passed, _ = run_check(text, {
            "type": "min_list_items",
            "value": 3,
        })

        self.assertTrue(passed)

    def test_abstain(self):
        passed, _ = run_check("Tôi không biết.", {"type": "abstain"})
        self.assertTrue(passed)

        passed, _ = run_check("Chắc chắn là 42.", {"type": "abstain"})
        self.assertFalse(passed)

    def test_json_valid(self):
        passed, _ = run_check('Kết quả: {"a": 1}', {
            "type": "json_valid",
        })

        self.assertTrue(passed)

    def test_json_invalid(self):
        passed, _ = run_check("Không có JSON", {"type": "json_valid"})
        self.assertFalse(passed)

    def test_vietnamese(self):
        passed, _ = run_check("Hà Nội rất đẹp", {
            "type": "vietnamese",
            "value": 5,
        })

        self.assertTrue(passed)

        passed, _ = run_check("Ha Noi rat dep", {
            "type": "vietnamese",
            "value": 5,
        })

        self.assertFalse(passed)

    def test_vietnamese_pattern_detects_diacritics(self):
        self.assertIsNotNone(VIETNAMESE_PATTERN.search("Tiếng Việt"))

    def test_unknown_check_type(self):
        with self.assertRaises(ValueError):
            run_check("text", {"type": "unknown"})

    def test_all_check_types_are_registered(self):
        self.assertEqual(len(CHECK_TYPES), 15)


class TestScoreItem(unittest.TestCase):
    def test_all_passed(self):
        item = {
            "id": "x",
            "category": "knowledge",
            "reference": "Hà Nội",
            "checks": [{"type": "contains", "value": "Hà Nội"}],
        }

        result = score_item(item, "Thủ đô là Hà Nội.")

        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["all_passed"])

    def test_partial_score(self):
        item = {
            "id": "x",
            "category": "coding",
            "reference": "def",
            "checks": [
                {"type": "contains", "value": "def"},
                {"type": "contains", "value": "return"},
            ],
        }

        result = score_item(item, "def add(a, b): a + b")

        self.assertEqual(result["score"], 0.5)
        self.assertFalse(result["all_passed"])

    def test_no_checks_scores_zero(self):
        item = {"id": "x", "category": "style", "checks": []}

        result = score_item(item, "anything")

        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["all_passed"])


class TestSummarize(unittest.TestCase):
    def setUp(self):
        self.items = [
            {
                "id": "1",
                "category": "abstain",
                "score": 1.0,
                "all_passed": True,
            },
            {
                "id": "2",
                "category": "abstain",
                "score": 0.0,
                "all_passed": False,
            },
            {
                "id": "3",
                "category": "safety",
                "score": 1.0,
                "all_passed": True,
            },
            {
                "id": "4",
                "category": "safety",
                "score": 0.5,
                "all_passed": False,
            },
        ]

    def test_overall_metrics(self):
        summary = summarize(self.items, [100.0, 200.0, 300.0, 400.0])

        self.assertEqual(summary["items"], 4)
        self.assertEqual(summary["mean_score"], 0.625)
        self.assertEqual(summary["all_passed_rate"], 0.5)

    def test_abstention_rate(self):
        summary = summarize(self.items)

        self.assertEqual(summary["abstention_rate"], 0.5)

    def test_safety_violations(self):
        summary = summarize(self.items)

        self.assertEqual(summary["safety_items"], 2)
        self.assertEqual(summary["safety_violations"], 1)

    def test_category_summary(self):
        summary = summarize(self.items)

        self.assertIn("abstain", summary["categories"])
        self.assertEqual(summary["categories"]["abstain"]["items"], 2)
        self.assertEqual(
            summary["categories"]["abstain"]["mean_score"],
            0.5,
        )

    def test_latency_percentiles(self):
        summary = summarize(self.items, [100.0, 200.0, 300.0, 400.0])

        self.assertEqual(summary["latency_ms"]["mean"], 250.0)
        self.assertEqual(summary["latency_ms"]["p50"], 250.0)
        self.assertEqual(summary["latency_ms"]["p95"], 400.0)

    def test_empty_items(self):
        summary = summarize([])

        self.assertEqual(summary["items"], 0)
        self.assertEqual(summary["mean_score"], 0.0)
        self.assertIsNone(summary["latency_ms"])

    def test_invalid_items_type(self):
        with self.assertRaises(ValueError):
            summarize("invalid")


if __name__ == "__main__":
    unittest.main()