"""Tests for the eval runner helpers."""

import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from eval.run_eval import (
    load_items,
    render_markdown,
    select_per_category,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"


class TestSelectPerCategory(unittest.TestCase):
    def setUp(self):
        self.items = [
            {"id": f"a-{index}", "category": "style"}
            for index in range(5)
        ] + [
            {"id": f"b-{index}", "category": "coding"}
            for index in range(5)
        ]

    def test_limits_each_category(self):
        selected = select_per_category(self.items, 2)

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            [item["id"] for item in selected],
            ["a-0", "a-1", "b-0", "b-1"],
        )

    def test_zero_returns_empty(self):
        self.assertEqual(select_per_category(self.items, 0), [])


class TestLoadItems(unittest.TestCase):
    def test_filters_split_and_blank_lines(self):
        items = load_items(EVAL_SET, "dev")

        self.assertTrue(items)
        self.assertTrue(all(item["split"] == "dev" for item in items))

    def test_all_split(self):
        items = load_items(EVAL_SET, "all")

        self.assertEqual(len(items), 140)

    def test_missing_file_raises(self):
        with self.assertRaises(OSError):
            load_items(ROOT / "eval" / "khong-ton-tai.jsonl", "dev")

    def test_invalid_json_raises(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("{khong-phai-json}\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_items(path, "dev")


class TestRenderMarkdown(unittest.TestCase):
    def test_contains_sections(self):
        report = {
            "model": "starmentai",
            "split": "dev",
            "think": "auto",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "errors": 0,
            "items": [
                {
                    "id": "dev-001",
                    "category": "style",
                    "score": 0.5,
                    "all_passed": False,
                    "response": "Trả lời mẫu",
                },
                {
                    "id": "dev-002",
                    "category": "coding",
                    "score": 1.0,
                    "all_passed": True,
                    "response": "def add(a, b): return a + b",
                },
            ],
            "summary": {
                "items": 2,
                "mean_score": 0.75,
                "all_passed_rate": 0.5,
                "abstention_rate": 0.0,
                "safety_items": 0,
                "safety_violations": 0,
                "categories": {
                    "style": {
                        "items": 1,
                        "mean_score": 0.5,
                        "all_passed_rate": 0.0,
                    },
                    "coding": {
                        "items": 1,
                        "mean_score": 1.0,
                        "all_passed_rate": 1.0,
                    },
                },
                "latency_ms": None,
            },
        }

        markdown = render_markdown(report)

        self.assertIn("starmentai", markdown)
        self.assertIn("## Categories", markdown)

        failed_section = markdown.split("## Failed items")[1]

        self.assertIn("dev-001", failed_section)
        self.assertIn("Trả lời mẫu", failed_section)
        self.assertNotIn("dev-002", failed_section)


if __name__ == "__main__":
    unittest.main()