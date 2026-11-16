"""Validate the eval set schema and coverage."""

import json
import unittest

from pathlib import Path

from eval.score import CHECK_TYPES

ROOT = Path(__file__).resolve().parents[1]
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"

REQUIRED_CATEGORIES = {
    "style",
    "vietnamese",
    "knowledge",
    "coding",
    "multi_turn",
    "ambiguous",
    "abstain",
    "safety",
    "length",
}

VALID_ROLES = {"system", "user", "assistant"}


def load_items() -> list[dict]:
    items = []

    with EVAL_SET.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                items.append(json.loads(line))

    return items


class TestEvalSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = load_items()

    def test_file_exists(self):
        self.assertTrue(EVAL_SET.exists())

    def test_minimum_size(self):
        self.assertGreaterEqual(len(self.items), 120)

    def test_unique_ids(self):
        ids = [item["id"] for item in self.items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_valid_splits(self):
        splits = {item["split"] for item in self.items}

        self.assertEqual(splits, {"dev", "test"})

    def test_split_sizes(self):
        dev = [item for item in self.items if item["split"] == "dev"]
        test = [item for item in self.items if item["split"] == "test"]

        self.assertGreaterEqual(len(dev), 50)
        self.assertGreaterEqual(len(test), 50)

    def test_category_coverage(self):
        categories = {item["category"] for item in self.items}

        self.assertTrue(REQUIRED_CATEGORIES.issubset(categories))

    def test_category_presence_in_both_splits(self):
        for split in ("dev", "test"):
            categories = {
                item["category"]
                for item in self.items
                if item["split"] == split
            }

            self.assertTrue(
                REQUIRED_CATEGORIES.issubset(categories),
                f"Missing categories in {split}: "
                f"{REQUIRED_CATEGORIES - categories}",
            )

    def test_messages_are_valid(self):
        for item in self.items:
            messages = item.get("messages")

            self.assertIsInstance(messages, list, item["id"])
            self.assertTrue(messages, item["id"])

            for message in messages:
                self.assertIsInstance(message, dict, item["id"])
                self.assertIn(message.get("role"), VALID_ROLES, item["id"])

                content = message.get("content")

                self.assertIsInstance(content, str, item["id"])
                self.assertTrue(content.strip(), item["id"])

    def test_last_message_is_from_user(self):
        for item in self.items:
            self.assertEqual(
                item["messages"][-1]["role"],
                "user",
                item["id"],
            )

    def test_checks_are_valid(self):
        for item in self.items:
            checks = item.get("checks")

            self.assertIsInstance(checks, list, item["id"])
            self.assertTrue(checks, item["id"])

            for check in checks:
                self.assertIsInstance(check, dict, item["id"])
                self.assertIn(check.get("type"), CHECK_TYPES, item["id"])

    def test_reference_is_present(self):
        for item in self.items:
            reference = item.get("reference")

            self.assertIsInstance(reference, str, item["id"])
            self.assertTrue(reference.strip(), item["id"])

    def test_splits_do_not_share_user_prompts(self):
        dev_prompts = {
            item["messages"][0]["content"]
            for item in self.items
            if item["split"] == "dev"
        }

        test_prompts = {
            item["messages"][0]["content"]
            for item in self.items
            if item["split"] == "test"
        }

        self.assertEqual(dev_prompts & test_prompts, set())

    def test_multi_turn_items_have_history(self):
        for item in self.items:
            if item["category"] != "multi_turn":
                continue

            self.assertGreaterEqual(
                len(item["messages"]),
                3,
                item["id"],
            )


if __name__ == "__main__":
    unittest.main()