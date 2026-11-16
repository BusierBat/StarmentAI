import unittest

from core.token_counter import (
    count_message,
    count_messages,
    count_text,
)


class TestTokenCounter(unittest.TestCase):
    def test_empty_text_is_zero(self):
        self.assertEqual(count_text(""), 0)

    def test_heuristic_count(self):
        self.assertEqual(count_text("hello", chars_per_token=2.5), 2)

    def test_message_count_includes_role_and_overhead(self):
        message = {"role": "user", "content": "hello"}

        self.assertEqual(
            count_message(message, chars_per_token=1, overhead=6),
            15,
        )

    def test_messages_skip_non_dict_entries(self):
        messages = [
            {"role": "user", "content": "hello"},
            "invalid",
        ]

        self.assertEqual(
            count_messages(messages, chars_per_token=1, overhead=6),
            15,
        )

    def test_invalid_chars_per_token(self):
        with self.assertRaises(ValueError):
            count_text("hello", chars_per_token=0)

    def test_message_requires_dict(self):
        with self.assertRaises(ValueError):
            count_message("invalid")


if __name__ == "__main__":
    unittest.main()