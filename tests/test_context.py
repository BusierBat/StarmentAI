
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import CONFIG
from core.context_manager import ContextManager


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.manager = ContextManager(
            context_limit=100,
            reserved_tokens=20,
            system_prompt_tokens=0,
            style_reminder_tokens=0,
            chars_per_token=1,
        )

    def test_initial_budget(self):
        self.assertEqual(self.manager.message_budget, 80)

    def test_estimate_tokens(self):
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        self.assertEqual(
            self.manager.estimate_tokens(messages),
            15,
        )

    def test_system_prompt_tokens_are_reserved(self):
        manager = ContextManager(
            context_limit=100,
            reserved_tokens=20,
            system_prompt_tokens=30,
            style_reminder_tokens=0,
            chars_per_token=1,
        )

        self.assertEqual(manager.message_budget, 50)

    def test_budget_exhausted_by_system_prompt(self):
        with self.assertRaises(ValueError):
            ContextManager(
                context_limit=30,
                reserved_tokens=20,
                system_prompt_tokens=30,
                style_reminder_tokens=0,
            )

    def test_default_system_prompt_tokens_from_config(self):
        with patch(
            "core.context_manager.CONFIG",
            SimpleNamespace(
                num_ctx=1000,
                num_predict=None,
                system_prompt_tokens=123,
            ),
        ):
            manager = ContextManager(style_reminder_tokens=0)

        self.assertEqual(manager.system_prompt_tokens, 123)
        self.assertEqual(manager.message_budget, 627)

    def test_reserved_tokens_default_to_num_predict(self):
        with patch(
            "core.context_manager.CONFIG",
            SimpleNamespace(
                num_ctx=1000,
                num_predict=300,
                system_prompt_tokens=0,
            ),
        ):
            manager = ContextManager(style_reminder_tokens=0)

        self.assertEqual(manager.reserved_tokens, 300)
        self.assertEqual(manager.message_budget, 700)

    def test_build_context_keeps_recent_messages(self):
        history = [
            {"role": "user", "content": "Old"},
            {"role": "assistant", "content": "Old reply"},
            {"role": "user", "content": "New"},
        ]

        result = self.manager.build_context(history)

        self.assertEqual(result, history)

    def test_old_messages_are_removed(self):
        history = [
            {"role": "user", "content": "A" * 40},
            {"role": "assistant", "content": "B" * 60},
            {"role": "user", "content": "C" * 30},
        ]

        result = self.manager.build_context(history)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "C" * 30)

    def test_context_selection_is_contiguous(self):
        history = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "B" * 40},
            {"role": "assistant", "content": "b" * 40},
            {"role": "user", "content": "C"},
        ]

        result = self.manager.build_context(history)

        contents = [message["content"] for message in result]

        self.assertNotIn("A", contents)
        self.assertNotIn("a", contents)
        self.assertNotIn("B" * 40, contents)
        self.assertEqual(result[-1]["content"], "C")

    def test_newest_oversized_message_is_preserved(self):
        history = [
            {"role": "user", "content": "X" * 100}
        ]

        result = self.manager.build_context(history)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "X" * 100)

    def test_original_history_is_not_modified(self):
        history = [
            {"role": "user", "content": "Hello"}
        ]

        original = [message.copy() for message in history]

        self.manager.build_context(history)

        self.assertEqual(history, original)

    def test_invalid_roles_are_ignored(self):
        history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]

        result = self.manager.build_context(history)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")

    def test_invalid_messages_are_ignored(self):
        history = [
            "invalid",
            {"role": "user", "content": 123},
            {"role": "assistant", "content": "Valid"},
        ]

        result = self.manager.build_context(history)

        self.assertEqual(
            result,
            [{"role": "assistant", "content": "Valid"}],
        )

    def test_non_list_history(self):
        with self.assertRaises(ValueError):
            self.manager.build_context("invalid")

    def test_invalid_context_limit(self):
        with self.assertRaises(ValueError):
            ContextManager(context_limit=0)

    def test_invalid_reserved_tokens(self):
        with self.assertRaises(ValueError):
            ContextManager(
                context_limit=100,
                reserved_tokens=100,
                system_prompt_tokens=0,
                style_reminder_tokens=0,
            )

    def test_invalid_chars_per_token(self):
        with self.assertRaises(ValueError):
            ContextManager(chars_per_token=0)

    def test_invalid_message_overhead(self):
        with self.assertRaises(ValueError):
            ContextManager(message_overhead=-1)

    def test_safety_margin_is_reserved(self):
        manager = ContextManager(
            context_limit=100,
            reserved_tokens=20,
            system_prompt_tokens=30,
            style_reminder_tokens=10,
            chars_per_token=1,
            safety_margin=10,
        )

        self.assertEqual(manager.message_budget, 30)

    def test_negative_safety_margin_is_rejected(self):
        with self.assertRaises(ValueError):
            ContextManager(safety_margin=-1)

    def test_context_usage_reports_safety_margin(self):
        manager = ContextManager(
            context_limit=100,
            reserved_tokens=20,
            system_prompt_tokens=30,
            style_reminder_tokens=10,
            chars_per_token=1,
            safety_margin=10,
        )

        usage = manager.get_context_usage([])

        self.assertEqual(usage["safety_margin"], 10)
        self.assertEqual(usage["message_budget"], 30)

    def test_context_usage(self):
        messages = [
            {"role": "user", "content": "Hello"}
        ]

        usage = self.manager.get_context_usage(messages)

        self.assertEqual(usage["estimated_tokens"], 15)
        self.assertEqual(usage["message_budget"], 80)
        self.assertEqual(usage["remaining_budget"], 65)
        self.assertEqual(usage["style_reminder_tokens"], 0)

    def test_real_config_budget_is_positive(self):
        manager = ContextManager()

        reserved = (
            manager.reserved_tokens
            + manager.system_prompt_tokens
            + manager.style_reminder_tokens
            + manager.safety_margin
        )

        self.assertGreater(manager.message_budget, 0)
        self.assertLessEqual(reserved, CONFIG.num_ctx)
        self.assertEqual(
            manager.message_budget,
            CONFIG.num_ctx - reserved,
        )

    def test_default_style_reminder_is_counted(self):
        manager = ContextManager()

        self.assertGreater(manager.style_reminder_tokens, 0)
        self.assertEqual(
            manager.style_reminder_tokens,
            ContextManager(
                style_reminder_tokens=None,
                chars_per_token=CONFIG.chars_per_token,
            ).style_reminder_tokens,
        )

    def test_long_history_stays_within_budget(self):
        manager = ContextManager()

        history = []

        for index in range(12):
            history.append({
                "role": "user",
                "content": (
                    f"Câu hỏi số {index} về lập trình Python cơ bản"
                ),
            })
            history.append({
                "role": "assistant",
                "content": (
                    f"Trả lời số {index}: giải thích ngắn gọn, "
                    "rõ ràng, đúng trọng tâm."
                ),
            })

        context = manager.build_context(history)
        usage = manager.get_context_usage(context)

        self.assertLessEqual(
            usage["estimated_tokens"],
            manager.message_budget,
        )
        self.assertEqual(
            context[-1]["content"],
            history[-1]["content"],
        )
        self.assertFalse(usage["exceeds_message_budget"])


if __name__ == "__main__":
    unittest.main()
