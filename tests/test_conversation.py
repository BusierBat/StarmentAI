
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.conversation import Conversation, ConversationError


class FakeClient:
    def __init__(self, response="Test response"):
        self.response = response
        self.should_fail = False
        self.closed = False
        self.calls = []

    def chat(self, messages, think=None):
        self.last_think = think
        self.calls.append(messages)

        if self.should_fail:
            raise RuntimeError("Simulated API failure")

        return self.response

    def close(self):
        self.closed = True


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.conversation = Conversation(
            client=self.client
        )

    def tearDown(self):
        self.conversation.close()

    def test_initial_history_is_empty(self):
        self.assertEqual(self.conversation.history, [])

    def test_send_message(self):
        response = self.conversation.send_message("Hello")

        self.assertEqual(response, "Test response")
        self.assertEqual(len(self.conversation.history), 2)

    def test_history_contains_user_and_assistant(self):
        self.conversation.send_message("Hello")

        history = self.conversation.history

        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_history_returns_copy(self):
        self.conversation.send_message("Hello")

        history = self.conversation.history
        history.clear()

        self.assertEqual(len(self.conversation.history), 2)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.conversation.send_message("   ")

    def test_non_string_input(self):
        with self.assertRaises(ValueError):
            self.conversation.send_message(123)

    def test_input_length_limit(self):
        with patch(
            "core.conversation.CONFIG",
            SimpleNamespace(
                max_user_input_length=5,
                max_history_messages=6,
            ),
        ):
            with self.assertRaises(ValueError):
                self.conversation.send_message("123456")

    def test_failed_request_does_not_modify_history(self):
        self.client.should_fail = True

        with self.assertRaises(ConversationError):
            self.conversation.send_message("Hello")

        self.assertEqual(self.conversation.history, [])

    def test_clear_history(self):
        self.conversation.send_message("Hello")
        self.conversation.clear()

        self.assertEqual(self.conversation.history, [])

    def test_history_limit_preserves_recent_messages(self):
        with patch(
            "core.conversation.CONFIG",
            SimpleNamespace(
                max_history_messages=2,
                max_user_input_length=2000,
                adaptive_think=True,
            ),
        ):
            self.conversation.send_message("First")
            self.conversation.send_message("Second")

        history = self.conversation.history

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "Second")

    def test_external_client_is_not_closed(self):
        self.conversation.close()

        self.assertFalse(self.client.closed)

    def test_record_exchange_with_matching_revision(self):
        self.conversation.record_exchange(
            "Hello",
            "Hi",
            expected_revision=0,
        )

        self.assertEqual(len(self.conversation.history), 2)

    def test_record_exchange_revision_conflict(self):
        self.conversation.send_message("First")

        with self.assertRaises(ConversationError):
            self.conversation.record_exchange(
                "Hello",
                "Hi",
                expected_revision=0,
            )

    def test_revision_increments_after_send(self):
        self.assertEqual(self.conversation.revision, 0)

        self.conversation.send_message("Hello")

        self.assertEqual(self.conversation.revision, 1)


if __name__ == "__main__":
    unittest.main()
