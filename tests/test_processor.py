
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import CONFIG
from core.conversation import Conversation
from core.context_manager import ContextManager
from core.ollama_client import OllamaConnectionError
from core.processor import Processor, ProcessorError
from main import print_context_warning


class FakeClient:
    def __init__(self, response="StarmentAI response"):
        self.response = response
        self.should_fail = False
        self.closed = False
        self.stream_chunks = ["Hello", " ", "world"]
        self.chat_calls = []
        self.stream_calls = []
        self.last_think = None

    def chat(self, messages, think=None):
        self.last_think = think
        self.chat_calls.append(messages)

        if self.should_fail:
            raise OllamaConnectionError(
                "Simulated connection failure"
            )

        return self.response

    def stream_chat(self, messages, think=None):
        self.last_think = think
        self.stream_calls.append(messages)

        if self.should_fail:
            raise OllamaConnectionError(
                "Simulated connection failure"
            )

        for chunk in self.stream_chunks:
            yield chunk

    def close(self):
        self.closed = True


class TestWillThink(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()

        self.processor = Processor(
            conversation=Conversation(client=self.client)
        )

    def tearDown(self):
        self.processor.close()

    def test_easy_question_skips_thinking(self):
        self.assertFalse(self.processor.will_think("Xin chào"))

    def test_hard_question_uses_thinking(self):
        self.assertTrue(
            self.processor.will_think("Vì sao bầu trời màu xanh?")
        )

    def test_adaptive_off_falls_back_to_static_think(self):
        with patch("core.processor.CONFIG") as config:
            config.adaptive_think = False
            config.think = True

            self.assertTrue(self.processor.will_think("Xin chào"))

    def test_stream_passes_think_decision(self):
        list(self.processor.process_stream("Xin chào"))

        self.assertIsNotNone(self.client.last_think)
        self.assertFalse(self.client.last_think)


class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()

        self.conversation = Conversation(
            client=self.client
        )

        self.context_manager = ContextManager(
            context_limit=1024,
            reserved_tokens=256,
            system_prompt_tokens=0,
            style_reminder_tokens=0,
        )

        self.processor = Processor(
            conversation=self.conversation,
            context_manager=self.context_manager,
        )

    def tearDown(self):
        self.processor.close()

    def test_successful_processing(self):
        result = self.processor.process("Hello")

        self.assertTrue(result["success"])
        self.assertEqual(
            result["response"],
            "StarmentAI response",
        )

    def test_result_contains_context_usage(self):
        result = self.processor.process("Hello")

        self.assertIn("context", result)
        self.assertIn(
            "estimated_tokens",
            result["context"],
        )

    def test_history_is_updated(self):
        self.processor.process("Hello")

        history = self.conversation.history

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.processor.process("   ")

    def test_non_string_input(self):
        with self.assertRaises(ValueError):
            self.processor.process(123)

    def test_api_failure(self):
        self.client.should_fail = True

        with self.assertRaises(ProcessorError):
            self.processor.process("Hello")

        self.assertEqual(self.conversation.history, [])

    def test_empty_model_response(self):
        self.client.response = "   "

        with self.assertRaises(ProcessorError):
            self.processor.process("Hello")

        self.assertEqual(self.conversation.history, [])

    def test_history_conflict_aborts_save(self):
        conversation = self.conversation

        def conflicting_chat(messages, think=None):
            conversation.clear()
            return "Late response"

        self.client.chat = conflicting_chat

        with self.assertRaises(ProcessorError):
            self.processor.process("Hello")

        self.assertEqual(conversation.history, [])

    def test_clear_conversation(self):
        self.processor.process("Hello")
        self.processor.clear_conversation()

        self.assertEqual(self.conversation.history, [])

    def test_context_keeps_latest_user_message(self):
        result = self.processor.process("Latest message")

        self.assertTrue(result["success"])

    def test_successful_streaming(self):
        self.client.stream_chunks = ["Hello", " ", "world"]

        chunks = list(self.processor.process_stream("Hello"))

        self.assertEqual("".join(chunks), "Hello world")
        self.assertEqual(len(self.conversation.history), 2)
        self.assertEqual(
            self.conversation.history[1]["content"],
            "Hello world",
        )

    def test_stream_failure_keeps_history_empty(self):
        self.client.should_fail = True

        with self.assertRaises(ProcessorError):
            list(self.processor.process_stream("Hello"))

        self.assertEqual(self.conversation.history, [])

    def test_stream_empty_response(self):
        self.client.stream_chunks = ["   "]

        with self.assertRaises(ProcessorError):
            list(self.processor.process_stream("Hello"))

        self.assertEqual(self.conversation.history, [])

    def test_stream_on_closed_processor(self):
        self.processor.close()

        with self.assertRaises(ProcessorError):
            list(self.processor.process_stream("Hello"))

    def test_default_context_manager_uses_safety_margin(self):
        processor = Processor(
            conversation=Conversation(client=self.client)
        )

        self.assertEqual(
            processor.context_manager.safety_margin,
            CONFIG.context_safety_margin,
        )
        self.assertGreater(
            processor.context_manager.style_reminder_tokens,
            0,
        )

        processor.close()

    def test_last_context_usage_is_recorded(self):
        self.processor.process("Hello")

        usage = self.processor.last_context_usage

        self.assertIsNotNone(usage)
        self.assertIn("estimated_tokens", usage)
        self.assertIn("message_budget", usage)
        self.assertIn("exceeds_message_budget", usage)

    def test_context_warning_is_silent_within_budget(self):
        processor = SimpleNamespace(
            last_context_usage={
                "exceeds_message_budget": False,
            }
        )

        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_context_warning(processor)

        self.assertEqual(output.getvalue(), "")

    def test_context_warning_reports_overflow(self):
        processor = SimpleNamespace(
            last_context_usage={
                "exceeds_message_budget": True,
                "estimated_tokens": 900,
                "message_budget": 575,
                "context_limit": 2048,
                "reserved_tokens": 512,
            }
        )

        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            print_context_warning(processor)

        text = output.getvalue()

        self.assertIn("WARNING", text)
        self.assertIn("900", text)
        self.assertIn("575", text)


if __name__ == "__main__":
    unittest.main()
