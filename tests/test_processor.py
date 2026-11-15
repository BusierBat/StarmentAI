
import unittest

from core.conversation import Conversation
from core.context_manager import ContextManager
from core.ollama_client import OllamaConnectionError
from core.processor import Processor, ProcessorError


class FakeClient:
    def __init__(self, response="StarmentAI response"):
        self.response = response
        self.should_fail = False
        self.closed = False
        self.stream_chunks = ["Hello", " ", "world"]
        self.chat_calls = []
        self.stream_calls = []

    def chat(self, messages):
        self.chat_calls.append(messages)

        if self.should_fail:
            raise OllamaConnectionError(
                "Simulated connection failure"
            )

        return self.response

    def stream_chat(self, messages):
        self.stream_calls.append(messages)

        if self.should_fail:
            raise OllamaConnectionError(
                "Simulated connection failure"
            )

        for chunk in self.stream_chunks:
            yield chunk

    def close(self):
        self.closed = True


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

        def conflicting_chat(messages):
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


if __name__ == "__main__":
    unittest.main()
