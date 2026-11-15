
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from config import CONFIG, _load_system_prompt
from core.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
    question_needs_thinking,
)


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def tearDown(self):
        self.client.close()

    def make_response(self, status_code=200, data=None):
        response = Mock()
        response.status_code = status_code
        response.ok = 200 <= status_code < 300
        response.text = "Test response"
        response.json.return_value = data or {
            "message": {"content": "Hello from StarmentAI"}
        }
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    def test_successful_chat(self):
        response = self.make_response()

        with patch.object(
            self.client.session,
            "post",
            return_value=response,
        ):
            result = self.client.chat([
                {"role": "user", "content": "Hello"}
            ])

        self.assertEqual(result, "Hello from StarmentAI")

    def test_empty_messages(self):
        with self.assertRaises(ValueError):
            self.client.chat([])

    def test_invalid_message_role(self):
        with self.assertRaises(ValueError):
            self.client.chat([
                {"role": "invalid", "content": "Hello"}
            ])

    def test_invalid_message_content(self):
        with self.assertRaises(ValueError):
            self.client.chat([
                {"role": "user", "content": 123}
            ])

    def test_connection_timeout(self):
        with patch.object(
            self.client.session,
            "post",
            side_effect=requests.exceptions.Timeout,
        ):
            with self.assertRaises(OllamaConnectionError):
                self.client.chat([
                    {"role": "user", "content": "Hello"}
                ])

    def test_connection_error(self):
        with patch.object(
            self.client.session,
            "post",
            side_effect=requests.exceptions.ConnectionError,
        ):
            with self.assertRaises(OllamaConnectionError):
                self.client.chat([
                    {"role": "user", "content": "Hello"}
                ])

    def test_http_error(self):
        response = self.make_response(
            status_code=400,
            data={"error": "Bad request"},
        )

        with patch.object(
            self.client.session,
            "post",
            return_value=response,
        ):
            with self.assertRaises(OllamaResponseError):
                self.client.chat([
                    {"role": "user", "content": "Hello"}
                ])

    def test_invalid_json_response(self):
        response = self.make_response()
        response.json.side_effect = ValueError("Invalid JSON")

        with patch.object(
            self.client.session,
            "post",
            return_value=response,
        ):
            with self.assertRaises(OllamaResponseError):
                self.client.chat([
                    {"role": "user", "content": "Hello"}
                ])

    def test_missing_message(self):
        response = self.make_response(
            data={"unexpected": "data"}
        )

        with patch.object(
            self.client.session,
            "post",
            return_value=response,
        ):
            with self.assertRaises(OllamaResponseError):
                self.client.chat([
                    {"role": "user", "content": "Hello"}
                ])

    def test_empty_response_content(self):
        response = self.make_response(
            data={"message": {"content": ""}}
        )

        with patch.object(
            self.client.session,
            "post",
            return_value=response,
        ):
            with self.assertRaises(OllamaResponseError):
                self.client.chat([
                    {"role": "user", "content": "Hello"}
                ])

    def test_stream_chat_yields_chunks(self):
        response = self.make_response()
        response.iter_lines.return_value = [
            b'{"message": {"content": "Hel"}}',
            b'{"message": {"content": "lo"}}',
            b'{"done": true}',
        ]

        with patch.object(
            self.client.session,
            "post",
            return_value=response,
        ):
            chunks = list(self.client.stream_chat([
                {"role": "user", "content": "Hello"}
            ]))

        self.assertEqual(chunks, ["Hel", "lo"])

    def test_stream_chat_stops_on_error_line(self):
        response = self.make_response()
        response.iter_lines.return_value = [
            b'{"error": "model is loading"}',
        ]

        with patch.object(
            self.client.session,
            "post",
            return_value=response,
        ):
            with self.assertRaises(OllamaResponseError):
                list(self.client.stream_chat([
                    {"role": "user", "content": "Hello"}
                ]))

    def test_payload_contains_num_predict(self):
        with patch(
            "core.ollama_client.CONFIG",
            SimpleNamespace(
                chat_url="http://localhost:11434/api/chat",
                model_name="starmentai",
                temperature=0.7,
                top_p=0.9,
                num_ctx=2048,
                num_thread=4,
                num_predict=512,
                think=False,
                keep_alive="5m",
                system_prompt="",
                style_reminder="",
            ),
        ):
            client = OllamaClient()

            payload = client._build_payload([
                {"role": "user", "content": "Hello"}
            ])

        self.assertEqual(payload["options"]["num_predict"], 512)

        client.close()

    def test_payload_prepends_system_and_appends_reminder(self):
        with patch(
            "core.ollama_client.CONFIG",
            SimpleNamespace(
                chat_url="http://localhost:11434/api/chat",
                model_name="starmentai",
                temperature=0.7,
                top_p=0.9,
                num_ctx=2048,
                num_thread=4,
                num_predict=512,
                think=False,
                keep_alive="5m",
                system_prompt="Bạn là StarmentAI.",
                style_reminder="Nhắc lại: giữ giọng điệu.",
            ),
        ):
            client = OllamaClient()

            payload = client._build_payload([
                {"role": "user", "content": "Hello"}
            ])

        roles = [m["role"] for m in payload["messages"]]

        self.assertEqual(roles, ["system", "user"])
        self.assertEqual(
            payload["messages"][0]["content"],
            "Bạn là StarmentAI.",
        )
        self.assertTrue(
            payload["messages"][-1]["content"].endswith(
                "\n\nNhắc lại: giữ giọng điệu."
            )
        )

        client.close()

    def test_payload_keeps_existing_system_message(self):
        with patch(
            "core.ollama_client.CONFIG",
            SimpleNamespace(
                chat_url="http://localhost:11434/api/chat",
                model_name="starmentai",
                temperature=0.7,
                top_p=0.9,
                num_ctx=2048,
                num_thread=4,
                num_predict=512,
                think=False,
                keep_alive="5m",
                system_prompt="Bạn là StarmentAI.",
                style_reminder="Nhắc lại: giữ giọng điệu.",
            ),
        ):
            client = OllamaClient()

            payload = client._build_payload([
                {"role": "system", "content": "Tùy chỉnh."},
                {"role": "user", "content": "Hello"},
            ])

        self.assertEqual(
            payload["messages"][0]["content"],
            "Tùy chỉnh.",
        )
        self.assertEqual(len(payload["messages"]), 2)

        client.close()

    def test_payload_skips_prompt_extras_when_unconfigured(self):
        with patch(
            "core.ollama_client.CONFIG",
            SimpleNamespace(
                chat_url="http://localhost:11434/api/chat",
                model_name="starmentai",
                temperature=0.7,
                top_p=0.9,
                num_ctx=2048,
                num_thread=4,
                num_predict=512,
                think=False,
                keep_alive="5m",
                system_prompt="",
                style_reminder="",
            ),
        ):
            client = OllamaClient()

            payload = client._build_payload([
                {"role": "user", "content": "Hello"}
            ])

        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(
            payload["messages"][0]["content"],
            "Hello",
        )

        client.close()


class TestSystemPromptConfig(unittest.TestCase):
    def test_config_parses_system_prompt_from_modelfile(self):
        self.assertIn("StarmentAI", CONFIG.system_prompt)
        self.assertIn("GIỌNG ĐIỆU", CONFIG.system_prompt)

    def test_load_system_prompt_missing_file_returns_empty(self):
        self.assertEqual(
            _load_system_prompt("khong-ton-tai.modelfile"),
            "",
        )


class TestPayloadThinkMode(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def tearDown(self):
        self.client.close()

    def test_think_override(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "Hello"}],
            think=True,
        )

        self.assertTrue(payload["think"])

        payload = self.client._build_payload(
            [{"role": "user", "content": "Hello"}],
            think=False,
        )

        self.assertFalse(payload["think"])

    def test_think_default_uses_config(self):
        payload = self.client._build_payload(
            [{"role": "user", "content": "Hello"}]
        )

        self.assertEqual(payload["think"], CONFIG.think)


class TestQuestionNeedsThinking(unittest.TestCase):
    def test_greeting_skips_thinking(self):
        self.assertFalse(question_needs_thinking("Xin chào StarmentAI!"))

    def test_thanks_skips_thinking(self):
        self.assertFalse(question_needs_thinking("Cảm ơn nha"))

    def test_simple_math_skips_thinking(self):
        self.assertFalse(question_needs_thinking("17*23=?"))

    def test_reasoning_needs_thinking(self):
        self.assertTrue(
            question_needs_thinking("Vì sao bầu trời màu xanh?")
        )

    def test_coding_needs_thinking(self):
        self.assertTrue(
            question_needs_thinking(
                "Giúp t viết hàm fibonacci bằng python"
            )
        )

    def test_long_question_needs_thinking(self):
        self.assertTrue(question_needs_thinking("chi tiết " * 40))

    def test_two_questions_need_thinking(self):
        self.assertTrue(
            question_needs_thinking("AI là gì? Nó hoạt động ra sao?")
        )

    def test_empty_skips_thinking(self):
        self.assertFalse(question_needs_thinking("   "))


if __name__ == "__main__":
    unittest.main()
