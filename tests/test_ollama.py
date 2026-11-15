
import unittest
from unittest.mock import Mock, patch

import requests

from core.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
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


if __name__ == "__main__":
    unittest.main()
