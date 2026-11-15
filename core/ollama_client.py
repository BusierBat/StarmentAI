
"""
StarmentAI - Ollama Client
Handles communication with the local Ollama API.
"""

import threading

import requests

from config import CONFIG


class OllamaError(Exception):
    """Base exception for Ollama errors."""


class OllamaConnectionError(OllamaError):
    """Ollama connection or timeout error."""


class OllamaResponseError(OllamaError):
    """Invalid response from Ollama."""


class OllamaClient:
    def __init__(self):
        self.url = CONFIG.chat_url
        self.model = CONFIG.model_name

        self.session = requests.Session()

        # Prevent multiple simultaneous requests.
        self._request_lock = threading.Lock()

    def chat(self, messages: list[dict]) -> str:
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "messages must be a non-empty list"
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    "Every message must be a dictionary"
                )

            if message.get("role") not in (
                "system",
                "user",
                "assistant",
            ):
                raise ValueError(
                    "Invalid message role"
                )

            if not isinstance(message.get("content"), str):
                raise ValueError(
                    "Message content must be a string"
                )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": CONFIG.think,
            "keep_alive": CONFIG.keep_alive,
            "options": {
                "temperature": CONFIG.temperature,
                "top_p": CONFIG.top_p,
                "num_ctx": CONFIG.num_ctx,
                "num_thread": CONFIG.num_thread,
            },
        }

        timeout = (
            CONFIG.connect_timeout,
            CONFIG.read_timeout,
        )

        try:
            with self._request_lock:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=timeout,
                )

        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                "Ollama request timed out."
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                "Cannot connect to Ollama. "
                "Check whether Ollama is running."
            ) from exc

        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(
                f"Request failed: {exc}"
            ) from exc

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "error",
                    response.text,
                )
            except ValueError:
                error_message = response.text

            raise OllamaResponseError(
                f"HTTP {response.status_code}: "
                f"{error_message}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                "Ollama returned invalid JSON."
            ) from exc

        message = data.get("message")

        if not isinstance(message, dict):
            raise OllamaResponseError(
                "Response does not contain a valid message."
            )

        content = message.get("content")

        if not isinstance(content, str):
            raise OllamaResponseError(
                "Response content is missing or invalid."
            )

        return content.strip()

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
