
"""
StarmentAI - Ollama Client
Optimized communication with the local Ollama API.
"""

import json
import threading
import time

from typing import Any, Iterator

import requests

from config import CONFIG


class OllamaError(Exception):
    """Base exception for Ollama errors."""


class OllamaConnectionError(OllamaError):
    """Connection or timeout error."""


class OllamaResponseError(OllamaError):
    """Invalid Ollama response."""


class OllamaClient:
    """Thread-safe Ollama API client."""

    MAX_RETRIES = 2
    RETRY_DELAY = 1.0

    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(self) -> None:
        self.url = CONFIG.chat_url
        self.model = CONFIG.model_name

        self.session = requests.Session()
        self._lock = threading.Lock()
        self._closed = False

    # ---------------- VALIDATION ----------------

    @classmethod
    def _validate_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("Messages must be a non-empty list.")

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Message {index} must be a dictionary."
                )

            if message.get("role") not in cls.VALID_ROLES:
                raise ValueError(
                    f"Invalid role in message {index}."
                )

            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Invalid content in message {index}."
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise OllamaConnectionError(
                "Ollama client is closed."
            )

    # ---------------- PAYLOAD ----------------

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        stream: bool = False,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": CONFIG.temperature,
            "top_p": CONFIG.top_p,
            "num_ctx": CONFIG.num_ctx,
            "num_thread": CONFIG.num_thread,
        }

        if CONFIG.num_predict is not None:
            options["num_predict"] = CONFIG.num_predict

        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": CONFIG.think,
            "keep_alive": CONFIG.keep_alive,
            "options": options,
        }

    # ---------------- HTTP REQUEST ----------------

    def _post(
        self,
        payload: dict[str, Any],
        stream: bool = False,
    ) -> requests.Response:
        timeout = (
            CONFIG.connect_timeout,
            CONFIG.read_timeout,
        )

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=timeout,
                    stream=stream,
                )

                if response.status_code >= 500:
                    if attempt < self.MAX_RETRIES:
                        response.close()
                        time.sleep(
                            self.RETRY_DELAY * (attempt + 1)
                        )
                        continue

                if not response.ok:
                    status = response.status_code

                    try:
                        data = response.json()
                        error = (
                            data.get("error", response.text)
                            if isinstance(data, dict)
                            else response.text
                        )
                    except ValueError:
                        error = response.text

                    response.close()

                    raise OllamaResponseError(
                        f"HTTP {status}: {error}"
                    )

                return response

            except requests.exceptions.Timeout as exc:
                if attempt == self.MAX_RETRIES:
                    raise OllamaConnectionError(
                        "Ollama request timed out."
                    ) from exc

            except requests.exceptions.ConnectionError as exc:
                if attempt == self.MAX_RETRIES:
                    raise OllamaConnectionError(
                        "Cannot connect to Ollama. "
                        "Check whether Ollama is running."
                    ) from exc

            except requests.exceptions.RequestException as exc:
                raise OllamaConnectionError(
                    f"Request failed: {exc}"
                ) from exc

            time.sleep(self.RETRY_DELAY * (attempt + 1))

        raise OllamaConnectionError(
            "Unable to complete Ollama request."
        )

    # ---------------- CHAT ----------------

    def chat(
        self,
        messages: list[dict[str, Any]],
    ) -> str:
        """Generate a complete AI response."""

        self._validate_messages(messages)

        payload = self._build_payload(messages)

        with self._lock:
            self._ensure_open()

            try:
                with self._post(payload) as response:
                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise OllamaResponseError(
                            "Invalid JSON response from Ollama."
                        ) from exc

                    if not isinstance(data, dict):
                        raise OllamaResponseError(
                            "Invalid response structure."
                        )

                    message = data.get("message")

                    if not isinstance(message, dict):
                        raise OllamaResponseError(
                            "Missing response message."
                        )

                    content = message.get("content")

                    if not isinstance(content, str):
                        raise OllamaResponseError(
                            "Invalid response content."
                        )

                    content = content.strip()

                    if not content:
                        raise OllamaResponseError(
                            "Ollama returned an empty response."
                        )

                    return content

            except requests.exceptions.RequestException as exc:
                raise OllamaConnectionError(
                    f"Failed to read response: {exc}"
                ) from exc

    # ---------------- STREAMING ----------------

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
    ) -> Iterator[str]:
        """Generate response chunks using Ollama streaming."""

        self._validate_messages(messages)

        payload = self._build_payload(
            messages,
            stream=True,
        )

        with self._lock:
            self._ensure_open()

            try:
                with self._post(
                    payload,
                    stream=True,
                ) as response:

                    for line in response.iter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                        except (ValueError, TypeError) as exc:
                            raise OllamaResponseError(
                                "Invalid streaming JSON."
                            ) from exc

                        if not isinstance(data, dict):
                            raise OllamaResponseError(
                                "Invalid streaming response."
                            )

                        if data.get("error"):
                            raise OllamaResponseError(
                                str(data["error"])
                            )

                        message = data.get("message", {})

                        if not isinstance(message, dict):
                            raise OllamaResponseError(
                                "Invalid streaming message."
                            )

                        content = message.get("content", "")

                        if not isinstance(content, str):
                            raise OllamaResponseError(
                                "Invalid streaming content."
                            )

                        if content:
                            yield content

                        if data.get("done") is True:
                            break

            except requests.exceptions.RequestException as exc:
                raise OllamaConnectionError(
                    f"Streaming failed: {exc}"
                ) from exc

    # ---------------- LIFECYCLE ----------------

    def close(self) -> None:
        """Close HTTP session safely."""

        with self._lock:
            if self._closed:
                return

            self._closed = True
            self.session.close()

    def __enter__(self) -> "OllamaClient":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()
