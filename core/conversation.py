
"""
StarmentAI - Conversation Manager
Thread-safe conversation history and message exchange.
"""

import threading

from typing import Any

from config import CONFIG
from core.ollama_client import OllamaClient


class ConversationError(Exception):
    """Base exception for conversation errors."""


class Conversation:
    """Manage conversation history and Ollama communication."""

    def __init__(
        self,
        client: OllamaClient | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self._owns_client = client is None

        self._history: list[dict[str, str]] = []

        # Protect shared conversation state.
        self._lock = threading.RLock()

        # Serialize AI requests without blocking history reads.
        self._send_lock = threading.Lock()

        self._revision = 0
        self._closed = False

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a safe copy of conversation history."""

        with self._lock:
            return [message.copy() for message in self._history]

    @property
    def is_closed(self) -> bool:
        """Check whether this conversation is closed."""

        with self._lock:
            return self._closed

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @staticmethod
    def _validate_message(
        content: str,
        message_type: str,
    ) -> str:
        """Validate and normalize message content."""

        if not isinstance(content, str):
            raise ValueError(
                f"{message_type} must be a string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                f"{message_type} cannot be empty."
            )

        return content

    def _validate_user_input(self, user_input: str) -> str:
        """Validate user input and enforce length limits."""

        user_input = self._validate_message(
            user_input,
            "User input",
        )

        limit = CONFIG.max_user_input_length

        if len(user_input) > limit:
            raise ValueError(
                f"Input exceeds the maximum length of {limit} characters."
            )

        return user_input

    def _ensure_open(self) -> None:
        """Prevent operations on a closed conversation."""

        if self._closed:
            raise ConversationError(
                "Conversation is already closed."
            )

    # --------------------------------------------------
    # Message exchange
    # --------------------------------------------------

    def send_message(self, user_input: str) -> str:
        """Send a message to Ollama and store the exchange."""

        user_input = self._validate_user_input(user_input)

        with self._send_lock:
            with self._lock:
                self._ensure_open()

                # Snapshot history for the AI request.
                pending_messages = [
                    message.copy()
                    for message in self._history
                ]

                pending_messages.append({
                    "role": "user",
                    "content": user_input,
                })

                revision = self._revision

            # Network request runs outside the history lock.
            try:
                response = self.client.chat(pending_messages)

            except Exception as exc:
                raise ConversationError(
                    f"Failed to generate response: {exc}"
                ) from exc

            if not isinstance(response, str) or not response.strip():
                raise ConversationError(
                    "AI returned an empty or invalid response."
                )

            assistant_response = response.strip()

            with self._lock:
                self._ensure_open()

                # Detect history changes during generation.
                if self._revision != revision:
                    raise ConversationError(
                        "Conversation history changed during generation. "
                        "Please try again."
                    )

                self._record_exchange_unlocked(
                    user_input,
                    assistant_response,
                )

            return assistant_response

    def record_exchange(
        self,
        user_input: str,
        assistant_response: str,
        expected_revision: int | None = None,
    ) -> None:
        """Store a completed user-assistant exchange."""

        user_input = self._validate_user_input(user_input)

        assistant_response = self._validate_message(
            assistant_response,
            "Assistant response",
        )

        with self._lock:
            self._ensure_open()

            if (
                expected_revision is not None
                and self._revision != expected_revision
            ):
                raise ConversationError(
                    "Conversation history changed during generation. "
                    "Please try again."
                )

            self._record_exchange_unlocked(
                user_input,
                assistant_response,
            )

    # --------------------------------------------------
    # History management
    # --------------------------------------------------

    def _record_exchange_unlocked(
        self,
        user_input: str,
        assistant_response: str,
    ) -> None:
        """Record an exchange while holding the history lock."""

        self._history.extend([
            {
                "role": "user",
                "content": user_input,
            },
            {
                "role": "assistant",
                "content": assistant_response,
            },
        ])

        self._trim_history_unlocked()
        self._revision += 1

    def _trim_history_unlocked(self) -> None:
        """Trim history while preserving message pairs."""

        limit = CONFIG.max_history_messages

        # Each exchange contains two messages.
        effective_limit = (limit // 2) * 2

        if len(self._history) > effective_limit:
            excess = len(self._history) - effective_limit
            remove_count = ((excess + 1) // 2) * 2

            del self._history[:remove_count]

    def clear(self) -> None:
        """Clear conversation history."""

        with self._lock:
            self._ensure_open()

            self._history.clear()
            self._revision += 1

    # --------------------------------------------------
    # Lifecycle management
    # --------------------------------------------------

    def close(self) -> None:
        """Close the conversation and release owned resources."""

        with self._send_lock:
            with self._lock:
                if self._closed:
                    return

                self._closed = True
                self._revision += 1

            if self._owns_client:
                try:
                    self.client.close()
                except Exception as exc:
                    raise ConversationError(
                        f"Failed to close Ollama client: {exc}"
                    ) from exc

    def __enter__(self) -> "Conversation":
        with self._lock:
            self._ensure_open()

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()
