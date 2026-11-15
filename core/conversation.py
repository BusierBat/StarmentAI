
"""
StarmentAI - Conversation Manager
Manages conversation history and message exchange.
"""

import threading

from config import CONFIG
from core.ollama_client import OllamaClient


class ConversationError(Exception):
    """Base exception for conversation errors."""


class Conversation:
    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient()
        self._owns_client = client is None

        self._history: list[dict[str, str]] = []
        self._lock = threading.Lock()

    @property
    def history(self) -> list[dict[str, str]]:
        """Return a copy of the conversation history."""

        with self._lock:
            return [
                message.copy()
                for message in self._history
            ]

    def send_message(self, user_input: str) -> str:
        """Send a user message and return the AI response."""

        if not isinstance(user_input, str):
            raise ValueError(
                "User input must be a string."
            )

        user_input = user_input.strip()

        if not user_input:
            raise ValueError(
                "User input cannot be empty."
            )

        if len(user_input) > CONFIG.max_user_input_length:
            raise ValueError(
                f"Input exceeds the limit of "
                f"{CONFIG.max_user_input_length} characters."
            )

        with self._lock:
            user_message = {
                "role": "user",
                "content": user_input,
            }

            pending_messages = (
                self._history + [user_message]
            )

            try:
                response = self.client.chat(
                    pending_messages
                )

            except Exception as exc:
                raise ConversationError(
                    f"Failed to generate response: {exc}"
                ) from exc

            if (
                not isinstance(response, str)
                or not response.strip()
            ):
                raise ConversationError(
                    "AI returned an empty or invalid response."
                )

            assistant_response = response.strip()

            self._record_exchange_unlocked(
                user_input,
                assistant_response,
            )

            return assistant_response

    def record_exchange(
        self,
        user_input: str,
        assistant_response: str,
    ) -> None:
        """Store a completed user-assistant exchange."""

        if not isinstance(user_input, str):
            raise ValueError(
                "User input must be a string."
            )

        if not isinstance(assistant_response, str):
            raise ValueError(
                "Assistant response must be a string."
            )

        user_input = user_input.strip()
        assistant_response = assistant_response.strip()

        if not user_input or not assistant_response:
            raise ValueError(
                "Messages cannot be empty."
            )

        if len(user_input) > CONFIG.max_user_input_length:
            raise ValueError(
                f"Input exceeds the limit of "
                f"{CONFIG.max_user_input_length} characters."
            )

        with self._lock:
            self._record_exchange_unlocked(
                user_input,
                assistant_response,
            )

    def _record_exchange_unlocked(
        self,
        user_input: str,
        assistant_response: str,
    ) -> None:
        """Store an exchange while the conversation lock is held."""

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

        self._trim_history()

    def _trim_history(self) -> None:
        """Remove old messages while preserving conversation pairs."""

        limit = CONFIG.max_history_messages

        while len(self._history) > limit:
            if (
                len(self._history) >= 2
                and self._history[0]["role"] == "user"
                and self._history[1]["role"] == "assistant"
            ):
                del self._history[:2]
            else:
                self._history.pop(0)

    def clear(self) -> None:
        """Clear the current conversation."""

        with self._lock:
            self._history.clear()

    def close(self) -> None:
        """Close the client if this instance owns it."""

        if self._owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
