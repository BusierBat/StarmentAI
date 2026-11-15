
"""
StarmentAI - Context Manager
Controls how much conversation history is sent to the model.
"""

from config import CONFIG


class ContextManager:
    def __init__(
        self,
        context_limit: int | None = None,
        reserved_tokens: int | None = None,
        chars_per_token: float = 2.5,
    ):
        self.context_limit = (
            context_limit
            if context_limit is not None
            else CONFIG.num_ctx
        )

        self.reserved_tokens = (
            reserved_tokens
            if reserved_tokens is not None
            else max(256, self.context_limit // 2)
        )

        self.chars_per_token = chars_per_token

        if self.context_limit <= 0:
            raise ValueError(
                "Context limit must be greater than zero."
            )

        if self.reserved_tokens < 0:
            raise ValueError(
                "Reserved tokens cannot be negative."
            )

        if self.reserved_tokens >= self.context_limit:
            raise ValueError(
                "Reserved tokens must be smaller than context limit."
            )

        if self.chars_per_token <= 0:
            raise ValueError(
                "Characters per token must be greater than zero."
            )

        self.message_budget = (
            self.context_limit - self.reserved_tokens
        )

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token usage from message content."""

        total_characters = sum(
            len(message.get("content", ""))
            for message in messages
        )

        return int(
            total_characters / self.chars_per_token + 0.999
        )

    def build_context(
        self,
        history: list[dict],
    ) -> list[dict]:
        """
        Return a reduced copy of conversation history.

        Newer messages have higher priority.
        Original history is never modified.
        """

        if not isinstance(history, list):
            raise ValueError(
                "History must be a list."
            )

        valid_messages = []

        for message in history:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")

            if role not in ("user", "assistant"):
                continue

            if not isinstance(content, str):
                continue

            valid_messages.append({
                "role": role,
                "content": content,
            })

        selected = []

        # Process messages from newest to oldest.
        for message in reversed(valid_messages):
            candidate = [message] + selected

            if (
                self.estimate_tokens(candidate)
                > self.message_budget
            ):
                # Always preserve the newest message.
                if not selected:
                    selected = [message]

                break

            selected = candidate

        return selected

    def get_context_usage(
        self,
        messages: list[dict],
    ) -> dict:
        """Return estimated context usage information."""

        estimated = self.estimate_tokens(messages)

        return {
            "estimated_tokens": estimated,
            "message_budget": self.message_budget,
            "context_limit": self.context_limit,
            "reserved_tokens": self.reserved_tokens,
            "remaining_budget": max(
                0,
                self.message_budget - estimated,
            ),
        }

    def clear(self) -> None:
        """Reset the context manager."""
        # No persistent state is maintained.
        pass
