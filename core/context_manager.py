
"""
StarmentAI - Context Manager
Controls how much conversation history is sent to the model.
"""

from math import ceil
from typing import Any

from config import CONFIG


class ContextManager:
    def __init__(
        self,
        context_limit: int | None = None,
        reserved_tokens: int | None = None,
        system_prompt_tokens: int | None = None,
        chars_per_token: float = 3.5,
        message_overhead: int = 6,
    ):
        self.context_limit = (
            context_limit
            if context_limit is not None
            else CONFIG.num_ctx
        )

        self.system_prompt_tokens = (
            system_prompt_tokens
            if system_prompt_tokens is not None
            else CONFIG.system_prompt_tokens
        )

        self.reserved_tokens = (
            reserved_tokens
            if reserved_tokens is not None
            else (
                CONFIG.num_predict
                if CONFIG.num_predict is not None
                else max(128, min(512, self.context_limit // 4))
            )
        )

        self.chars_per_token = chars_per_token
        self.message_overhead = message_overhead

        if self.context_limit <= 0:
            raise ValueError(
                "Context limit must be greater than zero."
            )

        if self.system_prompt_tokens < 0:
            raise ValueError(
                "System prompt tokens cannot be negative."
            )

        if self.reserved_tokens < 0:
            raise ValueError(
                "Reserved tokens cannot be negative."
            )

        if self.chars_per_token <= 0:
            raise ValueError(
                "Characters per token must be greater than zero."
            )

        if self.message_overhead < 0:
            raise ValueError(
                "Message overhead cannot be negative."
            )

        self.message_budget = (
            self.context_limit
            - self.reserved_tokens
            - self.system_prompt_tokens
        )

        if self.message_budget <= 0:
            raise ValueError(
                "Context budget exhausted by system prompt "
                "and reserved tokens."
            )

    def _estimate_message_tokens(self, message: dict[str, str]) -> int:
        content = message.get("content", "")
        role = message.get("role", "")
        characters = len(content) + len(role)
        return self.message_overhead + ceil(characters / self.chars_per_token)

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token usage from message content."""
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list.")

        total = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            total += self._estimate_message_tokens(
                {
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                }
            )
        return total

    def _normalize_history(
        self,
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        valid_messages: list[dict[str, str]] = []

        for message in history:
            if not isinstance(message, dict):
                continue

            role = message.get("role")
            content = message.get("content")

            if role not in ("user", "assistant"):
                continue

            if not isinstance(content, str):
                continue

            valid_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return valid_messages

    def build_context(
        self,
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """
        Return a reduced copy of conversation history.

        Selection is contiguous from the newest message so the model
        never sees gaps in the middle of the conversation, and the
        latest user message is always preserved. Original history is
        never modified.
        """
        if not isinstance(history, list):
            raise ValueError("History must be a list.")

        valid_messages = self._normalize_history(history)
        if not valid_messages:
            return []

        latest = valid_messages[-1]
        earlier = valid_messages[:-1]

        selected: list[dict[str, str]] = [latest]
        used = self._estimate_message_tokens(latest)

        if used > self.message_budget:
            return selected

        index = len(earlier) - 1
        while index >= 0:
            block: list[dict[str, str]]
            next_index: int

            if (
                index > 0
                and earlier[index - 1].get("role") == "user"
                and earlier[index].get("role") == "assistant"
            ):
                block = earlier[index - 1 : index + 1]
                next_index = index - 2
            else:
                block = [earlier[index]]
                next_index = index - 1

            block_tokens = sum(
                self._estimate_message_tokens(message)
                for message in block
            )

            if used + block_tokens > self.message_budget:
                break

            selected = [*block, *selected]
            used += block_tokens
            index = next_index

        return selected

    def get_context_usage(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return estimated context usage information."""
        estimated = self.estimate_tokens(messages)

        return {
            "estimated_tokens": estimated,
            "message_budget": self.message_budget,
            "context_limit": self.context_limit,
            "reserved_tokens": self.reserved_tokens,
            "system_prompt_tokens": self.system_prompt_tokens,
            "remaining_budget": max(
                0,
                self.message_budget - estimated,
            ),
            "exceeds_message_budget": (
                estimated > self.message_budget
            ),
        }

    def clear(self) -> None:
        """Reset the context manager."""
        return None


print("")