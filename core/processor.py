
"""
StarmentAI - Processor
Coordinates validation, context management and AI generation.
"""

from typing import Any

from core.conversation import Conversation, ConversationError
from core.context_manager import ContextManager
from core.ollama_client import OllamaError, OllamaClient
from core.validator import Validator, ValidationError


class ProcessorError(ValueError):
    """Base exception for processor errors."""


class Processor:
    """Main processing engine for StarmentAI."""

    def __init__(
        self,
        conversation: Conversation | None = None,
        context_manager: ContextManager | None = None,
        client: OllamaClient | None = None,
    ) -> None:

        if conversation is not None and client is not None:
            raise ValueError(
                "Provide either conversation or client, not both."
            )

        self.conversation = (
            conversation
            if conversation is not None
            else Conversation(client=client)
        )

        self.context_manager = (
            context_manager
            if context_manager is not None
            else ContextManager()
        )

        self.client = self.conversation.client
        self._closed = False

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @staticmethod
    def _validate_input(user_input: str) -> str:
        """Validate user input."""

        try:
            return Validator.validate_user_input(user_input)

        except ValidationError as exc:
            raise ProcessorError(
                f"Invalid user input: {exc}"
            ) from exc

    @staticmethod
    def _validate_response(response: str) -> str:
        """Validate generated AI response."""

        try:
            return Validator.validate_response(response)

        except ValidationError as exc:
            raise ProcessorError(
                f"Invalid AI response: {exc}"
            ) from exc

    # --------------------------------------------------
    # Context management
    # --------------------------------------------------

    def _build_context(
        self,
        user_input: str,
    ) -> list[dict[str, str]]:
        """Build context from conversation history."""

        history = self.conversation.history

        pending_history = [
            *history,
            {
                "role": "user",
                "content": user_input,
            },
        ]

        try:
            context = self.context_manager.build_context(
                pending_history
            )

        except Exception as exc:
            raise ProcessorError(
                f"Failed to build conversation context: {exc}"
            ) from exc

        if not isinstance(context, list) or not context:
            raise ProcessorError(
                "Context manager returned an empty or invalid context."
            )

        latest_message = context[-1]

        if (
            not isinstance(latest_message, dict)
            or latest_message.get("role") != "user"
            or latest_message.get("content") != user_input
        ):
            raise ProcessorError(
                "Latest user message was excluded from context."
            )

        return context

    # --------------------------------------------------
    # AI generation
    # --------------------------------------------------

    def _generate_response(
        self,
        context: list[dict[str, str]],
    ) -> str:
        """Generate an AI response using Ollama."""

        try:
            raw_response = self.client.chat(context)

        except OllamaError as exc:
            raise ProcessorError(
                f"AI generation failed: {exc}"
            ) from exc

        return self._validate_response(raw_response)

    # --------------------------------------------------
    # Main processing
    # --------------------------------------------------

    def process(self, user_input: str) -> dict[str, Any]:
        """
        Validate input, build context, generate response
        and store the completed exchange.
        """

        if self._closed:
            raise ProcessorError(
                "Processor is already closed."
            )

        # Step 1: Validate input.
        user_input = self._validate_input(user_input)

        # Step 2: Build context.
        context = self._build_context(user_input)

        # Step 3: Generate AI response.
        response = self._generate_response(context)

        # Step 4: Store conversation exchange.
        try:
            self.conversation.record_exchange(
                user_input,
                response,
            )

        except (ValueError, ConversationError) as exc:
            raise ProcessorError(
                f"Failed to save conversation: {exc}"
            ) from exc

        # Step 5: Calculate context usage.
        try:
            context_usage = (
                self.context_manager.get_context_usage(context)
            )

        except Exception as exc:
            context_usage = None

        # Step 6: Return result.
        return {
            "success": True,
            "response": response,
            "context": context_usage,
        }

    # --------------------------------------------------
    # Conversation management
    # --------------------------------------------------

    def clear_conversation(self) -> None:
        """Clear current conversation history."""

        if self._closed:
            raise ProcessorError(
                "Processor is already closed."
            )

        try:
            self.conversation.clear()

        except ConversationError as exc:
            raise ProcessorError(
                f"Failed to clear conversation: {exc}"
            ) from exc

    # --------------------------------------------------
    # Lifecycle management
    # --------------------------------------------------

    def close(self) -> None:
        """Release processor resources."""

        if self._closed:
            return

        self._closed = True
        self.conversation.close()

    def __enter__(self) -> "Processor":
        if self._closed:
            raise ProcessorError(
                "Processor is already closed."
            )

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()
