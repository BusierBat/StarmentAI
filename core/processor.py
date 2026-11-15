
"""
StarmentAI - Processor
Coordinates validation, context management and AI generation.
"""

from core.conversation import Conversation
from core.context_manager import ContextManager
from core.ollama_client import OllamaError, OllamaClient
from core.validator import Validator, ValidationError


class ProcessorError(Exception):
    """Base exception for processor errors."""


class Processor:
    def __init__(
        self,
        conversation: Conversation | None = None,
        context_manager: ContextManager | None = None,
        client: OllamaClient | None = None,
    ):
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

    def process(self, user_input: str) -> dict:
        """Validate input, generate a response and store the exchange."""

        # Step 1: Validate user input.
        try:
            user_input = Validator.validate_user_input(
                user_input
            )

        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

        # Step 2: Build conversation context.
        history = self.conversation.history

        pending_history = history + [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        context = self.context_manager.build_context(
            pending_history
        )

        if not context or context[-1]["role"] != "user":
            raise ProcessorError(
                "The latest user message was excluded from context."
            )

        # Step 3: Send request to Ollama.
        try:
            raw_response = self.client.chat(context)

        except OllamaError as exc:
            raise ProcessorError(
                f"AI generation failed: {exc}"
            ) from exc

        # Step 4: Validate AI response.
        try:
            response = Validator.validate_response(
                raw_response
            )

        except ValidationError as exc:
            raise ProcessorError(
                f"Invalid AI response: {exc}"
            ) from exc

        # Step 5: Store the completed exchange.
        try:
            self.conversation.record_exchange(
                user_input,
                response,
            )

        except ValueError as exc:
            raise ProcessorError(
                f"Failed to save conversation: {exc}"
            ) from exc

        # Step 6: Return processing result.
        return {
            "success": True,
            "response": response,
            "context": self.context_manager.get_context_usage(
                context
            ),
        }

    def clear_conversation(self) -> None:
        """Clear conversation history."""

        self.conversation.clear()

    def close(self) -> None:
        """Release owned resources."""

        self.conversation.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
