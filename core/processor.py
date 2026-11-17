"""
StarmentAI - Processor
Coordinates validation, context management and AI generation.
"""

from typing import Any, Iterator

from config import CONFIG
from core.conversation import Conversation, ConversationError
from core.context_manager import ContextManager
from core.ollama_client import (
    OllamaError,
    OllamaClient,
    question_needs_thinking,
)
from core.token_counter import count_text
from core.validator import Validator, ValidationError
from core.system_clock import SystemClock, SystemClockError
from core.intent_classifier import analyze_intent, IntentAnalysis
from core.web_search import WebSearch
from core.information_processor import process_web_search
from core.source_manager import get_source_manager


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
            else ContextManager(
                style_reminder_tokens=count_text(
                    CONFIG.style_reminder,
                    chars_per_token=CONFIG.chars_per_token,
                ),
                safety_margin=CONFIG.context_safety_margin,
            )
        )

        self.client = self.conversation.client
        self._closed = False
        self.last_context_usage: dict[str, Any] | None = None

        # Real-time data components
        self._system_clock = SystemClock(
            default_timezone=CONFIG.default_timezone
        )
        self._web_search = WebSearch(
            timeout=CONFIG.search_timeout,
            max_results=CONFIG.search_max_results,
        ) if CONFIG.enable_web_search else None
        self._source_manager = get_source_manager()

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

    def will_think(self, user_input: str) -> bool:
        """Decide the thinking mode for the next generation."""

        if not CONFIG.adaptive_think:
            return CONFIG.think

        return question_needs_thinking(user_input)

    # --------------------------------------------------
    # Real-time data
    # --------------------------------------------------

    def _get_time_context(self) -> str:
        """Get formatted system time for model context."""
        if not CONFIG.enable_system_time:
            return ""
        try:
            return self._system_clock.format_for_model()
        except SystemClockError:
            return (
                "CURRENT_DATETIME: UNAVAILABLE\n"
                "TIMEZONE: UNKNOWN\n"
                "UTC_OFFSET: UNKNOWN\n"
                "(Không thể xác định thời gian hệ thống lúc này. "
                "Hãy nói rõ với người dùng rằng hiện chưa xác định "
                "được chính xác ngày/giờ, không tự đoán.)"
            )

    _SEARCH_ERROR_REASONS = {
        "SEARCH_RATE_LIMITED": (
            "nh\u00e0 cung c\u1ea5p t\u00ecm ki\u1ebfm t\u1ea1m gi\u1edbi h\u1ea1n "
            "(rate limit), th\u1eed l\u1ea1i sau"
        ),
        "SEARCH_BLOCKED": (
            "nh\u00e0 cung c\u1ea5p t\u00ecm ki\u1ebfm ch\u1eb7n truy c\u1eadp "
            "t\u1ef1 \u0111\u1ed9ng"
        ),
        "Request timed out": "qu\u00e1 th\u1eddi gian ch\u1edd",
        "Connection error": "l\u1ed7i k\u1ebft n\u1ed1i m\u1ea1ng",
    }

    @classmethod
    def _humanize_search_error(cls, error: str | None) -> str:
        """Turn a provider error code into a short Vietnamese reason."""
        if not error:
            return "l\u1ed7i kh\u00f4ng x\u00e1c \u0111\u1ecbnh"
        return cls._SEARCH_ERROR_REASONS.get(error, error)

    def _search_unavailable_message(self, reason: str) -> str:
        """Honest notice used when web search cannot be completed."""
        return (
            f"Kh\u00f4ng th\u1ec3 t\u00ecm ki\u1ebfm l\u00fac n\u00e0y ({reason}). "
            "H\u00e3y n\u00f3i r\u00f5 v\u1edbi ng\u01b0\u1eddi d\u00f9ng r\u1eb1ng "
            "hi\u1ec7n ch\u01b0a l\u1ea5y \u0111\u01b0\u1ee3c d\u1eef li\u1ec7u "
            "t\u00ecm ki\u1ebfm v\u00e0 tuy\u1ec7t \u0111\u1ed1i kh\u00f4ng "
            "t\u1ef1 b\u1ecba th\u00f4ng tin."
        )

    def _search_no_results_message(self, query: str) -> str:
        """Honest notice used when a search returns no results."""
        return (
            f"Kh\u00f4ng t\u00ecm th\u1ea5y k\u1ebft qu\u1ea3 cho \"{query}\". "
            "H\u00e3y n\u00f3i r\u00f5 v\u1edbi ng\u01b0\u1eddi d\u00f9ng v\u00e0 "
            "kh\u00f4ng t\u1ef1 b\u1ecba th\u00f4ng tin."
        )

    def _perform_web_search(self, query: str) -> str:
        """Perform web search and return formatted context.

        On failure (network error, rate limit, no results) an honest notice is
        returned so the model tells the user the truth instead of fabricating
        data or silently falling back to stale knowledge.
        """
        if not CONFIG.enable_web_search or self._web_search is None:
            return ""

        try:
            search_response = self._web_search.search(query)
        except Exception:
            return self._search_unavailable_message(
                self._humanize_search_error(None)
            )

        if not search_response.success:
            return self._search_unavailable_message(
                self._humanize_search_error(search_response.error)
            )

        if not search_response.results:
            return self._search_no_results_message(query)

        try:
            # Process results
            processed = process_web_search(query, search_response)

            # Track sources for citations (real add_source path).
            self._source_manager.start_new_response(query)
            for source in processed.sources:
                self._source_manager.add_source(
                    title=source.title,
                    url=source.url,
                    domain=source.domain,
                    snippet=source.snippet,
                    relevance=processed.confidence,
                )
            self._source_manager.finalize()

            # Format for model context
            return processed.format_for_model(
                max_sources=CONFIG.search_max_results,
                max_snippet=CONFIG.search_max_snippet,
            )

        except Exception:
            # Unexpected processing failure: degrade gracefully.
            return ""

    def _build_enhanced_context(
        self,
        user_input: str,
        intent: IntentAnalysis,
    ) -> list[dict[str, str]]:
        """Build context with real-time data injection."""
        # Get base context from history
        history = self.conversation.history

        pending_history = [
            *history,
            {
                "role": "user",
                "content": user_input,
            },
        ]

        try:
            context = self.context_manager.build_context(pending_history)

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

        # Inject real-time data as system message at the beginning
        system_parts = []

        if CONFIG.system_prompt:
            system_parts.append(CONFIG.system_prompt)

        # Add time context if needed
        if intent.needs_time:
            time_context = self._get_time_context()
            if time_context:
                system_parts.append(f"THÔNG TIN THỜI GIAN THỰC:\n{time_context}")

        # Add search context if needed
        if intent.needs_search:
            search_context = self._perform_web_search(user_input)
            if search_context:
                system_parts.append(f"THÔNG TIN TÌM KIẾM:\n{search_context}")

        # If we have real-time data, prepend as system message
        if system_parts:
            system_content = "\n\n".join(system_parts)
            # Check if first message is already system
            if context and context[0].get("role") == "system":
                context[0] = {"role": "system", "content": system_content}
            else:
                context.insert(0, {"role": "system", "content": system_content})

        try:
            self.last_context_usage = (
                self.context_manager.get_context_usage(context)
            )

        except Exception:
            self.last_context_usage = None

        return context

    # --------------------------------------------------
    # AI generation
    # --------------------------------------------------

    def _generate_response(
        self,
        context: list[dict[str, str]],
        think: bool,
    ) -> str:
        """Generate an AI response using Ollama."""

        try:
            raw_response = self.client.chat(context, think=think)

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

        user_input = self._validate_input(user_input)

        revision = self.conversation.revision

        think = self.will_think(user_input)

        # Analyze intent for real-time data needs
        intent = analyze_intent(user_input)

        # Build context with real-time data
        context = self._build_enhanced_context(user_input, intent)

        response = self._generate_response(context, think)

        try:
            self.conversation.record_exchange(
                user_input,
                response,
                expected_revision=revision,
            )

        except (ValueError, ConversationError) as exc:
            raise ProcessorError(
                f"Failed to save conversation: {exc}"
            ) from exc

        try:
            context_usage = (
                self.context_manager.get_context_usage(context)
                if self.last_context_usage is None
                else self.last_context_usage
            )

        except Exception:
            context_usage = None

        return {
            "success": True,
            "response": response,
            "context": context_usage,
            "intent": {
                "type": intent.intent_type.name,
                "needs_time": intent.needs_time,
                "needs_search": intent.needs_search,
                "confidence": intent.confidence,
            },
        }

    def process_stream(
        self,
        user_input: str,
    ) -> Iterator[str]:

        if self._closed:
            raise ProcessorError(
                "Processor is already closed."
            )

        user_input = self._validate_input(user_input)

        revision = self.conversation.revision

        think = self.will_think(user_input)

        # Analyze intent for real-time data needs
        intent = analyze_intent(user_input)

        # Build context with real-time data
        context = self._build_enhanced_context(user_input, intent)

        chunks: list[str] = []

        try:
            for chunk in self.client.stream_chat(
                context,
                think=think,
            ):
                chunks.append(chunk)
                yield chunk

        except OllamaError as exc:
            raise ProcessorError(
                f"AI generation failed: {exc}"
            ) from exc

        response = self._validate_response("".join(chunks))

        try:
            self.conversation.record_exchange(
                user_input,
                response,
                expected_revision=revision,
            )

        except (ValueError, ConversationError) as exc:
            raise ProcessorError(
                f"Failed to save conversation: {exc}"
            ) from exc

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
        if self._web_search:
            # WebSearch doesn't have close, but provider might
            pass

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