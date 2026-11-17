"""
StarmentAI - Core Package
Core modules for the StarmentAI assistant.
"""

from core.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaConnectionError,
    OllamaResponseError,
    question_needs_thinking,
)
from core.processor import Processor, ProcessorError
from core.context_manager import ContextManager
from core.conversation import Conversation, ConversationError
from core.validator import Validator, ValidationError
from core.token_counter import count_text, count_message, count_messages
from core.system_clock import SystemClock, DateTimeInfo, get_system_clock, get_current_datetime, format_datetime_for_prompt
from core.search_provider import SearchProvider, SearchProviderError, SearchResult, SearchResponse, SearchProviderRegistry
from core.web_search import DuckDuckGoProvider, WebSearch
from core.information_processor import InformationProcessor, ProcessedInformation, SourceReference, process_web_search
from core.source_manager import SourceManager, SourceCitation, SourceCollection, get_source_manager
from core.intent_classifier import (
    IntentType,
    SearchTrigger,
    TimeTrigger,
    IntentAnalysis,
    analyze_intent,
    should_use_time,
    should_search,
    get_time_context,
)

__all__ = [
    # Ollama
    "OllamaClient",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaResponseError",
    "question_needs_thinking",
    # Processor
    "Processor",
    "ProcessorError",
    # Context
    "ContextManager",
    # Conversation
    "Conversation",
    "ConversationError",
    # Validator
    "Validator",
    "ValidationError",
    # Token counter
    "count_text",
    "count_message",
    "count_messages",
    # System clock
    "SystemClock",
    "DateTimeInfo",
    "get_system_clock",
    "get_current_datetime",
    "format_datetime_for_prompt",
    # Search
    "SearchProvider",
    "SearchProviderError",
    "SearchResult",
    "SearchResponse",
    "SearchProviderRegistry",
    "DuckDuckGoProvider",
    "WebSearch",
    # Information processor
    "InformationProcessor",
    "ProcessedInformation",
    "SourceReference",
    "process_web_search",
    # Source manager
    "SourceManager",
    "SourceCitation",
    "SourceCollection",
    "get_source_manager",
    # Intent classifier
    "IntentType",
    "SearchTrigger",
    "TimeTrigger",
    "IntentAnalysis",
    "analyze_intent",
    "should_use_time",
    "should_search",
    "get_time_context",
]