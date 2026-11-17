"""
StarmentAI - Intent Classification
Determines when to use system time and web search.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final


class IntentType(Enum):
    """Types of user intent."""
    GREETING = auto()
    SIMPLE_QUESTION = auto()
    TIME_RELATED = auto()
    SEARCH_NEEDED = auto()
    TECHNICAL = auto()
    CREATIVE = auto()
    CONVERSATIONAL = auto()
    UNKNOWN = auto()


class SearchTrigger(Enum):
    """Reasons to trigger web search."""
    CURRENT_EVENTS = auto()
    WEATHER = auto()
    PRICES = auto()
    SOFTWARE_VERSIONS = auto()
    TECH_NEWS = auto()
    PERSON_ORG_EVENT = auto()
    FACTUAL_LOOKUP = auto()
    NOT_TRIGGERED = auto()


class TimeTrigger(Enum):
    """Reasons to use system time."""
    CURRENT_DATE = auto()
    CURRENT_TIME = auto()
    TIMEZONE = auto()
    WEEKDAY = auto()
    RELATIVE_TIME = auto()
    NOT_TRIGGERED = auto()


@dataclass(frozen=True, slots=True)
class IntentAnalysis:
    """Result of intent analysis."""
    intent_type: IntentType
    needs_time: bool
    time_triggers: tuple[TimeTrigger, ...]
    needs_search: bool
    search_triggers: tuple[SearchTrigger, ...]
    confidence: float
    reasoning: str


# Patterns for time-related queries
TIME_PATTERNS: Final[list[tuple[re.Pattern, TimeTrigger]]] = [
    (re.compile(r"\b(bây giờ|hiện tại|hôm nay|ngày mai|ngày kia|tuần này|tháng này|năm nay)\b", re.IGNORECASE), TimeTrigger.CURRENT_DATE),
    (re.compile(r"\b(mấy giờ|giờ hiện tại|giờ là|what time|current time)\b", re.IGNORECASE), TimeTrigger.CURRENT_TIME),
    (re.compile(r"\b(múi giờ|timezone|giờ ở|giờ tại|giờ.*khác|giờ.*nước ngoài)\b", re.IGNORECASE), TimeTrigger.TIMEZONE),
    (re.compile(r"\b(thứ gì|thứ mấy|ngày.*thứ|what day|weekday)\b", re.IGNORECASE), TimeTrigger.WEEKDAY),
    (re.compile(r"\b(hôm qua|ngày mai|tuần sau|tháng sau|năm sau|trước|sau)\b", re.IGNORECASE), TimeTrigger.RELATIVE_TIME),
]

# Patterns for search-needed queries
SEARCH_PATTERNS: Final[list[tuple[re.Pattern, SearchTrigger]]] = [
    # Current events / news
    (re.compile(r"\b(tin tức|tin mới|báo mới|news|mới nhất|vừa qua|vừa xong|breaking)\b", re.IGNORECASE), SearchTrigger.CURRENT_EVENTS),
    # Weather
    (re.compile(r"\b(thời tiết|dự báo|mưa|nắng|nhiệt độ|weather|forecast)\b", re.IGNORECASE), SearchTrigger.WEATHER),
    # Prices
    (re.compile(r"\b(giá|bao nhiêu tiền|giá cả|price|cost|giá hiện tại|giá mới)\b", re.IGNORECASE), SearchTrigger.PRICES),
    # Software versions
    (re.compile(r"\b(phiên bản|version|phiên bản mới|latest version|update|bản cập nhật)\b", re.IGNORECASE), SearchTrigger.SOFTWARE_VERSIONS),
    # Tech news
    (re.compile(r"\b(công nghệ mới|tech news|AI mới|model mới|phát hành|release|launch)\b", re.IGNORECASE), SearchTrigger.TECH_NEWS),
    # Person/Org/Event - more specific patterns
    (re.compile(r"\b(ceo|cto|chairman|founder|giám đốc|người sáng lập)\b", re.IGNORECASE), SearchTrigger.PERSON_ORG_EVENT),
    (re.compile(r"\b(tổ chức|công ty|company|startup|tập đoàn|hiệp hội)\b", re.IGNORECASE), SearchTrigger.PERSON_ORG_EVENT),
    (re.compile(r"\b(sự kiện|event|hội nghị|conference|giải đấu|cuộc thi)\b", re.IGNORECASE), SearchTrigger.PERSON_ORG_EVENT),
    (re.compile(r"(ai là|là ai)\b.*\b(ceo|cto|founder|giám đốc|chairman|tổ chức|công ty)\b", re.IGNORECASE), SearchTrigger.PERSON_ORG_EVENT),
    # Factual lookup
    (re.compile(r"\b(là gì|khái niệm|định nghĩa|definition|tìm hiểu|nghiên cứu)\b", re.IGNORECASE), SearchTrigger.FACTUAL_LOOKUP),
]

# Patterns for simple questions that DON'T need search
SIMPLE_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(r"\b(chào|hello|hi|hey|alo|yo)\b", re.IGNORECASE),
    re.compile(r"^(cảm ơn|thanks|thank you)\b", re.IGNORECASE),
    re.compile(r"^(tạm biệt|bye|goodbye)\b", re.IGNORECASE),
    re.compile(r"^(ok|okay|ừ|uhm|aha)\b", re.IGNORECASE),
    re.compile(r"^[\d\s+\-*/x×÷().,=%?]+$"),  # Math
    re.compile(r"(là ai|who are you|what are you)\b", re.IGNORECASE),
]

# Patterns for technical questions
TECH_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(r"\b(code|lập trình|programming|python|javascript|java|sql|api|database|algorithm|function|class|bug|debug|error)\b", re.IGNORECASE),
    re.compile(r"\b(cách|how to|hướng dẫn|tutorial|giúp|help me)\b", re.IGNORECASE),
]

# Patterns for creative tasks
CREATIVE_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(r"\b(viết|sáng tác|creative|story|poem|thơ|truyện|email|bài viết)\b", re.IGNORECASE),
]


def is_simple_question(text: str) -> bool:
    """Check if question is simple and doesn't need search/time."""
    normalized = text.strip().lower()
    for pattern in SIMPLE_PATTERNS:
        if pattern.search(normalized):
            return True
    return False


def detect_time_triggers(text: str) -> list[TimeTrigger]:
    """Detect time-related triggers in text."""
    triggers = []
    normalized = text.lower()
    for pattern, trigger in TIME_PATTERNS:
        if pattern.search(normalized):
            triggers.append(trigger)
    return triggers


def detect_search_triggers(text: str) -> list[SearchTrigger]:
    """Detect search-related triggers in text."""
    triggers = []
    normalized = text.lower()
    for pattern, trigger in SEARCH_PATTERNS:
        if pattern.search(normalized):
            triggers.append(trigger)
    return triggers


_QUESTION_WORDS: Final[frozenset[str]] = frozenset({
    "t\u1ea1i sao", "v\u00ec sao", "l\u00e0m sao", "nh\u01b0 th\u1ebf n\u00e0o",
    "gi\u1ea3i th\u00edch", "so s\u00e1nh", "kh\u00e1c nhau", "why", "how",
    "explain", "compare",
})


def _classify(
    text: str,
    normalized: str,
    time_triggers: list[TimeTrigger],
    search_triggers: list[SearchTrigger],
) -> IntentType:
    """Classify intent from already-computed triggers (single pass)."""
    if not normalized:
        return IntentType.UNKNOWN

    if is_simple_question(text):
        return IntentType.GREETING if any(
            p.search(normalized) for p in SIMPLE_PATTERNS[:4]
        ) else IntentType.SIMPLE_QUESTION

    # Search triggers take priority over time triggers
    # (e.g., "gi\u00e1 v\u00e0ng h\u00f4m nay").
    if search_triggers:
        return IntentType.SEARCH_NEEDED

    if time_triggers:
        return IntentType.TIME_RELATED

    for pattern in TECH_PATTERNS:
        if pattern.search(normalized):
            return IntentType.TECHNICAL

    for pattern in CREATIVE_PATTERNS:
        if pattern.search(normalized):
            return IntentType.CREATIVE

    if "?" in text or any(word in normalized for word in _QUESTION_WORDS):
        return IntentType.SIMPLE_QUESTION

    return IntentType.CONVERSATIONAL


def classify_intent(text: str) -> IntentType:
    """Classify the general intent of the question."""
    normalized = text.strip().lower()
    return _classify(
        text,
        normalized,
        detect_time_triggers(text),
        detect_search_triggers(text),
    )


def analyze_intent(text: str) -> IntentAnalysis:
    """
    Comprehensive intent analysis for a user query.

    Triggers are detected once and shared with the classifier, so a query is
    never scanned twice.

    Returns:
        IntentAnalysis with all relevant flags and triggers.
    """
    normalized = text.strip().lower()
    time_triggers = detect_time_triggers(text)
    search_triggers = detect_search_triggers(text)

    intent_type = _classify(text, normalized, time_triggers, search_triggers)

    needs_time = bool(time_triggers) or intent_type == IntentType.TIME_RELATED
    needs_search = bool(search_triggers) or intent_type == IntentType.SEARCH_NEEDED

    # Calculate confidence
    confidence = 0.5
    if time_triggers:
        confidence += 0.2
    if search_triggers:
        confidence += 0.2
    if intent_type in (IntentType.TIME_RELATED, IntentType.SEARCH_NEEDED):
        confidence += 0.1
    confidence = min(1.0, confidence)

    # Build reasoning
    reasoning_parts = [f"Loại ý định: {intent_type.name}"]
    if time_triggers:
        reasoning_parts.append(f"Cần thời gian: {', '.join(t.name for t in time_triggers)}")
    if search_triggers:
        reasoning_parts.append(f"Cần tìm kiếm: {', '.join(s.name for s in search_triggers)}")
    reasoning = "; ".join(reasoning_parts)

    return IntentAnalysis(
        intent_type=intent_type,
        needs_time=needs_time,
        time_triggers=tuple(time_triggers),
        needs_search=needs_search,
        search_triggers=tuple(search_triggers),
        confidence=confidence,
        reasoning=reasoning,
    )


def should_use_time(text: str) -> bool:
    """Quick check if system time should be provided."""
    return analyze_intent(text).needs_time


def should_search(text: str) -> bool:
    """Quick check if web search should be performed."""
    return analyze_intent(text).needs_search


def get_time_context() -> str:
    """Get formatted time context for model."""
    from core.system_clock import format_datetime_for_prompt
    return format_datetime_for_prompt()