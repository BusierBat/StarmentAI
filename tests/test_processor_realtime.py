"""
Regression tests for real-time data integration in the Processor:
- system time injection into the model context
- web-search context injection
- source/citation tracking (regression for the SearchResult bug)
- SystemClock preferred-timezone resolution
"""

import unittest

from config import CONFIG
from core.conversation import Conversation
from core.context_manager import ContextManager
from core.ollama_client import OllamaConnectionError
from core.processor import Processor
from core.source_manager import SourceManager
from core.search_provider import SearchResponse, SearchResult
from core.system_clock import SystemClock, HAS_ZONEINFO


class FakeClient:
    def __init__(self, response="StarmentAI response"):
        self.response = response
        self.should_fail = False
        self.stream_chunks = ["Hello", " ", "world"]
        self.chat_calls = []
        self.stream_calls = []
        self.last_think = None

    def chat(self, messages, think=None):
        self.last_think = think
        self.chat_calls.append(messages)
        if self.should_fail:
            raise OllamaConnectionError("Simulated connection failure")
        return self.response

    def stream_chat(self, messages, think=None):
        self.last_think = think
        self.stream_calls.append(messages)
        for chunk in self.stream_chunks:
            yield chunk

    def close(self):
        pass


class FakeSearch:
    """Stand-in for WebSearch that returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.queries = []

    def search(self, query, max_results=None, timeout=None):
        self.queries.append(query)
        return self.response


def _make_response(query="q", domains=(("Title", "https://ex.com", "Snippet", "ex.com"),)):
    results = tuple(
        SearchResult(title=t, url=u, snippet=s, source=d, rank=i + 1)
        for i, (t, u, s, d) in enumerate(domains)
    )
    return SearchResponse(
        query=query,
        results=results,
        provider="duckduckgo",
        search_time_ms=1.0,
        total_results=len(results),
    )


class RealtimeProcessorTestBase(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.conversation = Conversation(client=self.client)
        self.context_manager = ContextManager(
            context_limit=4096,
            reserved_tokens=512,
            system_prompt_tokens=0,
            style_reminder_tokens=0,
        )
        self.processor = Processor(
            conversation=self.conversation,
            context_manager=self.context_manager,
        )

    def tearDown(self):
        self.processor.close()


class TestTimeInjection(RealtimeProcessorTestBase):
    def test_time_question_injects_system_time(self):
        self.processor.process("H\u00f4m nay l\u00e0 th\u1ee9 m\u1ea5y?")

        messages = self.client.chat_calls[-1]
        self.assertEqual(messages[0]["role"], "system")
        content = messages[0]["content"]
        self.assertIn("TH\u00d4NG TIN TH\u1edcI GIAN TH\u1ef0C:", content)
        self.assertIn("CURRENT_DATETIME: 20", content)
        self.assertIn("TIMEZONE", content)
        self.assertIn("UTC_OFFSET", content)

    def test_non_time_question_omits_time_block(self):
        self.processor.process("2 + 2 = ?")

        messages = self.client.chat_calls[-1]
        # The persona prompt mentions CURRENT_DATETIME in its instructions, so
        # assert on the injected real-time block header instead, which only
        # appears when the clock is actually provided.
        joined = "\n".join(m.get("content", "") for m in messages)
        self.assertNotIn("TH\u00d4NG TIN TH\u1edcI GIAN TH\u1ef0C:", joined)
        self.assertNotIn("CURRENT_DATETIME: 20", joined)

    def test_time_context_uses_config_timezone(self):
        ctx = self.processor._get_time_context()
        self.assertTrue(ctx)
        self.assertIn("CURRENT_DATETIME:", ctx)
        self.assertIn("TIMEZONE:", ctx)
        # On this machine the OS reports UTC+7 (tzlocal -> Asia/Bangkok);
        # the resolver should prefer the configured Asia/Ho_Chi_Minh when
        # zoneinfo is available.
        if HAS_ZONEINFO and CONFIG.default_timezone == "Asia/Ho_Chi_Minh":
            self.assertEqual(self.processor._system_clock.default_timezone,
                             "Asia/Ho_Chi_Minh")


class TestSearchInjection(RealtimeProcessorTestBase):
    def test_search_question_injects_search_block(self):
        self.processor._web_search = FakeSearch(_make_response())
        self.processor.process("Gi\u00e1 v\u00e0ng h\u00f4m nay")

        messages = self.client.chat_calls[-1]
        content = messages[0]["content"]
        self.assertIn("TH\u00d4NG TIN T\u00ccM KI\u1ebeM", content)
        self.assertIn("ex.com", content)

    def test_search_tracks_sources(self):
        self.processor._source_manager = SourceManager()
        self.processor._web_search = FakeSearch(_make_response())

        ctx = self.processor._perform_web_search("gi\u00e1 v\u00e0ng")

        # Context returned to the model mentions the source.
        self.assertIn("ex.com", ctx)

        # A finalized collection with one citation is tracked (regression:
        # the previous implementation built a fake SearchResult and the
        # isinstance check silently dropped every source).
        history = self.processor._source_manager.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(len(history[0]), 1)
        self.assertEqual(history[0].citations[0].domain, "ex.com")

    def test_search_failure_returns_honest_note(self):
        failed = SearchResponse(
            query="q", results=(), provider="duckduckgo",
            search_time_ms=1.0, error="Connection error",
        )
        self.processor._web_search = FakeSearch(failed)

        ctx = self.processor._perform_web_search("q")
        # A non-empty, honest notice must be produced so the model tells the
        # user the truth instead of silently fabricating or going quiet.
        self.assertTrue(ctx)
        self.assertIn("l\u1ed7i k\u1ebft n\u1ed1i", ctx)
        self.assertIn("kh\u00f4ng t\u1ef1 b\u1ecba", ctx)

        # A full process() still succeeds even when search fails.
        result = self.processor.process("Gi\u00e1 v\u00e0ng h\u00f4m nay")
        self.assertTrue(result["success"])

    def test_rate_limited_search_reports_limit(self):
        limited = SearchResponse(
            query="q", results=(), provider="duckduckgo",
            search_time_ms=1.0, error="SEARCH_RATE_LIMITED",
        )
        self.processor._web_search = FakeSearch(limited)

        ctx = self.processor._perform_web_search("q")
        self.assertIn("gi\u1edbi h\u1ea1n", ctx)

    def test_no_results_reports_honestly(self):
        empty = SearchResponse(
            query="q", results=(), provider="duckduckgo", search_time_ms=1.0,
        )
        self.processor._web_search = FakeSearch(empty)

        ctx = self.processor._perform_web_search("q")
        self.assertIn("Kh\u00f4ng t\u00ecm th\u1ea5y", ctx)


class TestDuckDuckGoAnomalyDetection(unittest.TestCase):
    def test_challenge_page_is_reported_as_rate_limited(self):
        from unittest.mock import Mock, patch
        from core.web_search import DuckDuckGoProvider

        provider = DuckDuckGoProvider(min_interval=0)
        challenge_html = (
            "<html><body><h1>DuckDuckGo</h1>"
            "<p>Unfortunately, bots use DuckDuckGo too. Please complete the "
            "following challenge to confirm this search was made by a human. "
            "Select all squares containing a duck.</p></body></html>"
        )
        mock_response = Mock()
        mock_response.text = challenge_html
        mock_response.raise_for_status = Mock()

        try:
            with patch("core.web_search.requests.Session.post",
                       return_value=mock_response):
                resp = provider.search("test")
        finally:
            provider.close()

        self.assertFalse(resp.success)
        self.assertEqual(resp.error, DuckDuckGoProvider.ERROR_RATE_LIMITED)
        self.assertEqual(resp.total_results, 0)


class TestTimezoneResolution(unittest.TestCase):
    def setUp(self):
        self.clock = SystemClock(default_timezone="Asia/Ho_Chi_Minh")

    def test_prefers_configured_zone_on_offset_match(self):
        import datetime
        from zoneinfo import ZoneInfo

        aware = datetime.datetime.now(ZoneInfo("Asia/Bangkok"))
        resolved = self.clock._resolve_timezone_name("Asia/Bangkok", aware)
        if HAS_ZONEINFO:
            self.assertEqual(resolved, "Asia/Ho_Chi_Minh")
        else:
            self.assertEqual(resolved, "Asia/Bangkok")

    def test_keeps_detected_zone_on_offset_mismatch(self):
        import datetime
        from zoneinfo import ZoneInfo

        aware = datetime.datetime.now(ZoneInfo("UTC"))
        resolved = self.clock._resolve_timezone_name("UTC", aware)
        self.assertEqual(resolved, "UTC")

    def test_utc_default_is_never_replaced(self):
        import datetime
        from zoneinfo import ZoneInfo

        clock = SystemClock(default_timezone="UTC")
        aware = datetime.datetime.now(ZoneInfo("UTC"))
        self.assertEqual(clock._resolve_timezone_name("UTC", aware), "UTC")

    def test_none_aware_dt_returns_detected(self):
        self.assertEqual(
            self.clock._resolve_timezone_name("Asia/Bangkok", None),
            "Asia/Bangkok",
        )


if __name__ == "__main__":
    unittest.main()