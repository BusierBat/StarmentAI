"""
Tests for search provider and web search modules.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from core.search_provider import (
    SearchProvider,
    SearchProviderError,
    SearchResult,
    SearchResponse,
    SearchProviderRegistry,
    register_provider,
    get_provider,
    get_search_registry,
)
from core.web_search import DuckDuckGoProvider, DuckDuckGoSearchError, WebSearch
from core.information_processor import (
    InformationProcessor,
    ProcessedInformation,
    SourceReference,
    process_web_search,
)
from core.source_manager import (
    SourceManager,
    SourceCitation,
    SourceCollection,
    get_source_manager,
)


class TestSearchResult(unittest.TestCase):
    def test_search_result_creation(self):
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="example.com",
            rank=1,
        )
        self.assertEqual(result.title, "Test Title")
        self.assertEqual(result.url, "https://example.com")
        self.assertEqual(result.source, "example.com")
        self.assertEqual(result.rank, 1)

    def test_format_for_context(self):
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet content",
            source="example.com",
            rank=1,
        )
        formatted = result.format_for_context()
        self.assertIn("[1] Test Title", formatted)
        self.assertIn("URL: https://example.com", formatted)
        self.assertIn("Nội dung: Test snippet content", formatted)
        self.assertIn("Nguồn: example.com", formatted)


class TestSearchResponse(unittest.TestCase):
    def test_success_response(self):
        results = (
            SearchResult("Title 1", "https://ex1.com", "Snippet 1", "ex1.com", 1),
            SearchResult("Title 2", "https://ex2.com", "Snippet 2", "ex2.com", 2),
        )
        response = SearchResponse(
            query="test query",
            results=results,
            provider="test",
            search_time_ms=100.0,
            total_results=2,
        )
        self.assertTrue(response.success)
        self.assertEqual(len(response.results), 2)

    def test_error_response(self):
        response = SearchResponse(
            query="test",
            results=(),
            provider="test",
            search_time_ms=50.0,
            error="Connection failed",
        )
        self.assertFalse(response.success)
        self.assertEqual(response.error, "Connection failed")

    def test_format_for_context(self):
        results = (
            SearchResult("Title 1", "https://ex1.com", "Snippet 1", "ex1.com", 1),
        )
        response = SearchResponse(
            query="test query",
            results=results,
            provider="duckduckgo",
            search_time_ms=100.0,
        )
        formatted = response.format_for_context()
        self.assertIn("test query", formatted)
        self.assertIn("duckduckgo", formatted)
        self.assertIn("Title 1", formatted)


class TestSearchProviderRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = SearchProviderRegistry()

    def test_register_and_get(self):
        class TestProvider(SearchProvider):
            name = "test"
            base_url = "https://test.com"

            def search(self, query, max_results=10, timeout=10.0):
                return SearchResponse(query, (), "test", 0)

        provider = TestProvider()
        self.registry.register(provider)
        retrieved = self.registry.get("test")
        self.assertIs(provider, retrieved)

    def test_default_provider(self):
        class TestProvider(SearchProvider):
            name = "test"
            base_url = "https://test.com"

            def search(self, query, max_results=10, timeout=10.0):
                return SearchResponse(query, (), "test", 0)

        self.registry.register(TestProvider(), default=True)
        default = self.registry.get()
        self.assertEqual(default.name, "test")

    def test_list_providers(self):
        class P1(SearchProvider):
            name = "p1"
            base_url = "https://p1.com"
            def search(self, q, **kw): return SearchResponse(q, (), "p1", 0)
        class P2(SearchProvider):
            name = "p2"
            base_url = "https://p2.com"
            def search(self, q, **kw): return SearchResponse(q, (), "p2", 0)

        self.registry.register(P1())
        self.registry.register(P2())
        self.assertEqual(set(self.registry.list_providers()), {"p1", "p2"})


class TestDuckDuckGoProvider(unittest.TestCase):
    def setUp(self):
        self.provider = DuckDuckGoProvider(timeout=5.0, max_results=3)

    def tearDown(self):
        self.provider.close()

    def test_provider_properties(self):
        self.assertEqual(self.provider.name, "duckduckgo")
        self.assertEqual(self.provider.base_url, "https://html.duckduckgo.com/html/")

    @patch("core.web_search.requests.Session.post")
    def test_search_success(self, mock_post):
        # Mock HTML response
        mock_response = Mock()
        mock_response.text = """
        <html>
            <div class="result__body">
                <h2 class="result__title"><a class="result__snippet" href="https://example.com">Test Title</a></h2>
                <a class="result__snippet" href="https://example.com">Test snippet content</a>
            </div>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = self.provider.search("test query")

        self.assertTrue(response.success)
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0].title, "Test Title")
        self.assertEqual(response.results[0].url, "https://example.com")
        self.assertEqual(response.provider, "duckduckgo")

    @patch("core.web_search.requests.Session.post")
    def test_search_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        response = self.provider.search("test query")

        self.assertFalse(response.success)
        self.assertEqual(response.error, "Request timed out")

    @patch("core.web_search.requests.Session.post")
    def test_search_connection_error(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()

        response = self.provider.search("test query")

        self.assertFalse(response.success)
        self.assertEqual(response.error, "Connection error")

    def test_search_empty_query(self):
        response = self.provider.search("")
        self.assertFalse(response.success)
        self.assertEqual(response.error, "Empty query")


class TestInformationProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = InformationProcessor(max_sources=3)

    def test_process_empty_results(self):
        from core.search_provider import SearchResponse
        response = SearchResponse(
            query="test",
            results=(),
            provider="test",
            search_time_ms=100,
            error="No results",
        )
        info = self.processor.process_search_results("test", response)
        self.assertEqual(len(info.sources), 0)
        self.assertEqual(info.confidence, 0.0)

    def test_process_with_results(self):
        from core.search_provider import SearchResponse, SearchResult
        results = (
            SearchResult("Title 1", "https://example.com", "Snippet 1", "example.com", 1),
            SearchResult("Title 2", "https://github.com", "Snippet 2", "github.com", 2),
        )
        response = SearchResponse(
            query="test query",
            results=results,
            provider="test",
            search_time_ms=100,
        )
        info = self.processor.process_search_results("test query", response)
        self.assertEqual(len(info.sources), 2)
        self.assertGreater(info.confidence, 0.0)
        self.assertIn("Title 1", info.summary)

    def test_confidence_boost_trusted_domain(self):
        from core.search_provider import SearchResponse, SearchResult
        results = (
            SearchResult("Python Docs", "https://docs.python.org/3/", "Official Python docs", "docs.python.org", 1),
        )
        response = SearchResponse("test", results, "test", 100)
        info = self.processor.process_search_results("test", response)
        # Trusted domain should boost confidence
        self.assertGreater(info.confidence, 0.5)


class TestSourceManager(unittest.TestCase):
    def setUp(self):
        self.manager = SourceManager(max_sources_per_response=5)

    def test_start_new_response(self):
        collection = self.manager.start_new_response("test query")
        self.assertEqual(collection.query, "test query")
        self.assertEqual(len(collection), 0)

    def test_add_source(self):
        self.manager.start_new_response("test")
        citation = self.manager.add_source(
            title="Test",
            url="https://example.com",
            domain="example.com",
            snippet="Test snippet",
        )
        self.assertIsNotNone(citation)
        self.assertEqual(citation.id, "src1")
        self.assertEqual(citation.title, "Test")

    def test_add_from_search_result(self):
        from core.search_provider import SearchResult
        self.manager.start_new_response("test")
        result = SearchResult("Title", "https://ex.com", "Snippet", "ex.com", 1)
        citation = self.manager.add_from_search_result(result)
        self.assertIsNotNone(citation)
        self.assertEqual(citation.domain, "ex.com")

    def test_finalize(self):
        self.manager.start_new_response("test")
        self.manager.add_source("A", "https://a.com", "a.com")
        self.manager.add_source("B", "https://b.com", "b.com")
        collection = self.manager.finalize()
        self.assertEqual(len(collection), 2)
        self.assertEqual(collection.citations[0].id, "src1")
        self.assertEqual(collection.citations[1].id, "src2")

    def test_format_bibliography(self):
        collection = SourceCollection()
        collection.add(SourceCitation("src1", "Title", "https://ex.com", "ex.com", "Snippet", "2024-01-01"))
        formatted = collection.format_bibliography()
        self.assertIn("NGUỒN THAM KHẢO:", formatted)
        self.assertIn("Title", formatted)


class TestWebSearch(unittest.TestCase):
    @patch("core.web_search.DuckDuckGoProvider.search")
    def test_web_search_delegates(self, mock_search):
        from core.search_provider import SearchResponse, SearchResult
        mock_search.return_value = SearchResponse(
            "query", (SearchResult("T", "https://u", "S", "d", 1),), "ddg", 100
        )

        ws = WebSearch(provider_name="duckduckgo")
        response = ws.search("test query")

        self.assertTrue(response.success)
        mock_search.assert_called_once()


if __name__ == "__main__":
    unittest.main()