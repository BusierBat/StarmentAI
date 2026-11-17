"""
StarmentAI - Web Search (DuckDuckGo HTML)
Free web search using DuckDuckGo HTML scraping.
"""

import threading
import time
import urllib.parse
from typing import Any

import requests
from bs4 import BeautifulSoup

from core.search_provider import (
    SearchProvider,
    SearchProviderError,
    SearchResult,
    SearchResponse,
)


def _make_soup(html: str) -> BeautifulSoup:
    """Build a soup with the fastest available parser (lxml preferred).

    lxml is a C extension and parses noticeably faster (and with lower CPU
    cost) than the pure-Python ``html.parser``. When lxml is not installed we
    fall back transparently, so behaviour and results are unchanged.
    """
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


class DuckDuckGoSearchError(SearchProviderError):
    """DuckDuckGo search specific errors."""


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo HTML scraping search provider."""

    name = "duckduckgo"
    base_url = "https://html.duckduckgo.com/html/"

    # CSS selectors for result parsing
    RESULT_SELECTOR = "div.result__body"
    TITLE_SELECTOR = "a.result__snippet, a.result__url, h2.result__title a"
    SNIPPET_SELECTOR = "a.result__snippet, div.result__snippet"
    URL_SELECTOR = "a.result__url, a.result__snippet"

    # Markers that identify a bot-challenge / soft-block page rather than
    # genuine search results.
    ANOMALY_MARKERS = (
        "bots use duckduckgo",
        "select all squares",
        "unusual traffic",
        "are you a robot",
        "verify you are human",
        "anomaly",
    )

    # Error codes surfaced to callers.
    ERROR_RATE_LIMITED = "SEARCH_RATE_LIMITED"
    ERROR_BLOCKED = "SEARCH_BLOCKED"

    # Request settings
    DEFAULT_TIMEOUT = 10.0
    DEFAULT_MAX_RESULTS = 10
    DEFAULT_MIN_INTERVAL = 2.0
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_results: int = DEFAULT_MAX_RESULTS,
        user_agent: str | None = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> None:
        self.timeout = timeout
        self.max_results = max_results
        self.min_interval = max(0.0, min_interval)
        self.user_agent = user_agent or self.USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def search(
        self,
        query: str,
        max_results: int | None = None,
        timeout: float | None = None,
    ) -> SearchResponse:
        """
        Perform a DuckDuckGo search.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            timeout: Request timeout in seconds.

        Returns:
            SearchResponse with results or error.
        """
        start_time = time.perf_counter()
        max_results = max_results or self.max_results
        timeout = timeout or self.timeout

        if not query or not query.strip():
            return SearchResponse(
                query=query,
                results=(),
                provider=self.name,
                search_time_ms=0,
                error="Empty query",
            )

        try:
            self._throttle()
            html = self._fetch_html(query, timeout)

            if self._is_blocked(html):
                search_time_ms = (time.perf_counter() - start_time) * 1000
                return SearchResponse(
                    query=query,
                    results=(),
                    provider=self.name,
                    search_time_ms=search_time_ms,
                    error=self.ERROR_RATE_LIMITED,
                )

            results = self._parse_results(html, max_results)
            search_time_ms = (time.perf_counter() - start_time) * 1000

            return SearchResponse(
                query=query,
                results=tuple(results),
                provider=self.name,
                search_time_ms=search_time_ms,
                total_results=len(results),
            )

        except requests.exceptions.Timeout:
            search_time_ms = (time.perf_counter() - start_time) * 1000
            return SearchResponse(
                query=query,
                results=(),
                provider=self.name,
                search_time_ms=search_time_ms,
                error="Request timed out",
            )

        except requests.exceptions.ConnectionError:
            search_time_ms = (time.perf_counter() - start_time) * 1000
            return SearchResponse(
                query=query,
                results=(),
                provider=self.name,
                search_time_ms=search_time_ms,
                error="Connection error",
            )

        except Exception as exc:
            search_time_ms = (time.perf_counter() - start_time) * 1000
            return SearchResponse(
                query=query,
                results=(),
                provider=self.name,
                search_time_ms=search_time_ms,
                error=f"Search failed: {exc}",
            )

    def _throttle(self) -> None:
        """Sleep, if needed, to respect the minimum interval between requests."""
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _is_blocked(self, html: str) -> bool:
        """Return True when the response is a bot-challenge/soft-block page."""
        if not html:
            return True
        lowered = html.lower()
        return any(marker in lowered for marker in self.ANOMALY_MARKERS)

    def _fetch_html(self, query: str, timeout: float) -> str:
        """Fetch search results HTML from DuckDuckGo."""
        params = {"q": query, "kl": "vn-vn"}  # Vietnam region for better VN results
        response = self.session.post(
            self.base_url,
            data=params,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text

    def _parse_results(self, html: str, max_results: int) -> list[SearchResult]:
        """Parse search results from HTML."""
        soup = _make_soup(html)
        results: list[SearchResult] = []

        for rank, result_div in enumerate(
            soup.select(self.RESULT_SELECTOR), 1
        ):
            if rank > max_results:
                break

            try:
                title = self._extract_title(result_div)
                url = self._extract_url(result_div)
                snippet = self._extract_snippet(result_div)

                if not title or not url:
                    continue

                source = self._extract_domain(url)
                results.append(
                    SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source=source,
                        rank=rank,
                    )
                )

            except Exception:
                # Skip malformed results
                continue

        return results

    def _extract_title(self, div: BeautifulSoup) -> str:
        """Extract title from result div."""
        # Try multiple selectors
        for selector in ["h2.result__title a", "a.result__snippet", "a.result__url"]:
            elem = div.select_one(selector)
            if elem and elem.get_text(strip=True):
                return self._clean_snippet(elem.get_text(strip=True))
        return ""

    def _extract_url(self, div: BeautifulSoup) -> str:
        """Extract URL from result div."""
        # Try to find the main link
        link = div.select_one("a.result__snippet, a.result__url, h2.result__title a")
        if link and link.get("href"):
            href = link["href"]
            # DuckDuckGo uses redirect URLs, extract actual URL
            if href.startswith("//duckduckgo.com/l/?uddg="):
                # Extract the actual URL from redirect
                parsed = urllib.parse.urlparse(href)
                params = urllib.parse.parse_qs(parsed.query)
                if "uddg" in params:
                    return params["uddg"][0]
            return href
        return ""

    def _extract_snippet(self, div: BeautifulSoup) -> str:
        """Extract snippet from result div."""
        snippet_elem = div.select_one("a.result__snippet, div.result__snippet")
        if snippet_elem:
            return self._clean_snippet(snippet_elem.get_text(strip=True))
        return ""

    def close(self) -> None:
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self) -> "DuckDuckGoProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class WebSearch:
    """High-level web search interface."""

    def __init__(
        self,
        provider_name: str | None = None,
        timeout: float = 10.0,
        max_results: int = 5,
    ) -> None:
        from core.search_provider import get_search_registry
        self.registry = get_search_registry()
        self.provider_name = provider_name
        self.timeout = timeout
        self.max_results = max_results

    def search(
        self,
        query: str,
        max_results: int | None = None,
        timeout: float | None = None,
    ) -> SearchResponse:
        """Perform a web search."""
        provider = self.registry.get(self.provider_name)
        return provider.search(
            query=query,
            max_results=max_results or self.max_results,
            timeout=timeout or self.timeout,
        )

    def search_and_format(
        self,
        query: str,
        max_results: int | None = None,
        max_snippet: int = 300,
    ) -> str:
        """Search and return formatted context for model."""
        response = self.search(query, max_results=max_results)
        return response.format_for_context(
            max_results=max_results or self.max_results,
            max_snippet=max_snippet,
        )


# Register DuckDuckGo as default provider
from core.search_provider import register_provider
register_provider(DuckDuckGoProvider(), default=True)