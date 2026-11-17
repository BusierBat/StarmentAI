"""
StarmentAI - Search Provider
Base classes and provider management for web search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse


class SearchProviderError(Exception):
    """Base exception for search provider errors."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str
    rank: int
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "rank": self.rank,
            "retrieved_at": self.retrieved_at,
        }

    def format_for_context(self, max_snippet: int = 300) -> str:
        """Format result for model context."""
        snippet = self.snippet[:max_snippet]
        if len(self.snippet) > max_snippet:
            snippet += "..."
        return f"[{self.rank}] {self.title}\nURL: {self.url}\nNội dung: {snippet}\nNguồn: {self.source}"


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Response from a search query."""
    query: str
    results: tuple[SearchResult, ...]
    provider: str
    search_time_ms: float
    total_results: int = 0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def format_for_context(self, max_results: int = 5, max_snippet: int = 300) -> str:
        """Format all results for model context."""
        if not self.results:
            return "Không tìm thấy kết quả."

        lines = [f"Kết quả tìm kiếm cho: \"{self.query}\" (nguồn: {self.provider})"]
        for result in self.results[:max_results]:
            lines.append(result.format_for_context(max_snippet))
        return "\n\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "provider": self.provider,
            "search_time_ms": self.search_time_ms,
            "total_results": self.total_results,
            "error": self.error,
        }


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Base URL for the search engine."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 10,
        timeout: float = 10.0,
    ) -> SearchResponse:
        """Perform a search and return results."""
        pass

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for source identification."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"

    def _clean_snippet(self, snippet: str) -> str:
        """Clean and normalize snippet text."""
        if not snippet:
            return ""
        # Remove extra whitespace
        cleaned = " ".join(snippet.split())
        # Remove common artifacts
        cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
        return cleaned.strip()


class SearchProviderRegistry:
    """Registry for managing search providers."""

    def __init__(self) -> None:
        self._providers: dict[str, SearchProvider] = {}
        self._default: str | None = None

    def register(self, provider: SearchProvider, default: bool = False) -> None:
        """Register a search provider."""
        self._providers[provider.name] = provider
        if default or self._default is None:
            self._default = provider.name

    def get(self, name: str | None = None) -> SearchProvider:
        """Get a provider by name, or the default."""
        if name is None:
            name = self._default
        if name is None:
            raise SearchProviderError("No search provider registered")
        provider = self._providers.get(name)
        if provider is None:
            raise SearchProviderError(f"Search provider '{name}' not found")
        return provider

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    @property
    def default_name(self) -> str | None:
        return self._default


# Global registry
_registry = SearchProviderRegistry()


def get_search_registry() -> SearchProviderRegistry:
    """Get the global search provider registry."""
    return _registry


def register_provider(provider: SearchProvider, default: bool = False) -> None:
    """Register a search provider globally."""
    _registry.register(provider, default)


def get_provider(name: str | None = None) -> SearchProvider:
    """Get a search provider by name or default."""
    return _registry.get(name)