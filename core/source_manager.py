"""
StarmentAI - Source Manager
Manages source references and citations for information tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceCitation:
    """A citation for a piece of information."""
    id: str
    title: str
    url: str
    domain: str
    snippet: str
    retrieved_at: str
    relevance: float = 1.0  # 0.0 to 1.0

    def format_inline(self) -> str:
        """Format as inline citation."""
        return f"[{self.id}]"

    def format_full(self) -> str:
        """Format as full citation."""
        return f"[{self.id}] {self.title} — {self.domain} — {self.url}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "snippet": self.snippet,
            "retrieved_at": self.retrieved_at,
            "relevance": self.relevance,
        }


@dataclass
class SourceCollection:
    """Collection of sources for a response."""
    citations: list[SourceCitation] = field(default_factory=list)
    query: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, citation: SourceCitation) -> None:
        """Add a citation."""
        self.citations.append(citation)

    def get(self, citation_id: str) -> SourceCitation | None:
        """Get citation by ID."""
        for c in self.citations:
            if c.id == citation_id:
                return c
        return None

    def format_bibliography(self) -> str:
        """Format all citations as bibliography."""
        if not self.citations:
            return "Không có nguồn tham khảo."

        lines = ["NGUỒN THAM KHẢO:"]
        for c in self.citations:
            lines.append(f"  {c.format_full()}")
        return "\n".join(lines)

    def format_inline_citations(self, text: str) -> str:
        """Insert inline citations into text based on relevance."""
        # Simple approach: append citations at end
        if not self.citations:
            return text
        citation_str = " ".join(c.format_inline() for c in self.citations)
        return f"{text} {citation_str}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "query": self.query,
            "generated_at": self.generated_at,
        }

    def __len__(self) -> int:
        return len(self.citations)

    def __bool__(self) -> bool:
        return bool(self.citations)


class SourceManager:
    """Manages sources across the conversation."""

    def __init__(self, max_sources_per_response: int = 10) -> None:
        self.max_sources_per_response = max_sources_per_response
        self._current_sources: SourceCollection | None = None
        self._source_history: list[SourceCollection] = []

    def start_new_response(self, query: str = "") -> SourceCollection:
        """Start collecting sources for a new response."""
        self._current_sources = SourceCollection(query=query)
        return self._current_sources

    def add_source(
        self,
        title: str,
        url: str,
        domain: str,
        snippet: str = "",
        relevance: float = 1.0,
    ) -> SourceCitation | None:
        """Add a source to current collection."""
        if self._current_sources is None:
            self.start_new_response()

        if len(self._current_sources) >= self.max_sources_per_response:
            return None

        citation_id = f"src{len(self._current_sources) + 1}"
        citation = SourceCitation(
            id=citation_id,
            title=title,
            url=url,
            domain=domain,
            snippet=snippet,
            retrieved_at=datetime.now().isoformat(),
            relevance=relevance,
        )
        self._current_sources.add(citation)
        return citation

    def add_from_search_result(
        self,
        result: "SearchResult",
        relevance: float = 1.0,
    ) -> SourceCitation | None:
        """Add source from a search result."""
        from core.search_provider import SearchResult
        if not isinstance(result, SearchResult):
            return None
        return self.add_source(
            title=result.title,
            url=result.url,
            domain=result.source,
            snippet=result.snippet,
            relevance=relevance,
        )

    def finalize(self) -> SourceCollection | None:
        """Finalize current source collection and add to history."""
        if self._current_sources is None:
            return None

        # Sort by relevance
        self._current_sources.citations.sort(key=lambda c: -c.relevance)

        # Renumber IDs
        for i, citation in enumerate(self._current_sources.citations, 1):
            self._current_sources.citations[i - 1] = SourceCitation(
                id=f"src{i}",
                title=citation.title,
                url=citation.url,
                domain=citation.domain,
                snippet=citation.snippet,
                retrieved_at=citation.retrieved_at,
                relevance=citation.relevance,
            )

        self._source_history.append(self._current_sources)
        result = self._current_sources
        self._current_sources = None
        return result

    def get_current(self) -> SourceCollection | None:
        """Get current source collection."""
        return self._current_sources

    def get_history(self) -> list[SourceCollection]:
        """Get all source collections in history."""
        return list(self._source_history)

    def clear_history(self) -> None:
        """Clear source history."""
        self._source_history.clear()

    def format_for_model(self, collection: SourceCollection | None = None) -> str:
        """Format sources for model context."""
        if collection is None:
            collection = self._current_sources
        if collection is None or not collection:
            return ""
        return collection.format_bibliography()


# Global source manager
_default_manager: SourceManager | None = None


def get_source_manager() -> SourceManager:
    """Get the default source manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SourceManager()
    return _default_manager