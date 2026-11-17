"""
StarmentAI - Information Processor
Processes and normalizes data from various sources.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Reference to an information source."""
    title: str
    url: str
    domain: str
    retrieved_at: str = field(default_factory=lambda: datetime.now().isoformat())
    snippet: str = ""
    rank: int = 0

    def format_citation(self) -> str:
        """Format as a citation for model context."""
        return f"[{self.rank}] {self.title} ({self.domain}) - {self.url}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
            "retrieved_at": self.retrieved_at,
            "snippet": self.snippet,
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class ProcessedInformation:
    """Processed and normalized information package."""
    query: str
    sources: tuple[SourceReference, ...]
    summary: str
    raw_content: str
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.0  # 0.0 to 1.0

    def format_for_model(self, max_sources: int = 5, max_snippet: int = 200) -> str:
        """Format for model context injection."""
        if not self.sources:
            return f"Tìm kiếm: \"{self.query}\"\nKết quả: Không tìm thấy thông tin."

        lines = [
            f"THÔNG TIN TÌM KIẾM: \"{self.query}\"",
            f"Thời điểm: {self.processed_at}",
            f"Độ tin cậy ước lượng: {self.confidence:.0%}",
            "",
            "NGUỒN THAM KHẢO:",
        ]

        for source in self.sources[:max_sources]:
            snippet = source.snippet[:max_snippet]
            if len(source.snippet) > max_snippet:
                snippet += "..."
            lines.append(f"  {source.format_citation()}")
            if snippet:
                lines.append(f"    Trích dẫn: {snippet}")

        lines.extend(["", "TÓM TẮT:", self.summary])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sources": [s.to_dict() for s in self.sources],
            "summary": self.summary,
            "raw_content": self.raw_content,
            "processed_at": self.processed_at,
            "confidence": self.confidence,
        }


class InformationProcessor:
    """Processes raw search results into structured information."""

    # Patterns for detecting conflicting information
    CONFLICT_PATTERNS = [
        r"(tuy nhiên|nhưng|ngược lại|khác biệt|mâu thuẫn|không đồng ý)",
        r"(cũ|lỗi thời|đã thay đổi|được cập nhật)",
    ]

    # Trusted domains (official sources)
    TRUSTED_DOMAINS = {
        "github.com",
        "gitlab.com",
        "stackoverflow.com",
        "developer.mozilla.org",
        "docs.python.org",
        "docs.microsoft.com",
        "cloud.google.com",
        "aws.amazon.com",
        "kubernetes.io",
        "docker.com",
        "nodejs.org",
        "python.org",
        "rust-lang.org",
        "golang.org",
        "java.com",
        "oracle.com",
        "ibm.com",
        "redhat.com",
        "ubuntu.com",
        "debian.org",
        "archlinux.org",
        "wikipedia.org",
        "wikimedia.org",
        "britannica.com",
        "nasa.gov",
        "nih.gov",
        "who.int",
        "gov.vn",
        "chinhphu.vn",
        "vnexpress.net",
        "tuoitre.vn",
        "thanhnien.vn",
        "dantri.com.vn",
        "baomoi.com",
    }

    # Unreliable patterns
    UNRELIABLE_PATTERNS = [
        r"clickbait",
        r"quảng cáo|advertisement|sponsored",
        r"giả mạo|fake|scam",
    ]

    def __init__(
        self,
        max_sources: int = 5,
        min_confidence: float = 0.3,
        snippet_limit: int = 500,
    ) -> None:
        self.max_sources = max_sources
        self.min_confidence = min_confidence
        self.snippet_limit = snippet_limit

    def process_search_results(
        self,
        query: str,
        search_response: "SearchResponse",
    ) -> ProcessedInformation:
        """
        Process search response into structured information.

        Args:
            query: Original search query.
            search_response: SearchResponse from provider.

        Returns:
            ProcessedInformation with normalized data.
        """
        from core.search_provider import SearchResponse

        if not isinstance(search_response, SearchResponse):
            raise ValueError("Invalid search response")

        if not search_response.success or not search_response.results:
            return ProcessedInformation(
                query=query,
                sources=(),
                summary="Không có kết quả tìm kiếm hoặc tìm kiếm thất bại.",
                raw_content=search_response.error or "No results",
                confidence=0.0,
            )

        # Convert to source references
        sources = []
        for result in search_response.results[:self.max_sources]:
            source = SourceReference(
                title=result.title,
                url=result.url,
                domain=result.source,
                snippet=result.snippet[:self.snippet_limit],
                rank=result.rank,
            )
            sources.append(source)

        # Calculate confidence
        confidence = self._calculate_confidence(sources, query)

        # Generate summary
        summary = self._generate_summary(sources, query)

        # Combine raw content
        raw_content = "\n\n".join(
            f"[{s.rank}] {s.title} ({s.domain}): {s.snippet}"
            for s in sources
        )

        return ProcessedInformation(
            query=query,
            sources=tuple(sources),
            summary=summary,
            raw_content=raw_content,
            confidence=confidence,
        )

    def _calculate_confidence(
        self,
        sources: list[SourceReference],
        query: str,
    ) -> float:
        """Calculate confidence score based on source quality."""
        if not sources:
            return 0.0

        score = 0.0
        for source in sources:
            # Base score per source
            source_score = 0.5

            # Boost for trusted domains
            if source.domain in self.TRUSTED_DOMAINS:
                source_score += 0.3

            # Boost for HTTPS
            if source.url.startswith("https://"):
                source_score += 0.1

            # Boost for recent content indicators
            if any(
                word in source.snippet.lower()
                for word in ["2024", "2025", "2026", "mới nhất", "cập nhật", "latest"]
            ):
                source_score += 0.1

            # Penalty for unreliable patterns
            for pattern in self.UNRELIABLE_PATTERNS:
                if re.search(pattern, source.snippet, re.IGNORECASE):
                    source_score -= 0.2

            score += max(0.0, min(1.0, source_score))

        # Average and normalize
        avg_score = score / len(sources)
        return max(0.0, min(1.0, avg_score))

    def _generate_summary(
        self,
        sources: list[SourceReference],
        query: str,
    ) -> str:
        """Generate a concise summary from sources."""
        if not sources:
            return "Không có thông tin."

        # Extract key snippets with titles
        parts = []
        for s in sources[:3]:
            if s.snippet:
                parts.append(f"{s.title}: {s.snippet}")
            elif s.title:
                parts.append(s.title)

        if not parts:
            return f"Tìm thấy {len(sources)} nguồn nhưng không có trích dẫn."

        # Simple summary: combine first few snippets
        combined = " ".join(parts)
        # Limit length
        if len(combined) > 400:
            combined = combined[:400] + "..."

        return combined

    def detect_conflicts(
        self,
        info: ProcessedInformation,
    ) -> list[str]:
        """Detect potential conflicts in sources."""
        conflicts = []
        snippets = [s.snippet.lower() for s in info.sources if s.snippet]

        for i, snippet1 in enumerate(snippets):
            for snippet2 in snippets[i + 1 :]:
                for pattern in self.CONFLICT_PATTERNS:
                    if re.search(pattern, snippet1) and re.search(
                        pattern, snippet2
                    ):
                        conflicts.append(
                            f"Nguồn {i + 1} và nguồn {snippets.index(snippet2) + 1} "
                            "có thể mâu thuẫn nhau"
                        )
                        break

        return conflicts


def process_web_search(
    query: str,
    search_response: "SearchResponse",
    max_sources: int = 5,
) -> ProcessedInformation:
    """Convenience function to process web search results."""
    processor = InformationProcessor(max_sources=max_sources)
    return processor.process_search_results(query, search_response)