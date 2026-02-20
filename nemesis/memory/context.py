"""SessionContext -- lightweight in-memory context for the current session."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionContext:
    """Tracks queries and results within a single session.

    This is a pure in-memory structure (no graph DB interaction).
    It accumulates queries the user/agent has issued and their results,
    and can produce a text summary for context injection.
    """

    _queries: list[str] = field(default_factory=list)
    _results: list[dict] = field(default_factory=list)

    def add_query(self, query: str) -> None:
        """Record a query string."""
        self._queries.append(query)

    def add_result(self, query: str, data: dict) -> None:
        """Record a result consisting of a query and associated data."""
        self._results.append({"query": query, "data": data})

    def get_queries(self) -> list[str]:
        """Return all recorded queries."""
        return list(self._queries)

    def get_results(self) -> list[dict]:
        """Return all recorded results."""
        return list(self._results)

    def build_summary(self) -> str:
        """Build a human-readable text summary of the session context.

        Includes all queries and any file references found in result data.
        """
        lines: list[str] = []

        if self._queries:
            lines.append("Queries:")
            for q in self._queries:
                lines.append(f"  - {q}")

        if self._results:
            lines.append("Results:")
            for r in self._results:
                lines.append(f"  - query: {r['query']}")
                data = r.get("data", {})
                files = data.get("files", [])
                if files:
                    lines.append(f"    files: {', '.join(str(f) for f in files)}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all queries and results."""
        self._queries.clear()
        self._results.clear()
