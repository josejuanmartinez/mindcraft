from dataclasses import dataclass, field


@dataclass
class SearchResult:
    documents: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
