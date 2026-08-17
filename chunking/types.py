from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class RawPassage:
    query_id: int | None
    query_type: str | None
    language: str
    passage_index: int
    is_selected: int
    source: Literal["translated", "english", "demo"]
    text: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltChunk:
    chunk_id: str
    parent_id: str
    text: str
    parent_text: str
    language: str
    query_id: int | None
    query_type: str | None
    is_selected: int
    source: str
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "text": self.text,
            "parent_text": self.parent_text,
            "language": self.language,
            "query_id": self.query_id,
            "query_type": self.query_type,
            "is_selected": self.is_selected,
            "source": self.source,
            "strategy": self.strategy,
        }
