"""Plain data containers passed between pipeline stages."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based, as a reader would see it
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # e.g. "policy.pdf-p4-c2"
    text: str
    source: str        # original file name
    page_number: int   # 1-based page the chunk came from


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity (higher = more similar)


@dataclass(frozen=True)
class SourceRef:
    source: str
    page_number: int
    chunk_id: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list


@dataclass(frozen=True)
class DocumentInfo:
    """What is currently indexed. Persisted as document.json next to the index."""
    filename: str
    pages: int         # pages that contained extractable text
    chunks: int
    indexed_at: str    # ISO-8601 UTC
