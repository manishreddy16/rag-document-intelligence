"""FAISS vector index + chunk metadata.

FAISS only stores vectors and returns (position, score). It knows nothing about
text or pages, so we keep our own list of Chunk objects where list position i
== FAISS vector id i. Persistence = index.faiss + chunks.json.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .exceptions import VectorStoreError
from .models import Chunk, RetrievedChunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError as exc:
        raise VectorStoreError(
            "faiss is not installed. Run: pip install faiss-cpu") from exc


def _normalize(vectors: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(vectors, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class FaissVectorStore:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise VectorStoreError("Dimension must be > 0.")
        self.dimension = dimension
        # Flat inner-product index: exact search; with normalised vectors the
        # inner product equals cosine similarity.
        self._index = _faiss().IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Expected vectors of shape (n, {self.dimension}), got {vectors.shape}.")
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError("Number of chunks and vectors must match.")
        if not chunks:
            return
        self._index.add(_normalize(vectors))
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedChunk]:
        if len(self) == 0:
            raise VectorStoreError("The index is empty.")
        q = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Query dimension {q.shape[1]} != index dimension {self.dimension}.")
        scores, ids = self._index.search(_normalize(q), min(k, len(self)))
        return [RetrievedChunk(self._chunks[i], float(s))
                for s, i in zip(scores[0], ids[0]) if i != -1]  # -1 = no result

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            _faiss().write_index(self._index, str(directory / INDEX_FILE))
            (directory / CHUNKS_FILE).write_text(
                json.dumps([asdict(c) for c in self._chunks]), encoding="utf-8")
        except OSError as exc:
            raise VectorStoreError(f"Could not save index: {exc}") from exc

    @classmethod
    def load(cls, directory: Path) -> "FaissVectorStore":
        directory = Path(directory)
        try:
            index = _faiss().read_index(str(directory / INDEX_FILE))
            raw = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
            chunks = [Chunk(**d) for d in raw]
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            raise VectorStoreError(f"Could not load index: {exc}") from exc
        if index.ntotal != len(chunks):
            raise VectorStoreError("Index and chunk metadata are out of sync.")
        store = cls(index.d)
        store._index, store._chunks = index, chunks
        return store
