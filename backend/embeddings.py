"""Embedding service. Swap the model/provider by implementing EmbeddingService.

Embeddings are L2-normalised so inner product == cosine similarity.
The SAME service instance/model must embed both chunks and queries.
"""
from typing import Protocol

import numpy as np

from .exceptions import EmbeddingError


class EmbeddingService(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HuggingFaceEmbeddingService:
    """Pretrained sentence-transformers model, run locally on CPU.

    LangChain's HuggingFaceEmbeddings wraps the model; it is loaded lazily so
    importing this module stays cheap and tests don't need the model.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self._model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    encode_kwargs={"normalize_embeddings": True})
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {exc}"
                ) from exc
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("No texts to embed.")
        try:
            vecs = self._load().embed_documents(texts)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        try:
            vec = self._load().embed_query(text)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc
        return np.asarray(vec, dtype="float32")
