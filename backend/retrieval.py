"""Question -> query embedding -> FAISS search -> top-k chunks."""
from .embeddings import EmbeddingService
from .exceptions import InvalidQuestionError, NoDocumentError
from .models import RetrievedChunk
from .vector_store import FaissVectorStore


class Retriever:
    def __init__(self, embedder: EmbeddingService, store: FaissVectorStore, top_k: int):
        self.embedder, self.store, self.top_k = embedder, store, top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        question = (question or "").strip()
        if not question:
            raise InvalidQuestionError("Question must not be empty.")
        if len(self.store) == 0:
            raise NoDocumentError("No document has been uploaded yet. Upload a PDF first.")
        return self.store.search(self.embedder.embed_query(question), self.top_k)
