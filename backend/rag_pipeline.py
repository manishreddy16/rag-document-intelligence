"""Query pipeline: retrieve -> build prompt -> LLM -> answer + sources."""
from .exceptions import RetrievalError
from .llm_service import LLMService
from .models import Answer, SourceRef
from .prompts import build_prompt
from .retrieval import Retriever

SNIPPET_CHARS = 200


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMService):
        self.retriever, self.llm = retriever, llm

    def ask(self, question: str) -> Answer:
        retrieved = self.retriever.retrieve(question)
        if not retrieved:
            raise RetrievalError("No relevant content was retrieved.")
        text = self.llm.generate(build_prompt(question, retrieved))
        sources = [SourceRef(r.chunk.source, r.chunk.page_number, r.chunk.chunk_id,
                             r.score, r.chunk.text[:SNIPPET_CHARS]) for r in retrieved]
        return Answer(answer=text, sources=sources)
