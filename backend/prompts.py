"""Grounded-generation prompt: retrieved chunks + question -> one prompt string."""
from .models import RetrievedChunk

INSUFFICIENT = "I could not find this in the provided document."

TEMPLATE = """You are a document question-answering assistant.
Answer the question using ONLY the context below, which was retrieved from the user's document.
- Do not use outside knowledge and do not invent facts.
- If the context does not contain enough information, reply exactly: "{insufficient}"
- Be concise and answer the actual question.

Context:
{context}

Question: {question}

Answer:"""


def build_context(retrieved: list[RetrievedChunk]) -> str:
    blocks = [f"[Source {i}: {r.chunk.source}, page {r.chunk.page_number}]\n{r.chunk.text}"
              for i, r in enumerate(retrieved, start=1)]
    return "\n\n".join(blocks)


def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    return TEMPLATE.format(insufficient=INSUFFICIENT,
                           context=build_context(retrieved), question=question.strip())
