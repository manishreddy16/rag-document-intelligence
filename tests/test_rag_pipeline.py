import unittest
from pathlib import Path

from tests.fakes import FakeEmbeddingService, install_fake_faiss
from backend.chunking import chunk_pages
from backend.document_loader import extract_pdf_pages
from backend.exceptions import LLMError, RetrievalError
from backend.prompts import INSUFFICIENT, build_context, build_prompt
from backend.rag_pipeline import RAGPipeline
from backend.retrieval import Retriever
from backend.vector_store import FaissVectorStore

install_fake_faiss()
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_policy.pdf"


class FakeLLM:
    def __init__(self, reply="Employees get 12 days.", fail=False):
        self.reply, self.fail, self.last_prompt = reply, fail, None

    def generate(self, prompt):
        self.last_prompt = prompt
        if self.fail:
            raise LLMError("boom")
        return self.reply


class PipelineTests(unittest.TestCase):
    def setUp(self):
        emb = FakeEmbeddingService()
        chunks = chunk_pages(extract_pdf_pages(SAMPLE), SAMPLE.name, 200, 30)
        store = FaissVectorStore(emb.dim)
        store.add(chunks, emb.embed_documents([c.text for c in chunks]))
        self.retriever = Retriever(emb, store, top_k=2)
        self.llm = FakeLLM()
        self.pipe = RAGPipeline(self.retriever, self.llm)

    def test_answer_with_sources(self):
        ans = self.pipe.ask("How many days of casual leave?")
        self.assertEqual(ans.answer, "Employees get 12 days.")
        self.assertEqual(ans.sources[0].source, "sample_policy.pdf")
        self.assertEqual(ans.sources[0].page_number, 1)
        self.assertLessEqual(len(ans.sources[0].snippet), 200)

    def test_prompt_contains_context_question_and_rules(self):
        self.pipe.ask("How many days of casual leave?")
        p = self.llm.last_prompt
        self.assertIn("12 days of casual leave", p)
        self.assertIn("[Source 1: sample_policy.pdf, page 1]", p)
        self.assertIn("Question: How many days of casual leave?", p)
        self.assertIn(INSUFFICIENT, p)
        self.assertIn("ONLY the context", p)

    def test_llm_failure_propagates(self):
        with self.assertRaises(LLMError):
            RAGPipeline(self.retriever, FakeLLM(fail=True)).ask("casual leave")

    def test_empty_question(self):
        with self.assertRaises(RetrievalError):
            self.pipe.ask("  ")

    def test_build_context_empty(self):
        self.assertEqual(build_context([]), "")
        self.assertIn("Question: q", build_prompt(" q ", []))


if __name__ == "__main__":
    unittest.main()
