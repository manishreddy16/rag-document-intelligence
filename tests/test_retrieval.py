import unittest
from pathlib import Path

from tests.fakes import FakeEmbeddingService, install_fake_faiss
from backend.chunking import chunk_pages
from backend.document_loader import extract_pdf_pages
from backend.exceptions import RetrievalError
from backend.retrieval import Retriever
from backend.vector_store import FaissVectorStore

install_fake_faiss()
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_policy.pdf"


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.emb = FakeEmbeddingService()
        pages = extract_pdf_pages(SAMPLE)
        self.chunks = chunk_pages(pages, SAMPLE.name, 200, 30)
        self.store = FaissVectorStore(self.emb.dim)
        self.store.add(self.chunks,
                       self.emb.embed_documents([c.text for c in self.chunks]))
        self.retriever = Retriever(self.emb, self.store, top_k=2)

    def test_end_to_end_sample_pdf(self):
        res = self.retriever.retrieve("How many days of casual leave?")
        self.assertEqual(res[0].chunk.page_number, 1)
        self.assertEqual(res[0].chunk.source, "sample_policy.pdf")
        self.assertLessEqual(len(res), 2)

    def test_second_page_topic(self):
        res = self.retriever.retrieve("remote work VPN laptops")
        self.assertEqual(res[0].chunk.page_number, 2)

    def test_empty_question(self):
        with self.assertRaises(RetrievalError):
            self.retriever.retrieve("   ")

    def test_no_document_yet(self):
        empty = Retriever(self.emb, FaissVectorStore(self.emb.dim), 3)
        with self.assertRaises(RetrievalError):
            empty.retrieve("anything")


if __name__ == "__main__":
    unittest.main()
