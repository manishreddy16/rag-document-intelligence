import tempfile
import unittest

import numpy as np

from tests.fakes import FakeEmbeddingService, install_fake_faiss
from backend.exceptions import VectorStoreError
from backend.models import Chunk
from backend.vector_store import FaissVectorStore

install_fake_faiss()
EMB = FakeEmbeddingService()


def make_chunks():
    texts = ["casual leave is twelve days", "remote work needs approval",
             "laptops must use the vpn"]
    return [Chunk(f"d.pdf-p{i+1}-c1", t, "d.pdf", i + 1)
            for i, t in enumerate(texts)]


class VectorStoreTests(unittest.TestCase):
    def build(self):
        chunks = make_chunks()
        store = FaissVectorStore(EMB.dim)
        store.add(chunks, EMB.embed_documents([c.text for c in chunks]))
        return store

    def test_search_maps_back_to_metadata(self):
        res = self.build().search(EMB.embed_query("casual leave days"), 2)
        self.assertEqual(res[0].chunk.page_number, 1)
        self.assertEqual(res[0].chunk.chunk_id, "d.pdf-p1-c1")
        self.assertGreaterEqual(res[0].score, res[1].score)

    def test_k_larger_than_index(self):
        res = self.build().search(EMB.embed_query("vpn"), 10)
        self.assertEqual(len(res), 3)

    def test_rejects_wrong_dimension_and_count(self):
        store = FaissVectorStore(EMB.dim)
        with self.assertRaises(VectorStoreError):
            store.add(make_chunks(), np.zeros((3, 5)))
        with self.assertRaises(VectorStoreError):
            store.add(make_chunks(), np.zeros((2, EMB.dim)))

    def test_empty_search_fails(self):
        with self.assertRaises(VectorStoreError):
            FaissVectorStore(EMB.dim).search(EMB.embed_query("x"), 1)

    def test_save_and_load_roundtrip(self):
        store, d = self.build(), tempfile.mkdtemp()
        store.save(d)
        loaded = FaissVectorStore.load(d)
        self.assertEqual(len(loaded), 3)
        res = loaded.search(EMB.embed_query("remote work approval"), 1)
        self.assertEqual(res[0].chunk.page_number, 2)

    def test_load_missing_dir(self):
        with self.assertRaises(VectorStoreError):
            FaissVectorStore.load(tempfile.mkdtemp())


if __name__ == "__main__":
    unittest.main()
