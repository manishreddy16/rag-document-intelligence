"""Test helpers. NOT the real model / real FAISS.

FakeEmbeddingService: deterministic hashed bag-of-words vectors (lexical, not
semantic) so retrieval logic can be tested without downloading a model.
install_fake_faiss(): if the real `faiss` is not installed, injects a tiny
numpy stand-in so our mapping/persistence logic can still be tested.
With real faiss installed (your machine) the real library is used.
"""
import pickle
import re
import sys
import types
import zlib

import numpy as np


class FakeEmbeddingService:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class _FlatIP:
    def __init__(self, d):
        self.d, self.x = d, np.zeros((0, d), dtype="float32")

    @property
    def ntotal(self):
        return len(self.x)

    def add(self, x):
        self.x = np.vstack([self.x, x])

    def search(self, q, k):
        sims = q @ self.x.T
        order = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, order, axis=1)
        pad = k - order.shape[1]
        if pad:
            order = np.pad(order, ((0, 0), (0, pad)), constant_values=-1)
            scores = np.pad(scores, ((0, 0), (0, pad)), constant_values=-1e9)
        return scores, order


def install_fake_faiss():
    try:
        import faiss  # noqa: F401
        return False
    except ImportError:
        m = types.ModuleType("faiss")
        m.IndexFlatIP = _FlatIP
        def _write(idx, p):
            with open(p, "wb") as f:
                f.write(pickle.dumps(idx))

        def _read(p):
            with open(p, "rb") as f:
                return pickle.loads(f.read())

        m.write_index, m.read_index = _write, _read
        sys.modules["faiss"] = m
        return True
