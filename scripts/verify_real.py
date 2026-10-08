"""Real-component smoke test. Run on YOUR machine (needs internet on first run):

    python scripts/verify_real.py            # HF model + real FAISS + retrieval
    python scripts/verify_real.py --gemini   # also one real Gemini call (needs .env key + model)

It uses the project's own modules with the REAL Hugging Face model and REAL
faiss-cpu (no fakes). It prints PASS/FAIL per step and exits non-zero on failure.
The API key is never printed.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from backend.chunking import chunk_pages  # noqa: E402
from backend.config import load_settings  # noqa: E402
from backend.document_loader import extract_pdf_pages  # noqa: E402
from backend.embeddings import HuggingFaceEmbeddingService  # noqa: E402
from backend.vector_store import FaissVectorStore  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_policy.pdf"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main(with_gemini: bool) -> int:
    s = load_settings()

    # 1. Real FAISS (not the numpy stand-in used by unit tests)
    import faiss
    check("real faiss imported", hasattr(faiss, "IndexFlatIP") and hasattr(faiss, "__version__"),
          f"faiss {getattr(faiss, '__version__', '?')}")

    # 2. Real Hugging Face embeddings
    emb = HuggingFaceEmbeddingService(s.embedding_model)
    t = time.time()
    doc_vecs = emb.embed_documents(["Employees get 12 days of casual leave.", "The VPN is mandatory."])
    check("embed_documents shape", doc_vecs.shape == (2, 384), f"{doc_vecs.shape} in {time.time() - t:.1f}s (incl. model load)")
    q = emb.embed_query("how many vacation days")
    check("embed_query shape", q.shape == (384,), str(q.shape))
    check("vectors L2-normalised", abs(float(np.linalg.norm(q)) - 1.0) < 1e-3, f"norm={np.linalg.norm(q):.4f}")

    # 3. PDF -> chunks -> embeddings -> real FAISS -> search -> save/load -> search
    pages = extract_pdf_pages(SAMPLE, s.min_extracted_chars)
    chunks = chunk_pages(pages, SAMPLE.name, s.chunk_size, s.chunk_overlap)
    store = FaissVectorStore(384)
    store.add(chunks, emb.embed_documents([c.text for c in chunks]))
    check("index built", len(store) == len(chunks) > 0, f"{len(store)} chunks")

    expectations = [("How many days of casual leave are allowed?", 1, "casual leave"),
                    ("Is the VPN required for remote work?", 2, "vpn"),
                    ("How many sick days do I get?", 1, "sick")]
    with tempfile.TemporaryDirectory() as tmp:
        store.save(Path(tmp))
        reloaded = FaissVectorStore.load(Path(tmp))
        check("save/load keeps vector count", len(reloaded) == len(store))
        for question, page, keyword in expectations:
            for label, st in (("fresh", store), ("reloaded", reloaded)):
                top = st.search(emb.embed_query(question), s.top_k)
                ok = bool(top) and top[0].chunk.page_number == page and keyword in top[0].chunk.text.lower()
                check(f"retrieval ({label}): {question!r}", ok,
                      f"top page={top[0].chunk.page_number} score={top[0].score:.3f}" if top else "no results")
    print("\nNote: the sample PDF has only 2 chunks, so this shows the plumbing works and\n"
          "ranking is sensible; it is NOT a retrieval-quality evaluation.")

    # 4. Optional: one real Gemini call, answerable + unanswerable
    if with_gemini:
        from backend.llm_service import create_llm_service
        from backend.prompts import INSUFFICIENT, build_prompt
        from backend.retrieval import Retriever
        llm = create_llm_service(s)
        retriever = Retriever(emb, store, s.top_k)
        for question in ("How many days of casual leave do employees get?",
                         "Who won the 2018 football World Cup?"):
            try:
                ans = llm.generate(build_prompt(question, retriever.retrieve(question)))
                print(f"\nQ: {question}\nA: {ans}")
                check(f"gemini answered ({question[:30]}...)", bool(ans.strip()))
                if "World Cup" in question:
                    print(f"   (expected refusal text: {INSUFFICIENT!r}; judge by eye - models can deviate)")
            except Exception as exc:  # report type + message; key is never in these messages
                check(f"gemini call ({question[:30]}...)", False, f"{type(exc).__name__}: {exc}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main("--gemini" in sys.argv))
