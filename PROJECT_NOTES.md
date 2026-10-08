# PROJECT_NOTES - interview study guide (matches the actual code)

## 0. Verification status - read first
- **Verified offline (authoring sandbox, no internet):** pypdf extraction and chunking on a real PDF; logic of the vector store, retrieval, ingestion, RAG pipeline, Gemini client (with mocks) and the error-to-HTTP mapping, using fakes. Result: 54 tests, 45 passed, 9 skipped, 0 failed. The 9 skipped are the FastAPI tests (FastAPI/httpx not installable there).
- **NOT VERIFIED by the author:** real `all-MiniLM-L6-v2`, real `faiss-cpu`, real Gemini, the running FastAPI app, the UI in a browser.
- **How you close that gap:** run `python scripts/verify_real.py` (and `--gemini`), then the checklist in README. Only claim "I ran it and it works" after you have.
- No accuracy or latency numbers exist. Do not quote any.

## 1. One-paragraph pitch
A FastAPI app where you upload a text-based PDF and ask questions about it. The PDF is split into page-tagged chunks; a Hugging Face sentence-embedding model turns each chunk into a 384-number vector stored in a FAISS index. A question is embedded with the same model, FAISS returns the most similar chunks, and those chunks go into a prompt telling Gemini to answer only from them. The response contains the answer plus file, page, chunk id, score and snippet for each source.

## 2. Concepts

### RAG
- **What:** Retrieval-Augmented Generation = retrieve relevant text first, then give it to an LLM together with the question.
- **Why:** An LLM doesn't know your private PDF and can't fit a big one in its prompt. Retrieval picks the few relevant passages, and showing them as sources lets a human check the answer.
- **RAG vs a plain LLM:** plain LLM answers from training data only (stale, no access to your files, can't cite). RAG answers from supplied text.
- **RAG vs fine-tuning:** fine-tuning changes model weights (costly, slow to update, poor at citing exact passages). RAG changes the *input*: update the document, re-index, done. Fine-tuning is better for style/behaviour, RAG for knowledge that changes.
- **Advantages:** works on private/new documents, checkable sources, no training. **Limitations:** retrieval can miss the right chunk; the LLM can still ignore instructions or misread; quality depends on chunking and embedding model; extra moving parts.

### LLM, tokens, inference, prompting, context
- **LLM:** a neural network trained to predict the next token; fluent, but can state wrong things confidently.
- **Token / tokenization:** the model reads text as tokens (words or word-pieces) produced by a tokenizer; limits and pricing are counted in tokens. Our chunk size is in *characters*, not tokens.
- **Inference:** running a trained model to get output (as opposed to training). Gemini inference happens on Google's servers; the embedding model's inference happens locally on your CPU.
- **Prompting:** the instructions + data you send the model. Ours (`prompts.py`) = rules + retrieved context + question.
- **Context (window):** the text the model can see for one request. RAG fills it with only the relevant chunks.

### Embeddings
- A list of numbers (384 here) representing the meaning of a text; similar meanings -> nearby vectors. Produced by an embedding *model*, not by FAISS.
- **Why needed:** to search by meaning ("vacation days" can match "annual leave"), not just exact words.
- **Why chunks and queries use the same model:** distances only make sense inside one vector space. A different model = a different space = meaningless scores. That's also why changing `EMBEDDING_MODEL` requires re-uploading.
- **Semantic similarity:** we use cosine similarity. Vectors are L2-normalised, so inner product == cosine.

### FAISS
- **Full form:** Facebook AI Similarity Search (library from Meta) for fast nearest-vector search.
- **What it stores:** only vectors. It returns (position, score). It doesn't make embeddings, store text/pages, or answer questions.
- **Why metadata is separate:** we keep a parallel Python list of `Chunk` objects; FAISS position `i` -> `chunks[i]` gives text, page, source. Saved as `index.faiss` + `chunks.json` (+ `document.json`). On load we check `index.ntotal == len(chunks)`.
- **Which index:** `IndexFlatIP` = exact brute-force inner-product search. Perfect for one document; for millions of vectors you'd consider IVF/HNSW (approximate).
- **Save/load:** `faiss.write_index` / `read_index`; after restart the app reloads the last index.

### Chunking
- **Why:** one vector for a whole document blurs topics; prompts have size limits; small chunks give focused retrieval and citable context.
- **Size/overlap:** 1000 chars / 150 overlap (configurable). Overlap lowers the chance an answer is cut at a boundary; it doesn't guarantee it.
- **Page-aware:** chunks never cross pages, so page numbers in sources are exact (physical PDF page order, not printed labels).
- **Tradeoffs:** bigger chunks = more context but blurrier match; smaller = sharper but may lose surrounding meaning; character-based splitting ignores real sentence/section structure.

### Retrieval
question -> `embed_query` -> FAISS search -> top-k (default 4) chunks with scores -> context. There is **no similarity threshold**: FAISS always returns the nearest k, even if none is truly relevant. "Not in the document" is decided by the LLM following the prompt.

### Prompt and grounding
`build_prompt` inserts each chunk as `[Source n: file, page p]` + text under "Context:", followed by the question and rules: use ONLY the context, don't invent, otherwise reply exactly "I could not find this in the provided document." This *tries* to keep answers tied to retrieved text. It does **not** eliminate hallucination: retrieval can miss, the model can ignore rules, and text inside a PDF could try to inject instructions (not defended against).

### Hugging Face (in THIS project)
Only source of the pretrained model `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90 MB, downloaded once, runs locally on CPU, no key). Used for both chunk and query embeddings.

### LangChain (in THIS project) - be honest
Narrow role: `langchain_huggingface.HuggingFaceEmbeddings` loads and runs the embedding model (`embeddings.py`, with `normalize_embeddings=True`). Chunking, FAISS use, prompt building, the Gemini call and the pipeline are plain Python on purpose. Don't claim chains, agents, LangChain retrievers or LangChain's FAISS wrapper.

### FastAPI, HTTP, REST
- **API:** a defined way for programs to talk. **HTTP:** the protocol (request -> response with a status code). **REST-style:** resources + HTTP verbs + status codes. **FastAPI:** Python framework implementing such an API; uses pydantic models for validation and auto-generates `/docs`. **JSON:** the text format of request/response bodies. **Endpoint:** a URL + method.
- `GET /health` - status, whether a document is loaded, whether key+model are set (never the key).
- `POST /upload` - multipart field `file` (PDF); indexes it, replacing any previous document.
- `POST /ask` - JSON `{"question": "..."}` -> `{"answer", "sources":[...]}`.
- Endpoints are plain `def` so FastAPI runs them in a thread pool; slow embedding/LLM calls don't block the event loop. Errors are JSON `{"error":{"code","message"}}`.

### Gemini (in THIS project)
`llm_service.py`: `GeminiLLMService.generate(prompt)` sends a POST to Google's `generateContent` REST endpoint using Python's `urllib` (no SDK). Key goes in the `x-goog-api-key` header (not the URL); model name and key come from `LLM_MODEL` / `LLM_API_KEY`; temperature 0.2; response text is read from `candidates[0].content.parts`. 429 -> rate-limit error; 400/401/403/404 -> config error (Gemini reports a bad key as 400); network/5xx/blocked/empty -> unavailable error. Transient failures (408/429/5xx/network) are retried with bounded exponential backoff (default 3 retries, ~1s/2s/4s + small jitter, stdlib `time.sleep`, `LLM_MAX_RETRIES`); permanent ones (400/401/403/404) fail immediately. Interview line: "retry only what can succeed later, bound it, back off exponentially so we don't hammer an overloaded service." Hidden behind an `LLMService` interface so another provider can be added. Model names get retired, which is why none is hard-coded. **A real successful Gemini generation has not been confirmed yet (last real runs: network error and HTTP 503 overload).**

## 3. Flows (match the code)

### Upload flow
User -> UI (`app.js`) -> `POST /upload` -> read at most limit+1 bytes -> `DocumentManager.ingest_pdf`: validate (.pdf, non-empty, size, `%PDF-` header) -> temp file -> pypdf per-page text (scanned/empty -> error) -> page-aware chunks -> Hugging Face embeddings (batches of 64) -> new FAISS index in memory -> save to temp dir then move into `data/index/` -> swap in-memory active document -> best-effort housekeeping of the uploaded PDF copy.
**Policy:** one active document; a failed upload leaves the previous index intact (tested with fakes).

### Question flow
User -> UI -> `POST /ask` -> validate question (1-1000 chars) -> 409 if no document -> `embed_query` -> FAISS top-k -> `build_prompt` (context + rules + question) -> Gemini -> `Answer` + `SourceRef`s -> JSON -> UI shows answer and sources (file, page, chunk id, score, snippet).

## 4. What can fail / how handled
Wrong type 415; too big 413; corrupt/encrypted 400; scanned 422; no document 409; bad question 422; embedding model failed 503; FAISS/index 500; Gemini rate limit 429; key/model missing/rejected 503; Gemini network/blocked/malformed 502; anything else generic 500 (details only in the server log). Key is redacted from error messages and never returned by `/health`. Frontend inserts server text with `textContent` (no HTML injection).

## 5. Known limitations and how to improve
One document; text PDFs only (add OCR); no auth; no chat memory; character chunking (token/semantic chunking); exact search only (IVF/HNSW at scale); small embedding model (larger model, re-ranker); no relevance threshold (add a minimum score before calling the LLM); no evaluation (build a labelled Q&A set, measure recall@k); prompt injection via PDF text not defended; index files are replaced one by one (a crash in between could leave them mismatched; a versioned directory + pointer swap would fix it); free-tier Gemini data-use caveat - don't upload confidential files.
A bug found in review and fixed: a failure while tidying the saved PDF copy *after* the new index was written to disk could leave memory (old) and disk (new) out of step; housekeeping is now best-effort after the swap (regression test added).

## 6. Resume alignment
| Claim | Supported? | Where / how |
|---|---|---|
| Python | Yes | all of `backend/`, tests, scripts |
| LangChain | **Partly (narrow)** | `embeddings.py`: `HuggingFaceEmbeddings` wrapper only |
| Hugging Face | Yes in code; real run **NOT VERIFIED by author** | `all-MiniLM-L6-v2` via `embeddings.py`; confirm with `verify_real.py` |
| FAISS | Yes in code; real library **NOT VERIFIED by author** | `vector_store.py` (`IndexFlatIP`, save/load) |
| FastAPI | Yes in code; app run and 9 API tests **NOT VERIFIED by author** | `api.py`: `/upload`, `/ask`, `/health` |
| Document processing | Yes, **text PDFs only** | `document_loader.py` (pypdf), `chunking.py` |
| Semantic embeddings | Yes (real model unverified by author) | `embeddings.py`, `ingestion.py` |
| Relevant-context retrieval | Yes (logic tested with fakes) | `retrieval.py` |
| Vector similarity search | Yes (real FAISS unverified by author) | `vector_store.py` |
| LLM prompts | Yes | `prompts.py` |
| Context-grounded responses | Prompt-level grounding only; no guarantee; Gemini unverified by author | `prompts.py`, `rag_pipeline.py` |
| Source references | Yes | `SourceRef` (file, page, chunk id, score, snippet) |
| Document ingestion | Yes | `ingestion.py`, `POST /upload` |
| QA workflow | Yes | `rag_pipeline.py`, `POST /ask` |

## 7. Questions to be able to answer
The two flows; why query and chunks share one model; why FAISS needs the parallel metadata list; why chunks respect page boundaries; replace-on-upload policy and why it swaps only after success; what LangChain does *and doesn't* do; how errors map to HTTP codes; why no similarity threshold exists and what that implies; what is and isn't verified.
