# RAG-Based Document Intelligence Assistant

Upload a **text-based PDF**, then ask questions about it. The app retrieves the
most relevant passages with semantic search and asks an LLM to answer **only from
those passages**, returning the answer plus source references (file, page,
chunk id, similarity score, snippet).

> **Verification status (honest).** The unit-test suite (54 tests) uses a *fake*
> embedding model, a *fake* LLM and, if `faiss` isn't installed, a numpy stand-in.
> The last authoring sandbox had **no internet**, so these were **NOT VERIFIED by the
> author**: the real Hugging Face model, real FAISS, real Gemini calls, the running
> FastAPI app + its 9 API tests, and the UI in a browser. Run
> `python scripts/verify_real.py` and the checklist under [Testing](#testing) on your
> machine to verify them yourself before you claim they work.

## Architecture

```
Browser UI (frontend/)  ->  FastAPI (backend/api.py)
 Upload:  PDF -> validate -> pypdf text per page -> page-aware chunks
          -> embeddings (all-MiniLM-L6-v2) -> FAISS index -> saved to data/index/
 Ask:     question -> same embedding model -> FAISS top-k -> context block
          -> grounded prompt -> Gemini -> answer + sources -> UI
```

## Technologies - what each actually does here

| Technology | Role in this project |
|---|---|
| Python | Everything in `backend/` |
| Hugging Face | Supplies the pretrained `sentence-transformers/all-MiniLM-L6-v2` model (384-dim, runs locally on CPU, ~90 MB first-use download, no API key) |
| LangChain | **Narrow role:** `langchain-huggingface`'s `HuggingFaceEmbeddings` loads/runs the embedding model (`backend/embeddings.py`). Chunking, FAISS, prompt and Gemini call are deliberately plain Python so each step is visible and testable |
| FAISS | `IndexFlatIP` exact vector search; used directly (`backend/vector_store.py`) |
| FastAPI | HTTP layer: `/upload`, `/ask`, `/health`, request/response schemas, status codes |
| Gemini (Google AI Studio) | The LLM, called over REST with the standard library, behind the `LLMService` interface |
| pypdf | PDF text extraction with real page numbers |

## Setup (Windows / PowerShell)

Python **3.10 or newer** is required (the code uses `X | None` type syntax). A
recent 3.11/3.12 is a conservative choice because `faiss-cpu` and `torch` ship
wheels per Python version; if `pip install` fails on a very new Python, use 3.11/3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env      # then edit .env
```

Edit `.env`: set `LLM_API_KEY` (create at https://aistudio.google.com) and
`LLM_MODEL` (a current Gemini model your key can use; check
https://ai.google.dev/gemini-api/docs/models and the deprecations page - names change, so
none is hard-coded). Never commit `.env`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLM_API_KEY` | - | Gemini key (secret) |
| `LLM_MODEL` | - | Gemini model name |
| `LLM_PROVIDER` | `gemini` | Only `gemini` is implemented |
| `LLM_TIMEOUT_SECONDS` | 30 | LLM request timeout (per attempt) |
| `LLM_MAX_RETRIES` | 3 | Retries for transient Gemini errors; 0 disables |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it requires re-uploading (dimension/space change) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Characters per chunk / overlap |
| `TOP_K` | 4 | Chunks retrieved per question |
| `MAX_UPLOAD_MB` | 20 | Upload size limit |
| `MAX_QUESTION_CHARS` | 1000 | Question length limit |
| `MIN_EXTRACTED_CHARS` | 50 | Below this the PDF is treated as scanned/empty |

## Run

```powershell
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000 (UI) or http://127.0.0.1:8000/docs (interactive API docs).
Try `data/sample/sample_policy.pdf` (a fictional 2-page policy) first.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status, whether a document is loaded, whether LLM key+model are set (never returns the key) |
| `POST /upload` | multipart field `file` (PDF). Indexes it, **replacing** any previous document |
| `POST /ask` | JSON `{"question": "..."}` -> `{"answer": "...", "sources": [{source, page_number, chunk_id, score, snippet}]}` |

Errors are JSON: `{"error": {"code": "...", "message": "..."}}`.

| Situation | HTTP | code |
|---|---|---|
| not a .pdf | 415 | unsupported_file_type |
| over size limit | 413 | file_too_large |
| corrupt / encrypted / empty file | 400 | invalid_document |
| scanned PDF (no text) | 422 | empty_document |
| empty/too-long/missing question | 422 | invalid_question / validation_error |
| no document uploaded yet | 409 | no_document |
| embedding model failed to load/run | 503 | embedding_error |
| FAISS/index problem | 500 | vector_store_error |
| LLM rate limit | 429 | llm_rate_limited |
| LLM key/model missing or rejected | 503 | llm_not_configured |
| LLM network/5xx/blocked/malformed | 502 | llm_unavailable |
| anything unexpected | 500 | internal_error (details only in server log) |

## Design decisions

- **One active document.** A new upload replaces the old one (no silent mixing of unrelated documents). The new index is fully built before it is swapped in, so a failed upload keeps the previous working index. The index persists in `data/index/` and is reloaded on restart.
- **Page-aware chunking.** Chunks never cross page boundaries, so page numbers in sources are exact. Page numbers are the PDF's physical page order (1-based), not printed page labels.
- **Source scores are shown even when the LLM says "not found"**: low scores signal weak retrieval.
- RAG can *reduce* unsupported answers; it does **not** guarantee zero hallucinations.

## Testing

```powershell
python -m unittest discover -s tests -v
```
Last result in the authoring sandbox (no FastAPI/httpx, no internet): **54 tests, 45 ran and passed, 9 skipped, 0 failed.**
The 9 skipped are `tests/test_api.py` (need FastAPI + httpx, installed by
`requirements.txt` / `requirements-dev.txt`). On your machine, after installing,
all 54 should run; if any fail, that is real information - please investigate.

| What | How tested |
|---|---|
| Extraction, chunking, validation | Real `pypdf` on a real generated PDF |
| FAISS store, retrieval, ingestion, pipeline | **Fake** hashed bag-of-words embeddings (lexical, not semantic); numpy stand-in for FAISS if `faiss` absent |
| Gemini client + retry | `urlopen` **mocked**; request/response/error handling and retry/backoff logic only (sleep is injected, so tests are instant) |
| API endpoints | FastAPI TestClient with fake embedder + fake LLM (9 tests; not run in the sandbox) |
| Real HF model, real FAISS, real retrieval | `python scripts/verify_real.py` (**you run it**) |
| Real Gemini | `python scripts/verify_real.py --gemini` (needs your key + model in `.env`) |
| UI in a browser | Manual checklist below |

### Real-verification checklist (run on your machine)
1. `python -m unittest discover -s tests -v` - note passed/skipped/failed exactly.
2. `python scripts/verify_real.py` - real model (384-dim), real FAISS, save/load, retrieval on the sample PDF.
3. Put `LLM_API_KEY` and `LLM_MODEL` in `.env`, run `python scripts/verify_real.py --gemini`.
4. `uvicorn backend.main:app --reload`, open http://127.0.0.1:8000, upload `data/sample/sample_policy.pdf`.
5. Ask "How many days of casual leave are allowed?" -> expect 12, source page 1.
6. Ask something not in the PDF -> expect "I could not find this in the provided document." (LLM behaviour; not guaranteed).
7. Try: a `.txt` file (415), an empty question, `/ask` before any upload (409), a wrong `LLM_API_KEY` (clean error), stop and restart the server then ask again (index reloads from `data/index/`).
8. Browser dev-tools console should show no errors.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `pip install` fails for faiss-cpu/torch | Very new Python; use 3.11/3.12 in a fresh venv |
| First upload is slow | The ~90 MB embedding model downloads once, then is cached |
| `embedding_error` (503) | Model download failed (no internet / proxy) |
| `llm_not_configured` (503) | `LLM_API_KEY` or `LLM_MODEL` empty, key rejected, or model name retired/not found (Gemini 400/401/403/404 map here) |
| `llm_rate_limited` (429) | Provider quota/rate limit; wait and retry |
| `vector_store_error` on save/load | Rare: FAISS file I/O can fail on non-ASCII Windows paths; move the project to a plain-ASCII path |
| Server ignores `.env` edits | Restart uvicorn |
| "No document" after restart | `data/index/` was deleted or damaged; upload again |

## Limitations
Text PDFs only (no OCR, DOCX, etc.); one document at a time; no authentication;
no conversation memory (each question is independent); character-based chunks;
exact (brute-force) FAISS search; no relevance threshold (the LLM prompt, not code, decides "not found"); index files are replaced one by one, not as a single atomic transaction; small embedding model can miss paraphrases; no
accuracy/latency evaluation has been done; free-tier Gemini inputs may be used by
Google to improve its products - do not upload confidential documents.

## Project structure
```
backend/  config, models, exceptions, error_map, document_loader, chunking,
          embeddings, vector_store, retrieval, prompts, llm_service,
          rag_pipeline, ingestion, schemas, api, main
frontend/ index.html, style.css, app.js
tests/    unittest suite + fakes
scripts/  make_sample_pdf.py, verify_real.py (real HF/FAISS/Gemini smoke test)
data/     sample/ (fictional sample PDF), uploads/ + index/ (runtime, git-ignored)
```

## Gemini retry behaviour

`GeminiLLMService` makes up to 1 + `LLM_MAX_RETRIES` (default 4) attempts, waiting about 1s, 2s, 4s (exponential, plus up to 0.25s jitter) between them.

- **Retried (transient):** HTTP 408, 429, 500, 502, 503, 504, and network/timeout errors.
- **Not retried (permanent):** HTTP 400, 401, 403, 404 (bad key/model/request), a missing key or model (no request is sent), unreadable/empty/blocked responses.
- After the last attempt the original exception type is raised (`LLMRateLimitError` for 429, otherwise `LLMUnavailableError`), so the API error mapping is unchanged.
- Logs contain only the error type and delay, never the key. Worst case a request can take roughly 4 x timeout + 7s.

### Verification status (honest)
- **Verified locally by the author (before this change):** real FAISS, real Hugging Face embeddings, persistence/reload, retrieval, running FastAPI app + UI up to the LLM call; 54 offline tests.
- **Verified with mocks only (this change):** retry/backoff logic (10 new tests, no network). The authoring sandbox had no internet, no FastAPI/httpx and no key, so the 9 API tests were still skipped there.
- **NOT verified:** a real successful Gemini generation. Before this change `verify_real.py --gemini` hit "network unreachable" and HTTP 503 ("high demand"). Retries may ride out brief 503s but do not guarantee success; re-run `python scripts/verify_real.py --gemini` yourself. Also confirm `LLM_MODEL` is a valid current model name: a wrong name gives a non-retried 404 (`llm_not_configured`).
