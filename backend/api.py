"""FastAPI application: /upload, /ask, /health (+ the static UI at /).

FastAPI is only the HTTP layer. Real work lives in DocumentManager (ingestion)
and RAGPipeline (question answering). Expected failures are RAGError subclasses
and are mapped to HTTP status codes in one place (error_map.py).
"""
import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, load_settings
from .embeddings import EmbeddingService, HuggingFaceEmbeddingService
from .error_map import status_for
from .exceptions import RAGError
from .ingestion import DocumentManager
from .llm_service import LLMService, create_llm_service, llm_is_configured
from .rag_pipeline import RAGPipeline
from .schemas import (AskRequest, AskResponse, DocumentOut, ErrorResponse,
                      HealthResponse, SourceOut, UploadResponse)

log = logging.getLogger("rag")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None,
               embedder: EmbeddingService | None = None,
               llm: LLMService | None = None) -> FastAPI:
    """App factory. Tests pass fake embedder/llm; production passes nothing."""
    settings = settings or load_settings()
    embedder = embedder or HuggingFaceEmbeddingService(settings.embedding_model)
    manager = DocumentManager(settings, embedder)
    try:
        if manager.load_existing():
            log.info("Loaded existing index for %s", manager.info.filename)
    except RAGError as exc:  # damaged index: start empty, user can re-upload
        log.warning("Ignoring saved index: %s", exc)

    app = FastAPI(title="RAG Document Intelligence Assistant", version="1.0")

    def redact(message: str) -> str:
        key = settings.llm_api_key
        return message.replace(key, "[redacted]") if key else message

    @app.exception_handler(RAGError)
    async def rag_error_handler(request: Request, exc: RAGError):
        status, code = status_for(exc)
        return _error(status, code, redact(str(exc)))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return _error(422, "validation_error",
                      "Invalid request: " + "; ".join(
                          f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}"
                          for e in exc.errors()))

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        log.exception("Unhandled error")  # details in server log only
        return _error(500, "internal_error", "Unexpected server error.")

    # Plain `def` endpoints: FastAPI runs them in a thread pool, so slow
    # embedding / LLM calls do not block the event loop.
    @app.get("/health", response_model=HealthResponse)
    def health():
        info = manager.info
        return HealthResponse(
            status="ok", document_loaded=info is not None,
            document=DocumentOut(**asdict(info)) if info else None,
            embedding_model=settings.embedding_model,
            llm_provider=settings.llm_provider,
            llm_configured=llm_is_configured(settings))

    _errors = {code: {"model": ErrorResponse} for code in (400, 409, 413, 415, 422, 429, 500, 502, 503)}

    @app.post("/upload", response_model=UploadResponse, responses=_errors)
    def upload(file: UploadFile = File(...)):
        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)  # never read more than limit+1 bytes
        info, replaced = manager.ingest_pdf(file.filename or "", data)
        return UploadResponse(
            message=f"Indexed {info.chunks} chunks from {info.pages} page(s).",
            document=DocumentOut(**asdict(info)), replaced_previous=replaced)

    @app.post("/ask", response_model=AskResponse, responses=_errors)
    def ask(body: AskRequest):
        pipeline = RAGPipeline(manager.retriever(), llm or create_llm_service(settings))
        result = pipeline.ask(body.question)
        return AskResponse(answer=result.answer,
                           sources=[SourceOut(**asdict(s)) for s in result.sources])

    if FRONTEND_DIR.is_dir():  # mounted last so API routes win
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
    return app
