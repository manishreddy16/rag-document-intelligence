"""Request/response models for the HTTP API (pydantic, used by FastAPI)."""
from typing import Optional

from pydantic import BaseModel, Field

from .config import load_settings

_MAX_Q = load_settings().max_question_chars


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_Q,
                          description="Question about the uploaded document")


class SourceOut(BaseModel):
    source: str
    page_number: int
    chunk_id: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class DocumentOut(BaseModel):
    filename: str
    pages: int
    chunks: int
    indexed_at: str


class UploadResponse(BaseModel):
    message: str
    document: DocumentOut
    replaced_previous: bool


class HealthResponse(BaseModel):
    status: str
    document_loaded: bool
    document: Optional[DocumentOut] = None
    embedding_model: str
    llm_provider: str
    llm_configured: bool  # True if key and model are set; the key itself is never returned


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
