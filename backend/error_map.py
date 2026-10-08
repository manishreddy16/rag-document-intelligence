"""Maps expected errors (RAGError subclasses) to HTTP status + machine code.

Kept free of FastAPI imports so the mapping can be unit-tested on its own.
"""
from .exceptions import (DocumentProcessingError, EmbeddingError,
                         EmptyDocumentError, FileTooLargeError,
                         InvalidQuestionError, LLMConfigError, LLMError,
                         LLMRateLimitError, LLMUnavailableError,
                         NoDocumentError, RAGError, UnsupportedFileTypeError,
                         VectorStoreError)

# (exception class, HTTP status, machine-readable code). Checked in order,
# so subclasses must come before their parents.
_ERROR_MAP = [
    (UnsupportedFileTypeError, 415, "unsupported_file_type"),
    (FileTooLargeError, 413, "file_too_large"),
    (EmptyDocumentError, 422, "empty_document"),
    (DocumentProcessingError, 400, "invalid_document"),
    (InvalidQuestionError, 422, "invalid_question"),
    (NoDocumentError, 409, "no_document"),
    (EmbeddingError, 503, "embedding_error"),
    (VectorStoreError, 500, "vector_store_error"),
    (LLMRateLimitError, 429, "llm_rate_limited"),
    (LLMConfigError, 503, "llm_not_configured"),
    (LLMUnavailableError, 502, "llm_unavailable"),
    (LLMError, 502, "llm_error"),
    (RAGError, 400, "request_error"),
]


def status_for(exc: Exception) -> tuple[int, str]:
    for cls, status, code in _ERROR_MAP:
        if isinstance(exc, cls):
            return status, code
    return 500, "internal_error"
