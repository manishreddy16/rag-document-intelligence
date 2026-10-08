"""Custom errors so the API layer can map failures to clear messages."""


class RAGError(Exception):
    """Base class for all expected, user-facing errors."""


class UnsupportedFileTypeError(RAGError):
    pass


class DocumentProcessingError(RAGError):
    """File is missing, corrupt, encrypted or unreadable."""


class FileTooLargeError(RAGError):
    """Upload exceeds MAX_UPLOAD_MB."""


class EmptyDocumentError(RAGError):
    """PDF opened fine but has (almost) no extractable text, e.g. a scan."""


class EmbeddingError(RAGError):
    """Embedding model could not be loaded or failed to embed text."""


class VectorStoreError(RAGError):
    """FAISS missing, index inconsistent, or save/load failed."""


class RetrievalError(RAGError):
    """Bad question, or no document has been indexed yet."""


class InvalidQuestionError(RetrievalError):
    """Question is empty / whitespace-only."""


class NoDocumentError(RetrievalError):
    """Nothing has been uploaded and indexed yet."""


class LLMError(RAGError):
    """LLM missing key/model, API failure, rate limit, or malformed response."""


class LLMConfigError(LLMError):
    """Missing/invalid key or model, or unsupported provider (server-side config)."""


class LLMRateLimitError(LLMError):
    """Provider returned HTTP 429."""


class LLMUnavailableError(LLMError):
    """Network failure, provider 5xx, blocked, empty or malformed response."""
