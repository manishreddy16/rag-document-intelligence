"""Single place for all settings. Values come from environment / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional so core modules work without it
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Paths (runtime data, git-ignored)
    upload_dir: Path = ROOT_DIR / "data" / "uploads"
    index_dir: Path = ROOT_DIR / "data" / "index"
    # Upload / extraction limits
    max_upload_mb: int = 20
    min_extracted_chars: int = 50
    # Chunking (measured in characters)
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_question_chars: int = 1000
    # Retrieval
    top_k: int = 4
    # Models (used from Checkpoint 3 onward)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "ollama"
    llm_model: str = ""          # set in .env
    llm_api_key: str = ""        # set in .env, never in code
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3     # retries for transient LLM errors (503/429/...)

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be > 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE")
        if self.llm_max_retries < 0:
            raise ValueError("LLM_MAX_RETRIES must be >= 0")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be > 0")


def load_settings() -> Settings:
    e = os.environ.get
    d = Settings.__dataclass_fields__
    return Settings(
        upload_dir=Path(e("UPLOAD_DIR", ROOT_DIR / "data" / "uploads")),
        index_dir=Path(e("INDEX_DIR", ROOT_DIR / "data" / "index")),
        max_upload_mb=int(e("MAX_UPLOAD_MB", d["max_upload_mb"].default)),
        min_extracted_chars=int(e("MIN_EXTRACTED_CHARS", d["min_extracted_chars"].default)),
        chunk_size=int(e("CHUNK_SIZE", d["chunk_size"].default)),
        chunk_overlap=int(e("CHUNK_OVERLAP", d["chunk_overlap"].default)),
        max_question_chars=int(e("MAX_QUESTION_CHARS", d["max_question_chars"].default)),
        top_k=int(e("TOP_K", d["top_k"].default)),
        embedding_model=e("EMBEDDING_MODEL", d["embedding_model"].default),
        llm_provider=e("LLM_PROVIDER", d["llm_provider"].default),
        llm_model=e("LLM_MODEL", ""),
        llm_api_key=e("LLM_API_KEY", ""),
        llm_timeout_seconds=int(e("LLM_TIMEOUT_SECONDS", d["llm_timeout_seconds"].default)),
        llm_max_retries=int(e("LLM_MAX_RETRIES", d["llm_max_retries"].default)),
    )
