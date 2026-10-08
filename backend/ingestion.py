"""Ingestion: PDF bytes -> validate -> extract -> chunk -> embed -> FAISS -> persist.

Replacement policy (deliberate, easy to explain): the app holds ONE active
document. Uploading a new PDF replaces the old one. The new index is built
completely in memory first; only if every step succeeds is it swapped in and
written to disk, so a failed upload never destroys the previously working index.
"""
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_pages
from .config import Settings
from .document_loader import extract_pdf_pages
from .embeddings import EmbeddingService
from .exceptions import (DocumentProcessingError, EmbeddingError,
                         FileTooLargeError, NoDocumentError,
                         UnsupportedFileTypeError, VectorStoreError)
from .models import DocumentInfo
from .retrieval import Retriever
from .vector_store import CHUNKS_FILE, INDEX_FILE, FaissVectorStore

log = logging.getLogger("rag")

INFO_FILE = "document.json"
EMBED_BATCH = 64  # texts per embedding call, keeps memory bounded


def safe_filename(name: str) -> str:
    """Strip any path and unusual characters; always ends in .pdf."""
    base = Path((name or "").replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._") or "document"
    return f"{stem[:80]}.pdf"


class DocumentManager:
    def __init__(self, settings: Settings, embedder: EmbeddingService):
        self.settings, self.embedder = settings, embedder
        self._store: FaissVectorStore | None = None
        self._info: DocumentInfo | None = None
        self._lock = threading.Lock()  # one ingestion at a time

    # ---- state ------------------------------------------------------
    @property
    def info(self) -> DocumentInfo | None:
        return self._info

    def retriever(self) -> Retriever:
        store = self._store  # single read: safe if an upload swaps concurrently
        if store is None or len(store) == 0:
            raise NoDocumentError("No document has been uploaded yet. Upload a PDF first.")
        return Retriever(self.embedder, store, self.settings.top_k)

    # ---- startup: reuse the last saved index ------------------------
    def load_existing(self) -> bool:
        d = self.settings.index_dir
        if not (d / INDEX_FILE).is_file():
            return False
        store = FaissVectorStore.load(d)  # raises VectorStoreError if damaged
        try:
            info = DocumentInfo(**json.loads((d / INFO_FILE).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            raise VectorStoreError(f"Could not read {INFO_FILE}: {exc}") from exc
        self._store, self._info = store, info
        return True

    # ---- upload -----------------------------------------------------
    def ingest_pdf(self, filename: str, data: bytes) -> tuple[DocumentInfo, bool]:
        """Returns (info, replaced_previous). Raises RAGError subclasses."""
        self._validate(filename, data)
        name = safe_filename(filename)
        with self._lock:
            upload_dir = self.settings.upload_dir
            upload_dir.mkdir(parents=True, exist_ok=True)
            incoming = upload_dir / f".incoming-{uuid.uuid4().hex}.pdf"
            try:
                incoming.write_bytes(data)
                pages = extract_pdf_pages(incoming, self.settings.min_extracted_chars)
                chunks = chunk_pages(pages, name, self.settings.chunk_size,
                                     self.settings.chunk_overlap)
                if not chunks:
                    raise DocumentProcessingError("No usable text chunks could be created.")
                texts = [c.text for c in chunks]
                vectors = _embed_in_batches(self.embedder, texts)
                store = FaissVectorStore(vectors.shape[1])
                store.add(chunks, vectors)
                info = DocumentInfo(
                    filename=name, pages=len(pages), chunks=len(chunks),
                    indexed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                self._persist(store, info)           # disk first ...
                replaced = self._info is not None
                self._store, self._info = store, info  # ... then switch in memory
                # The saved PDF copy is housekeeping only (answers come from the
                # index), so a failure here must not leave memory and disk out of step.
                try:
                    self._remove_old_uploads(keep=name)
                    os.replace(incoming, upload_dir / name)
                except OSError as exc:
                    log.warning("Indexed OK but could not keep the uploaded PDF copy: %s", exc)
                return info, replaced
            except OSError as exc:
                raise DocumentProcessingError(f"Could not store the upload: {exc}") from exc
            finally:
                incoming.unlink(missing_ok=True)

    # ---- helpers ----------------------------------------------------
    def _validate(self, filename: str, data: bytes) -> None:
        if not (filename or "").lower().endswith(".pdf"):
            raise UnsupportedFileTypeError("Only .pdf files are supported.")
        if not data:
            raise DocumentProcessingError("The uploaded file is empty.")
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise FileTooLargeError(
                f"File is larger than the {self.settings.max_upload_mb} MB limit.")
        if b"%PDF-" not in data[:1024]:
            raise DocumentProcessingError("File does not look like a valid PDF.")

    def _persist(self, store: FaissVectorStore, info: DocumentInfo) -> None:
        """Write to a temp dir, then move files into index_dir."""
        index_dir = self.settings.index_dir
        index_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-index-", dir=index_dir.parent))
        try:
            store.save(tmp)
            (tmp / INFO_FILE).write_text(json.dumps(asdict(info)), encoding="utf-8")
            index_dir.mkdir(parents=True, exist_ok=True)
            for fname in (INDEX_FILE, CHUNKS_FILE, INFO_FILE):
                os.replace(tmp / fname, index_dir / fname)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _remove_old_uploads(self, keep: str) -> None:
        for f in self.settings.upload_dir.glob("*.pdf"):
            if f.name != keep and not f.name.startswith(".incoming-"):
                f.unlink(missing_ok=True)


def _embed_in_batches(embedder: EmbeddingService, texts: list[str]):
    import numpy as np
    parts = [embedder.embed_documents(texts[i:i + EMBED_BATCH])
             for i in range(0, len(texts), EMBED_BATCH)]
    vectors = np.vstack(parts)
    if vectors.shape[0] != len(texts):
        raise EmbeddingError("Embedding count does not match chunk count.")
    return vectors
