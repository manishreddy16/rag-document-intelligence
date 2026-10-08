"""PDF text extraction (pypdf). Only text-based PDFs are supported; no OCR."""
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .exceptions import (DocumentProcessingError, EmptyDocumentError,
                         UnsupportedFileTypeError)
from .models import PageText


def extract_pdf_pages(path: Path, min_chars: int = 50) -> list[PageText]:
    """Return non-empty pages with real page numbers.

    Raises a RAGError subclass for every expected failure.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise UnsupportedFileTypeError("Only .pdf files are supported.")
    if not path.is_file():
        raise DocumentProcessingError("File not found.")
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise DocumentProcessingError("Encrypted PDFs are not supported.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(PageText(page_number=number, text=text))
    except DocumentProcessingError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError) as exc:
        raise DocumentProcessingError(f"Could not read PDF: {exc}") from exc

    if sum(len(p.text) for p in pages) < min_chars:
        raise EmptyDocumentError(
            "No extractable text found. The PDF may be scanned/image-based "
            "(OCR is not supported).")
    return pages
