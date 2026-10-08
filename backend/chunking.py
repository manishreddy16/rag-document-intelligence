"""Split page text into overlapping chunks that keep page metadata.

Chunks never cross page boundaries, so page numbers stay accurate.
"""
import re

from .models import Chunk, PageText

# Preferred break points, strongest first.
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Need chunk_size > 0 and 0 <= overlap < chunk_size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    chunks, seen, start = [], set(), 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _best_break(text, start, end)
        piece = text[start:end].strip()
        if piece and piece not in seen:  # skip empty / duplicate chunks
            seen.add(piece)
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)  # always move forward
    return chunks


def _best_break(text: str, start: int, end: int) -> int:
    """Pick the latest natural boundary in the second half of the window."""
    floor = start + (end - start) // 2
    for sep in _SEPARATORS:
        idx = text.rfind(sep, floor, end)
        if idx != -1:
            return idx + len(sep)
    return end  # no boundary found: hard cut


def chunk_pages(pages: list[PageText], source: str,
                chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for n, piece in enumerate(
                split_text(page.text, chunk_size, chunk_overlap), start=1):
            chunks.append(Chunk(
                chunk_id=f"{source}-p{page.page_number}-c{n}",
                text=piece, source=source, page_number=page.page_number))
    return chunks
