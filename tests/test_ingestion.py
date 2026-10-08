"""Ingestion tests. Use FakeEmbeddingService (NOT the real HF model) and the
numpy FAISS stand-in if real faiss is missing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.fakes import FakeEmbeddingService, install_fake_faiss
from backend.config import Settings
from backend.exceptions import (DocumentProcessingError, EmptyDocumentError,
                                FileTooLargeError, NoDocumentError,
                                UnsupportedFileTypeError)
from backend.ingestion import DocumentManager, safe_filename

install_fake_faiss()
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_policy.pdf"


def blank_pdf_bytes() -> bytes:
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    w.add_blank_page(200, 200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.settings = Settings(upload_dir=self.tmp / "up", index_dir=self.tmp / "idx",
                                 chunk_size=200, chunk_overlap=30, top_k=2,
                                 max_upload_mb=1)
        self.emb = FakeEmbeddingService()
        self.mgr = DocumentManager(self.settings, self.emb)
        self.data = SAMPLE.read_bytes()

    def test_ingest_then_retrieve(self):
        info, replaced = self.mgr.ingest_pdf("policy.pdf", self.data)
        self.assertFalse(replaced)
        self.assertEqual((info.filename, info.pages), ("policy.pdf", 2))
        self.assertGreater(info.chunks, 0)
        res = self.mgr.retriever().retrieve("How many days of casual leave?")
        self.assertEqual(res[0].chunk.source, "policy.pdf")
        self.assertEqual(res[0].chunk.page_number, 1)

    def test_no_document_yet(self):
        with self.assertRaises(NoDocumentError):
            self.mgr.retriever()

    def test_second_upload_replaces_first(self):
        self.mgr.ingest_pdf("a.pdf", self.data)
        info, replaced = self.mgr.ingest_pdf("b.pdf", self.data)
        self.assertTrue(replaced)
        self.assertEqual(self.mgr.info.filename, "b.pdf")
        self.assertEqual({p.name for p in self.settings.upload_dir.glob("*.pdf")}, {"b.pdf"})
        sources = {r.chunk.source for r in self.mgr.retriever().retrieve("leave")}
        self.assertEqual(sources, {"b.pdf"})

    def test_persistence_survives_restart(self):
        self.mgr.ingest_pdf("policy.pdf", self.data)
        fresh = DocumentManager(self.settings, self.emb)
        self.assertTrue(fresh.load_existing())
        self.assertEqual(fresh.info.filename, "policy.pdf")
        self.assertTrue(fresh.retriever().retrieve("remote work"))

    def test_load_existing_with_nothing_saved(self):
        self.assertFalse(self.mgr.load_existing())

    def test_failed_upload_keeps_previous_document(self):
        self.mgr.ingest_pdf("good.pdf", self.data)
        with self.assertRaises(EmptyDocumentError):
            self.mgr.ingest_pdf("blank.pdf", blank_pdf_bytes())
        self.assertEqual(self.mgr.info.filename, "good.pdf")
        self.assertTrue(self.mgr.retriever().retrieve("leave"))
        # no leftover temp files
        self.assertEqual([p.name for p in self.settings.upload_dir.iterdir()], ["good.pdf"])

    def test_validation_errors(self):
        with self.assertRaises(UnsupportedFileTypeError):
            self.mgr.ingest_pdf("notes.txt", b"hello")
        with self.assertRaises(DocumentProcessingError):
            self.mgr.ingest_pdf("x.pdf", b"")
        with self.assertRaises(DocumentProcessingError):
            self.mgr.ingest_pdf("x.pdf", b"this is not a pdf")
        with self.assertRaises(DocumentProcessingError):  # header ok, body corrupt
            self.mgr.ingest_pdf("x.pdf", b"%PDF-1.4\ngarbage garbage")
        with self.assertRaises(FileTooLargeError):
            self.mgr.ingest_pdf("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 1))

    def test_housekeeping_failure_does_not_undo_a_successful_index(self):
        with patch.object(DocumentManager, "_remove_old_uploads", side_effect=OSError("locked")):
            info, _ = self.mgr.ingest_pdf("policy.pdf", self.data)
        self.assertEqual(self.mgr.info.filename, "policy.pdf")
        self.assertTrue(self.mgr.retriever().retrieve("remote work"))
        fresh = DocumentManager(self.settings, self.emb)  # disk agrees with memory
        self.assertTrue(fresh.load_existing())
        self.assertEqual(fresh.info.filename, "policy.pdf")

    def test_safe_filename(self):
        self.assertEqual(safe_filename("../../etc/passwd.pdf"), "passwd.pdf")
        self.assertEqual(safe_filename("My Report (final).PDF"), "My_Report_final.pdf")
        self.assertEqual(safe_filename("C:\\docs\\a b.pdf"), "a_b.pdf")
        self.assertEqual(safe_filename(""), "document.pdf")


if __name__ == "__main__":
    unittest.main()
