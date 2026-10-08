import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter

from backend.document_loader import extract_pdf_pages
from backend.exceptions import (DocumentProcessingError, EmptyDocumentError,
                                UnsupportedFileTypeError)

try:
    from reportlab.pdfgen import canvas
except ImportError:  # dev-only dependency
    canvas = None


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_wrong_extension(self):
        f = self.tmp / "a.txt"
        f.write_text("hi")
        with self.assertRaises(UnsupportedFileTypeError):
            extract_pdf_pages(f)

    def test_missing_file(self):
        with self.assertRaises(DocumentProcessingError):
            extract_pdf_pages(self.tmp / "nope.pdf")

    def test_corrupt_pdf(self):
        f = self.tmp / "bad.pdf"
        f.write_bytes(b"this is not a pdf")
        with self.assertRaises(DocumentProcessingError):
            extract_pdf_pages(f)

    def test_blank_pdf_is_empty_document(self):
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        f = self.tmp / "blank.pdf"
        with open(f, "wb") as fh:
            w.write(fh)
        with self.assertRaises(EmptyDocumentError):
            extract_pdf_pages(f)

    @unittest.skipIf(canvas is None, "reportlab not installed")
    def test_text_pdf_keeps_page_numbers(self):
        f = self.tmp / "ok.pdf"
        c = canvas.Canvas(str(f))
        c.drawString(72, 700, "Casual leave is twelve days per year for employees.")
        c.showPage()
        c.drawString(72, 700, "Notice period is thirty days for resignation requests.")
        c.save()
        pages = extract_pdf_pages(f)
        self.assertEqual([p.page_number for p in pages], [1, 2])
        self.assertIn("twelve days", pages[0].text)


if __name__ == "__main__":
    unittest.main()
