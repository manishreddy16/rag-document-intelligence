import unittest
from backend.chunking import chunk_pages, split_text
from backend.models import PageText


class ChunkingTests(unittest.TestCase):
    def test_respects_size_and_makes_progress(self):
        text = " ".join(f"word{i}" for i in range(400))
        parts = split_text(text, 200, 40)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 200 for p in parts))

    def test_overlap_shares_text(self):
        text = ". ".join(f"Sentence number {i}" for i in range(60))
        parts = split_text(text, 200, 50)
        self.assertTrue(any(parts[i][-15:] in parts[i + 1]
                            for i in range(len(parts) - 1)))

    def test_short_text_single_chunk(self):
        self.assertEqual(split_text("Hello world", 100, 10), ["Hello world"])

    def test_empty_text(self):
        self.assertEqual(split_text("   \n ", 100, 10), [])

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            split_text("abc", 10, 10)

    def test_no_duplicate_chunks(self):
        parts = split_text("abcde " * 100, 12, 0)
        self.assertEqual(len(parts), len(set(parts)))

    def test_metadata_and_ids(self):
        pages = [PageText(1, "First page text. " * 30),
                 PageText(3, "Third page text. " * 30)]
        chunks = chunk_pages(pages, "doc.pdf", 200, 20)
        self.assertEqual({c.page_number for c in chunks}, {1, 3})
        self.assertTrue(all(c.source == "doc.pdf" for c in chunks))
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
        self.assertEqual(chunks[0].chunk_id, "doc.pdf-p1-c1")


if __name__ == "__main__":
    unittest.main()
