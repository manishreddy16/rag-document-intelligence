"""API tests via FastAPI TestClient with FAKE embedder + FAKE LLM.

These are skipped automatically if fastapi/httpx are not installed. They were
WRITTEN in a sandbox without FastAPI, so they have NOT BEEN RUN by the author
of this checkpoint: run them on your machine (pip install -r requirements-dev.txt).
"""
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    HAVE_API = True
except ImportError:  # pragma: no cover
    HAVE_API = False

from tests.fakes import FakeEmbeddingService, install_fake_faiss
from backend.config import Settings
from backend.exceptions import LLMConfigError, LLMRateLimitError, LLMUnavailableError

install_fake_faiss()
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_policy.pdf"


class FakeLLM:
    def __init__(self):
        self.error = None
        self.reply = "Employees get 12 days of casual leave."

    def generate(self, prompt):
        if self.error:
            raise self.error
        return self.reply


@unittest.skipUnless(HAVE_API, "fastapi / httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from backend.api import create_app
        tmp = Path(tempfile.mkdtemp())
        self.settings = Settings(upload_dir=tmp / "up", index_dir=tmp / "idx",
                                 chunk_size=200, chunk_overlap=30, top_k=2,
                                 max_upload_mb=1, llm_api_key="SECRET-KEY-123",
                                 llm_model="m")
        self.llm = FakeLLM()
        self.client = TestClient(create_app(self.settings, FakeEmbeddingService(), self.llm),
                                 raise_server_exceptions=False)

    def upload(self, name="policy.pdf", data=None, ctype="application/pdf"):
        data = SAMPLE.read_bytes() if data is None else data
        return self.client.post("/upload", files={"file": (name, data, ctype)})

    def test_health_empty_then_loaded(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual((body["status"], body["document_loaded"]), ("ok", False))
        self.assertTrue(body["llm_configured"])
        self.assertNotIn("SECRET-KEY-123", r.text)
        self.upload()
        self.assertTrue(self.client.get("/health").json()["document_loaded"])

    def test_upload_success_schema(self):
        r = self.upload()
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["document"]["filename"], "policy.pdf")
        self.assertEqual(body["document"]["pages"], 2)
        self.assertFalse(body["replaced_previous"])
        self.assertTrue(self.upload("other.pdf").json()["replaced_previous"])

    def test_upload_errors(self):
        r = self.upload("notes.txt", b"hello", "text/plain")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (415, "unsupported_file_type"))
        r = self.upload("x.pdf", b"%PDF-1.4\ngarbage")
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (400, "invalid_document"))
        r = self.upload("big.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 1))
        self.assertEqual(r.status_code, 413)
        r = self.client.post("/upload")  # no file field
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (422, "validation_error"))

    def test_ask_without_document(self):
        r = self.client.post("/ask", json={"question": "hi"})
        self.assertEqual((r.status_code, r.json()["error"]["code"]), (409, "no_document"))

    def test_ask_success_with_sources(self):
        self.upload()
        r = self.client.post("/ask", json={"question": "How many days of casual leave?"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["answer"], self.llm.reply)
        s = body["sources"][0]
        self.assertEqual((s["source"], s["page_number"]), ("policy.pdf", 1))
        self.assertEqual(set(s), {"source", "page_number", "chunk_id", "score", "snippet"})

    def test_invalid_questions(self):
        self.upload()
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/ask", json={"question": "   "}).status_code, 422)
        self.assertEqual(self.client.post("/ask", json={}).status_code, 422)
        long_q = "x" * (self.settings.max_question_chars + 1)
        self.assertEqual(self.client.post("/ask", json={"question": long_q}).status_code, 422)

    def test_llm_failures_map_to_status_codes(self):
        self.upload()
        for exc, status in [(LLMRateLimitError("slow down"), 429),
                            (LLMConfigError("no key"), 503),
                            (LLMUnavailableError("down"), 502)]:
            self.llm.error = exc
            r = self.client.post("/ask", json={"question": "casual leave"})
            self.assertEqual(r.status_code, status)

    def test_secret_is_redacted_from_errors(self):
        self.upload()
        self.llm.error = LLMUnavailableError("provider said key SECRET-KEY-123 is bad")
        r = self.client.post("/ask", json={"question": "casual leave"})
        self.assertNotIn("SECRET-KEY-123", r.text)

    def test_unexpected_error_is_generic_500(self):
        self.upload()
        self.llm.error = RuntimeError("internal detail SECRET-KEY-123")
        r = self.client.post("/ask", json={"question": "casual leave"})
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("internal detail", r.text)


if __name__ == "__main__":
    unittest.main()
