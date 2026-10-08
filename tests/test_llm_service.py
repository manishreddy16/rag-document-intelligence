import io
import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from backend.config import Settings
from backend.exceptions import (LLMConfigError, LLMError, LLMRateLimitError,
                                LLMUnavailableError)
from backend.llm_service import GeminiLLMService, create_llm_service

KEY = "SECRET-KEY-123"


def fake_response(payload: dict):
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    return ctx


def http_error(code, body=b"{}"):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body))


class GeminiTests(unittest.TestCase):
    def svc(self, key=KEY, model="some-model", max_retries=0):
        # max_retries=0 keeps the pre-existing tests single-attempt and instant.
        return GeminiLLMService(key, model, timeout=5, max_retries=max_retries)

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_success_and_request_shape(self, urlopen):
        urlopen.return_value = fake_response(
            {"candidates": [{"content": {"parts": [{"text": "Twelve "}, {"text": "days."}]}}]})
        self.assertEqual(self.svc().generate("hi"), "Twelve days.")
        req = urlopen.call_args[0][0]
        self.assertIn("some-model:generateContent", req.full_url)
        self.assertNotIn(KEY, req.full_url)  # key must not be in the URL
        self.assertEqual(req.get_header("X-goog-api-key"), KEY)
        self.assertEqual(json.loads(req.data)["contents"][0]["parts"][0]["text"], "hi")

    def test_missing_key_and_model(self):
        with self.assertRaises(LLMError):
            self.svc(key="").generate("x")
        with self.assertRaises(LLMError):
            self.svc(model="").generate("x")

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_http_errors_are_friendly_and_hide_key(self, urlopen):
        for code, word in [(429, "Rate limit"), (403, "key"), (404, "Model not found")]:
            urlopen.side_effect = http_error(code)
            with self.assertRaises(LLMError) as cm:
                self.svc().generate("x")
            self.assertIn(word, str(cm.exception))
            self.assertNotIn(KEY, str(cm.exception))

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_error_subclasses_for_api_mapping(self, urlopen):
        for code, cls in [(429, LLMRateLimitError), (403, LLMConfigError),
                          (404, LLMConfigError), (500, LLMUnavailableError)]:
            urlopen.side_effect = http_error(code)
            with self.assertRaises(cls):
                self.svc().generate("x")
        urlopen.side_effect = urllib.error.URLError("down")
        with self.assertRaises(LLMUnavailableError):
            self.svc().generate("x")
        with self.assertRaises(LLMConfigError):
            self.svc(key="").generate("x")

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_network_failure(self, urlopen):
        urlopen.side_effect = urllib.error.URLError("down")
        with self.assertRaises(LLMError):
            self.svc().generate("x")

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_malformed_and_empty_responses(self, urlopen):
        resp = MagicMock()
        resp.read.return_value = b"not json"
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        urlopen.return_value = ctx
        with self.assertRaises(LLMError):
            self.svc().generate("x")
        for payload in ({}, {"promptFeedback": {"blockReason": "SAFETY"}},
                        {"candidates": [{"content": {"parts": []}}]}):
            urlopen.return_value = fake_response(payload)
            with self.assertRaises(LLMError):
                self.svc().generate("x")

    def test_factory(self):
        self.assertIsInstance(create_llm_service(Settings(llm_provider="gemini")),
                              GeminiLLMService)
        with self.assertRaises(LLMError):
            create_llm_service(Settings(llm_provider="nope"))


OK = {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}


class RetryTests(unittest.TestCase):
    def svc(self, max_retries=3):
        self.sleeps = []
        return GeminiLLMService(KEY, "m", timeout=5, max_retries=max_retries,
                                backoff_seconds=1.0, sleep=self.sleeps.append)

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_transient_status_then_success(self, urlopen):
        for code in (503, 429, 408, 500, 502, 504):
            urlopen.reset_mock()
            urlopen.side_effect = [http_error(code), fake_response(OK)]
            self.assertEqual(self.svc().generate("x"), "done")
            self.assertEqual(urlopen.call_count, 2)
            self.assertEqual(len(self.sleeps), 1)

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_network_error_is_retried(self, urlopen):
        urlopen.side_effect = [urllib.error.URLError("down"), TimeoutError(), fake_response(OK)]
        self.assertEqual(self.svc().generate("x"), "done")
        self.assertEqual(urlopen.call_count, 3)

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_exponential_backoff_and_bounded_attempts(self, urlopen):
        urlopen.side_effect = [http_error(503)] * 10
        with self.assertRaises(LLMUnavailableError):
            self.svc(max_retries=3).generate("x")
        self.assertEqual(urlopen.call_count, 4)  # 1 initial + 3 retries, no more
        self.assertEqual(len(self.sleeps), 3)
        for got, base in zip(self.sleeps, (1, 2, 4)):
            self.assertGreaterEqual(got, base)
            self.assertLessEqual(got, base + 0.25)  # jitter is small

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_permanent_errors_are_not_retried(self, urlopen):
        for code, cls in [(400, LLMConfigError), (401, LLMConfigError),
                          (403, LLMConfigError), (404, LLMConfigError)]:
            urlopen.reset_mock()
            urlopen.side_effect = http_error(code)
            with self.assertRaises(cls):
                self.svc().generate("x")
            self.assertEqual(urlopen.call_count, 1)
        self.assertEqual(self.sleeps, [])

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_missing_key_makes_no_request(self, urlopen):
        svc = GeminiLLMService("", "m", max_retries=3, sleep=lambda s: None)
        with self.assertRaises(LLMConfigError):
            svc.generate("x")
        urlopen.assert_not_called()

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_exhausted_429_keeps_rate_limit_type(self, urlopen):
        urlopen.side_effect = [http_error(429)] * 4
        with self.assertRaises(LLMRateLimitError):
            self.svc().generate("x")

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_zero_retries_means_single_attempt(self, urlopen):
        urlopen.side_effect = http_error(503)
        with self.assertRaises(LLMUnavailableError):
            self.svc(max_retries=0).generate("x")
        self.assertEqual(urlopen.call_count, 1)

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_malformed_response_not_retried(self, urlopen):
        resp = MagicMock()
        resp.read.return_value = b"not json"
        ctx = MagicMock()
        ctx.__enter__.return_value = resp
        urlopen.return_value = ctx
        with self.assertRaises(LLMUnavailableError):
            self.svc().generate("x")
        self.assertEqual(urlopen.call_count, 1)

    @patch("backend.llm_service.urllib.request.urlopen")
    def test_key_not_in_errors_or_logs(self, urlopen):
        urlopen.side_effect = [http_error(503)] * 4
        with self.assertLogs("backend.llm_service", level="WARNING") as logs:
            with self.assertRaises(LLMError) as cm:
                self.svc().generate("x")
        self.assertNotIn(KEY, str(cm.exception))
        self.assertNotIn(KEY, "\n".join(logs.output))

    def test_factory_passes_retry_setting(self):
        svc = create_llm_service(Settings(llm_provider="gemini", llm_max_retries=5))
        self.assertEqual(svc.max_retries, 5)


if __name__ == "__main__":
    unittest.main()
