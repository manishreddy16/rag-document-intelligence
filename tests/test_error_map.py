import unittest

from backend import exceptions as ex
from backend.error_map import status_for


class ErrorMapTests(unittest.TestCase):
    def test_statuses(self):
        cases = [(ex.UnsupportedFileTypeError, 415), (ex.FileTooLargeError, 413),
                 (ex.EmptyDocumentError, 422), (ex.DocumentProcessingError, 400),
                 (ex.InvalidQuestionError, 422), (ex.NoDocumentError, 409),
                 (ex.RetrievalError, 400), (ex.EmbeddingError, 503),
                 (ex.VectorStoreError, 500), (ex.LLMRateLimitError, 429),
                 (ex.LLMConfigError, 503), (ex.LLMUnavailableError, 502),
                 (ex.LLMError, 502), (ex.RAGError, 400)]
        for cls, status in cases:
            self.assertEqual(status_for(cls("x"))[0], status, cls.__name__)

    def test_unknown_exception_is_500(self):
        self.assertEqual(status_for(RuntimeError("x")), (500, "internal_error"))


if __name__ == "__main__":
    unittest.main()
