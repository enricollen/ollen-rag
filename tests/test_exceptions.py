"""Tests for domain exception hierarchy and its HTTP mapping attributes."""
from src.exceptions import (
    OllenRagError, ParsingError, EmbeddingError,
    VectorStoreError, GenerationError, PromptNotFoundError,
)

def test_exception_hierarchy_and_codes():
    """Every domain error carries an error_code and an HTTP status_code."""
    cases = [
        (ParsingError, "PARSING_ERROR", 422),
        (EmbeddingError, "EMBEDDING_ERROR", 502),
        (VectorStoreError, "VECTOR_STORE_ERROR", 502),
        (GenerationError, "GENERATION_ERROR", 502),
        (PromptNotFoundError, "PROMPT_NOT_FOUND", 404),
    ]
    for cls, code, status in cases:
        exc = cls("boom")
        assert isinstance(exc, OllenRagError)
        assert exc.error_code == code
        assert exc.status_code == status
        assert str(exc) == "boom"