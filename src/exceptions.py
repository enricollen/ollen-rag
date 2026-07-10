"""Domain exception hierarchy; each error maps to an HTTP status and stable error code."""

class OllenRagError(Exception):
    """Base class for all ollen-rag-service domain errors."""
    error_code: str = "OLLEN_RAG_ERROR"
    status_code: int = 500

class ParsingError(OllenRagError):
    """Document could not be parsed (unsupported/corrupted/empty content)."""
    error_code = "PARSING_ERROR"
    status_code = 422

class EmbeddingError(OllenRagError):
    """Embedding provider failed to produce vectors."""
    error_code = "EMBEDDING_ERROR"
    status_code = 502

class VectorStoreError(OllenRagError):
    """OpenSearch (or other vector store) operation failed."""
    error_code = "VECTOR_STORE_ERROR"
    status_code = 502

class GenerationError(OllenRagError):
    """LLM generation failed."""
    error_code = "GENERATION_ERROR"
    status_code = 502

class PromptNotFoundError(OllenRagError):
    """Requested prompt template does not exist on disk."""
    error_code = "PROMPT_NOT_FOUND"
    status_code = 404