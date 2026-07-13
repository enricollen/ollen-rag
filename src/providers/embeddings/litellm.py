"""LiteLLM-backed embedding providers: one call surface for every vendor LiteLLM supports.

Mirrors src/providers/llm/litellm.py. The adapter class owns the vendor-neutral part (the embedding
call, response unwrapping, error mapping); each registered builder supplies only a model string plus
that vendor's credentials, because credential shape -- not routing -- is what actually differs.
A vendor needing only api_key + api_base needs no builder of its own and can use the generic
"litellm" provider.
"""
import os
from typing import Any

# litellm's __init__ calls dotenv.load_dotenv() at import time whenever LITELLM_MODE is "DEV"
# (its default), which would splice this project's whole .env -- OLLEN_RAG_WATSONX_APIKEY included --
# into os.environ, leaking secrets to every subprocess and defeating Settings(_env_file=None) in
# tests. Settings already owns .env loading, so opt out. Must precede the litellm import.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

from litellm import embedding  # noqa: E402
from llama_index.core.base.embeddings.base import BaseEmbedding  # noqa: E402
from llama_index.core.bridge.pydantic import PrivateAttr  # noqa: E402
from src.exceptions import EmbeddingError  # noqa: E402
from src.factories.embeddings import EmbeddingFactory  # noqa: E402
from src.settings import Settings  # noqa: E402

class LiteLLMEmbedding(BaseEmbedding):
    """Adapts litellm.embedding() to the llamaindex BaseEmbedding interface.

    *call_kwargs* carries the routed model string and this vendor's credentials, exactly as the
    builder assembled them; they are forwarded to every call verbatim.
    """
    _call_kwargs: dict[str, Any] = PrivateAttr()

    def __init__(self, model_name: str, call_kwargs: dict[str, Any], **kwargs: Any) -> None:
        # Initialize the pydantic base first, then attach the private call kwargs.
        super().__init__(model_name=model_name, **kwargs)
        self._call_kwargs = call_kwargs

    @classmethod
    def class_name(cls) -> str:
        """Identifier used by llamaindex serialization."""
        return "LiteLLMEmbedding"

    def _is_asymmetric(self) -> bool:
        """Cohere v3 embeddings are asymmetric: query and document must be embedded with different
        input_type values ('search_query' vs 'search_document') or nearest-neighbour search
        degrades badly. LiteLLM forwards a caller-supplied input_type, and otherwise defaults to
        'search_document' for both sides -- which silently breaks query retrieval. Only Cohere
        needs this; every other vendor is left untouched."""
        return self.model_name.lower().startswith("cohere/")

    def _embed(self, texts: list[str], input_type: str | None = None) -> list[list[float]]:
        """One batched litellm call; the vendor-neutral core every other method delegates to.
        *input_type* is passed through only for vendors that need it (see _is_asymmetric)."""
        call_kwargs = dict(self._call_kwargs)
        if input_type and self._is_asymmetric():
            call_kwargs["input_type"] = input_type
        try:
            response = embedding(input=texts, **call_kwargs)
        except Exception as exc:
            # The raised type depends on which vendor SDK handled the call, so an explicit
            # except-list would be long and incomplete; the vendor message survives in the cause.
            raise EmbeddingError(f"LiteLLM embedding failed ({self.model_name}): {exc}") from exc
        # data entries are OpenAI-shaped: {"object", "index", "embedding"}
        return [[float(x) for x in item["embedding"]] for item in response.data]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text], input_type="search_document")[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        # Batch variant, used for indexing multiple chunks at once: one HTTP round trip.
        return self._embed(texts, input_type="search_document")

    def _get_query_embedding(self, query: str) -> list[float]:
        # Queries embed on the query side of the asymmetric encoder (Cohere v3); other vendors
        # ignore input_type, so a query still embeds like any other text there.
        return self._embed([query], input_type="search_query")[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        # litellm has an async API, but the rest of this project is sync; delegate.
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        # litellm has an async API, but the rest of this project is sync; delegate.
        return self._get_text_embedding(text)

def _optional(kwargs: dict[str, Any], **maybe: str) -> dict[str, Any]:
    """Add only the credentials that were actually configured; LiteLLM treats an empty
    api_key as a real (invalid) key rather than as absent."""
    kwargs.update({key: value for key, value in maybe.items() if value})
    return kwargs

@EmbeddingFactory.register("litellm", model_field="litellm_embedding_model")
def create_litellm_embedding(settings: Settings) -> BaseEmbedding:
    """Any LiteLLM vendor whose credentials are just an api_key and/or an api_base.

    The model string is taken verbatim from OLLEN_RAG_LITELLM_EMBEDDING_MODEL, so no code change
    is needed to reach a new vendor.
    """
    if not settings.litellm_embedding_model:
        raise ValueError("OLLEN_RAG_LITELLM_EMBEDDING_MODEL must be set when OLLEN_RAG_EMBEDDING_PROVIDER=litellm")
    model = settings.litellm_embedding_model
    call_kwargs = _optional(
        {"model": model},
        api_base=settings.effective_litellm_embedding_api_base,
        api_key=settings.effective_litellm_embedding_api_key,
    )
    return LiteLLMEmbedding(model_name=model, call_kwargs=call_kwargs)

@EmbeddingFactory.register("litellm-watsonx", model_field="watsonx_embedding_model_id")
def create_litellm_watsonx_embedding(settings: Settings) -> BaseEmbedding:
    """watsonx.ai embeddings through LiteLLM rather than the native llama-index-embeddings-ibm.

    Reads the same OLLEN_RAG_WATSONX_* settings as the native provider, so switching between the
    two is a one-line .env change. The stored model id stays bare -- the "watsonx/" prefix is added
    here, at call time -- which is what lets one index be queried by either provider.
    """
    if not settings.watsonx_project_id:
        raise ValueError("OLLEN_RAG_WATSONX_PROJECT_ID must be set when OLLEN_RAG_EMBEDDING_PROVIDER=litellm-watsonx")
    model = settings.watsonx_embedding_model_id
    call_kwargs = {
        "model": f"watsonx/{model}",
        "api_key": settings.watsonx_apikey,
        "api_base": settings.watsonx_url,
        "project_id": settings.watsonx_project_id,
    }
    return LiteLLMEmbedding(model_name=model, call_kwargs=call_kwargs)

@EmbeddingFactory.register("litellm-openai", model_field="openai_embedding_model")
def create_litellm_openai_embedding(settings: Settings) -> BaseEmbedding:
    """OpenAI or any OpenAI-compatible embedding API through LiteLLM.

    Uses the dedicated OLLEN_RAG_OPENAI_* settings. The bare model name is prefixed with
    "openai/" unless already included. Setting OLLEN_RAG_OPENAI_API_BASE routes to any
    OpenAI-compatible server (vLLM, LocalAI, …) instead of the official OpenAI endpoint.
    """
    if not settings.openai_embedding_model:
        raise ValueError(
            "OLLEN_RAG_OPENAI_EMBEDDING_MODEL must be set when OLLEN_RAG_EMBEDDING_PROVIDER=litellm-openai"
        )
    raw = settings.openai_embedding_model
    model = raw if raw.startswith("openai/") else f"openai/{raw}"
    call_kwargs = _optional(
        {"model": model},
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
    )
    return LiteLLMEmbedding(model_name=raw, call_kwargs=call_kwargs)

@EmbeddingFactory.register("litellm-ollama", model_field="ollama_embedding_model")
def create_litellm_ollama_embedding(settings: Settings) -> BaseEmbedding:
    """Local Ollama embeddings through LiteLLM. Needs only an api_base -- no credentials at all.

    Uses its own model tag: OLLEN_RAG_OLLAMA_MODEL is a chat model and cannot embed.
    """
    model = settings.ollama_embedding_model
    # Pull the embedding model on first use if the local Ollama doesn't have it yet (the bundled
    # Ollama only pre-pulls the chat model), so selecting it never fails with 'model not found'.
    from src.providers.ollama import ensure_model
    ensure_model(settings.ollama_api_base, model)
    call_kwargs = {"model": f"ollama/{model}", "api_base": settings.ollama_api_base}
    return LiteLLMEmbedding(model_name=model, call_kwargs=call_kwargs)
