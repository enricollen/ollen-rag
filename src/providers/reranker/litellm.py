"""LiteLLM-backed rerank connectors: one call surface for every rerank vendor LiteLLM supports
(cohere, jina_ai, azure_ai, together_ai, voyage, infinity, watsonx, ...).

Mirrors src/providers/llm/litellm.py. Ollama has no rerank endpoint, so there is no
litellm-ollama connector here; use the local "sentence-transformers" provider instead.
"""
import os
from typing import Any

# litellm's __init__ calls dotenv.load_dotenv() at import time whenever LITELLM_MODE is "DEV"
# (its default), which would splice this project's whole .env -- OLLEN_RAG_WATSONX_APIKEY included --
# into os.environ, leaking secrets to every subprocess and defeating Settings(_env_file=None) in
# tests. Settings already owns .env loading, so opt out. Must precede the litellm import.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

from litellm import rerank  # noqa: E402
from llama_index.core.schema import MetadataMode, NodeWithScore  # noqa: E402
from src.exceptions import RerankError  # noqa: E402
from src.factories.reranker import RerankConnector, RerankerFactory, to_probability  # noqa: E402
from src.settings import Settings  # noqa: E402

class BaseLiteLLMRerankConnector(RerankConnector):
    """Base connector: scores nodes through litellm.rerank(), which follows the Cohere contract.

    LiteLLM returns results already sorted by descending relevance_score. That contract fixes the
    *shape* of the response, not the *scale* of the score: Cohere and Jina return a 0-1
    probability, while watsonx returns the raw cross-encoder logit. scores_are_logits says which,
    and the base class normalizes accordingly so every connector honors RerankConnector's 0-1
    contract. Sigmoiding an already-calibrated score would squash it into 0.5-0.73.

    Subclasses set model_name in __init__ and override _call_kwargs() to supply the routed model
    string and credentials, which LiteLLM forwards to the vendor verbatim.
    """
    # Whether this vendor's relevance_score is an unbounded logit rather than a 0-1 probability.
    scores_are_logits: bool = False

    def _call_kwargs(self) -> dict[str, Any]:
        """Vendor-specific model string and credentials, passed straight to litellm.rerank()."""
        raise NotImplementedError

    def _score(self, relevance_score: float) -> float:
        """Normalize one vendor score to the 0-1 relevance probability the contract promises."""
        return to_probability(float(relevance_score)) if self.scores_are_logits else float(relevance_score)

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        """Send every node's text as a document and rebuild the ranking from the response indices."""
        documents = [node.node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
        try:
            response = rerank(
                query=query,
                documents=documents,
                top_n=top_n,
                # The documents are already in hand; echoing them back wastes bandwidth.
                return_documents=False,
                **self._call_kwargs(),
            )
        except Exception as exc:
            # The raised type depends on which vendor SDK handled the call, so an explicit
            # except-list would be long and incomplete; the vendor message survives in the cause.
            raise RerankError(f"LiteLLM rerank failed ({self.model_name}): {exc}") from exc
        # RerankResponseResult is a TypedDict (litellm/types/rerank.py), so subscript, not getattr:
        # result["index"] == 0 is a valid first-node hit and must not be read as missing.
        return [
            NodeWithScore(node=nodes[result["index"]].node, score=self._score(result["relevance_score"]))
            for result in response.results or []
        ]

@RerankerFactory.register("litellm", model_field="litellm_rerank_model")
class LiteLLMRerankConnector(BaseLiteLLMRerankConnector):
    """Any LiteLLM rerank vendor whose credentials are just an api_key and/or an api_base.

    The model string is taken verbatim from OLLEN_RAG_LITELLM_RERANK_MODEL, so no code change is
    needed to reach a new vendor.

    Assumes a 0-1 relevance_score, which is what Cohere, Jina, Voyage and the other hosted rerank
    APIs return. Pointing this provider at a "watsonx/..." model string would yield raw logits;
    use the dedicated litellm-watsonx provider instead, which knows to normalize them.
    """
    scores_are_logits = False

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        if not self._settings.litellm_rerank_model:
            raise ValueError("OLLEN_RAG_LITELLM_RERANK_MODEL must be set when OLLEN_RAG_RERANKER_PROVIDER=litellm")
        self.model_name = self._settings.litellm_rerank_model

    def _call_kwargs(self) -> dict[str, Any]:
        """Send only the credentials that were actually configured; LiteLLM treats an empty
        api_key as a real (invalid) key rather than as absent."""
        kwargs: dict[str, Any] = {"model": self.model_name}
        if self._settings.effective_litellm_rerank_api_base:
            kwargs["api_base"] = self._settings.effective_litellm_rerank_api_base
        if self._settings.effective_litellm_rerank_api_key:
            kwargs["api_key"] = self._settings.effective_litellm_rerank_api_key
        return kwargs

@RerankerFactory.register("litellm-watsonx", model_field="watsonx_reranker_model_id")
class LiteLLMWatsonxRerankConnector(BaseLiteLLMRerankConnector):
    """watsonx.ai /ml/v1/text/rerank through LiteLLM, reading the same OLLEN_RAG_WATSONX_* block as
    the LLM and embedding connectors.

    project_id, api_key and api_base travel as call kwargs, never as unprefixed WATSONX_* process
    env vars: litellm's rerank entrypoint forwards unrecognized kwargs into the watsonx request
    body, so the env-var route its docs demonstrate is unnecessary here.

    watsonx returns the cross-encoder's raw logit as relevance_score, not the 0-1 probability the
    other rerank vendors return -- a live call scored two passages at 6.902 and -0.0005. The model
    behind it (cross-encoder/ms-marco-minilm-l-12-v2) is a single-label BCE cross-encoder, exactly
    like the ones the sentence-transformers provider loads locally, so sigmoid is its calibrated
    output and the base class applies it.
    """
    scores_are_logits = True

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        if not self._settings.watsonx_project_id:
            raise ValueError("OLLEN_RAG_WATSONX_PROJECT_ID must be set when OLLEN_RAG_RERANKER_PROVIDER=litellm-watsonx")
        self.model_name = self._settings.watsonx_reranker_model_id

    def _call_kwargs(self) -> dict[str, Any]:
        """watsonx needs project_id alongside the key and endpoint; all three travel as kwargs."""
        return {
            "model": f"watsonx/{self.model_name}",
            "api_key": self._settings.watsonx_apikey,
            "api_base": self._settings.watsonx_url,
            "project_id": self._settings.watsonx_project_id,
        }
