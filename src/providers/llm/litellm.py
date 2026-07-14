"""LiteLLM-backed LLM connectors: one call surface for every vendor LiteLLM supports.

The base class owns everything vendor-neutral (the completion call, parameter assembly, error
mapping). Subclasses supply only a model string plus that vendor's credentials, because credential
shape -- not routing -- is what actually differs: LiteLLM's model string already routes by vendor
("ollama/llama3.1", "watsonx/meta-llama/..."). A vendor needing only api_key + api_base needs no
subclass at all and can use the generic "litellm" provider.
"""
import os
from typing import Any

# litellm's __init__ calls dotenv.load_dotenv() at import time whenever LITELLM_MODE is "DEV"
# (its default), which would splice this project's whole .env -- OLLEN_RAG_WATSONX_APIKEY included --
# into os.environ, leaking secrets to every subprocess and defeating Settings(_env_file=None) in
# tests. Settings already owns .env loading, so opt out. Must precede the litellm import.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

from litellm import completion  # noqa: E402
from src.exceptions import GenerationError
from src.factories.llm import LLMConnector, LLMConnectorFactory
from src.settings import Settings, get_settings

class LiteLLMConnector(LLMConnector):
    """Base connector: turns a prompt into a single-turn chat completion via litellm.completion().

    Subclasses set model_name / max_new_tokens / temperature in __init__ (create_llm() reads that
    metadata off the connector, keeping the llamaindex adapter settings-blind) and override
    _call_kwargs() to supply the model string and credentials.
    """
    temperature: float = 0.1

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _call_kwargs(self) -> dict[str, Any]:
        """Vendor-specific model string and credentials, passed straight to litellm.completion()."""
        raise NotImplementedError

    def _params(self) -> dict[str, Any]:
        """Generation parameters. Subclasses extend this to add vendor-specific params: LiteLLM
        forwards any non-OpenAI parameter into the provider's request body verbatim."""
        return {"max_tokens": self.max_new_tokens, "temperature": self.temperature}

    def complete(self, prompt: str) -> str:
        """Send *prompt* as a single user chat turn and return the assistant text."""
        try:
            response = completion(
                messages=[{"role": "user", "content": prompt}],
                **self._call_kwargs(),
                **self._params(),
            )
        except Exception as exc:
            # The raised type depends on which vendor SDK handled the call, so an explicit
            # except-list would be long and incomplete; the vendor message survives in the cause.
            raise GenerationError(f"LiteLLM completion failed ({self.model_name}): {exc}") from exc
        return response.choices[0].message.content

@LLMConnectorFactory.register("litellm")
class GenericLiteLLMConnector(LiteLLMConnector):
    """Any LiteLLM vendor whose credentials are just an api_key and/or an api_base.

    The model string is taken verbatim from OLLEN_RAG_LITELLM_MODEL, so no code change is needed
    to reach a new vendor.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        if not self._settings.litellm_model:
            raise ValueError("OLLEN_RAG_LITELLM_MODEL must be set when OLLEN_RAG_LLM_PROVIDER=litellm")
        self.model_name = self._settings.litellm_model
        self.max_new_tokens = self._settings.litellm_max_new_tokens
        self.temperature = self._settings.litellm_temperature

    def _call_kwargs(self) -> dict[str, Any]:
        """Send only the credentials that were actually configured; LiteLLM treats an empty
        api_key as a real (invalid) key rather than as absent."""
        kwargs: dict[str, Any] = {"model": self.model_name}
        if self._settings.litellm_api_base:
            kwargs["api_base"] = self._settings.litellm_api_base
        if self._settings.litellm_api_key:
            kwargs["api_key"] = self._settings.litellm_api_key
        return kwargs

@LLMConnectorFactory.register("litellm-watsonx")
class LiteLLMWatsonxConnector(LiteLLMConnector):
    """watsonx.ai through LiteLLM rather than the native ibm-watsonx-ai SDK.

    Deliberately reads the same OLLEN_RAG_WATSONX_* settings as the native WatsonxConnector, so
    switching between the two is a one-line .env change and the two paths stay comparable.
    Credentials go as call kwargs, never as unprefixed WATSONX_* process env vars.

    The "watsonx/" prefix routes to /ml/v1/text/chat, which is what the native connector targets
    too, and for the same reason: the chat template supplies the stop token an instruct model needs
    (see WatsonxConnector). Note that OLLEN_RAG_WATSONX_REPETITION_PENALTY is *not* forwarded here.
    LiteLLM rejects it on the chat endpoint -- it belongs to /ml/v1/text/generation, reachable only
    through the separate "watsonx_text/" prefix -- and the native connector treats it as
    belt-and-braces anyway, since the chat template is what actually ends generation.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        if not self._settings.watsonx_project_id:
            raise ValueError("OLLEN_RAG_WATSONX_PROJECT_ID must be set when OLLEN_RAG_LLM_PROVIDER=litellm-watsonx")
        self.model_name = f"watsonx/{self._settings.watsonx_llm_model_id}"
        self.max_new_tokens = self._settings.watsonx_max_new_tokens
        self.temperature = self._settings.watsonx_temperature

    def _call_kwargs(self) -> dict[str, Any]:
        """watsonx needs project_id alongside the key and endpoint; all three travel as kwargs."""
        return {
            "model": self.model_name,
            "api_key": self._settings.watsonx_apikey,
            "api_base": self._settings.watsonx_url,
            "project_id": self._settings.watsonx_project_id,
        }

@LLMConnectorFactory.register("litellm-openai")
class LiteLLMOpenAIConnector(LiteLLMConnector):
    """OpenAI or any OpenAI-compatible API (vLLM, LocalAI, …) through LiteLLM.

    Uses the dedicated OLLEN_RAG_OPENAI_* settings so OpenAI credentials stay in their own
    namespace and do not mix with the catch-all "litellm" block. The bare model name is prefixed
    with "openai/" unless the caller already included it, which is what LiteLLM needs to route
    to an OpenAI-compatible backend. Setting OLLEN_RAG_OPENAI_API_BASE overrides the official
    OpenAI endpoint, making this connector usable with any self-hosted OpenAI-compatible server.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        if not self._settings.openai_model:
            raise ValueError("OLLEN_RAG_OPENAI_MODEL must be set when OLLEN_RAG_LLM_PROVIDER=litellm-openai")
        raw = self._settings.openai_model
        self.model_name = raw if raw.startswith("openai/") else f"openai/{raw}"
        self.max_new_tokens = self._settings.openai_max_new_tokens
        self.temperature = self._settings.openai_temperature

    def _call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model_name}
        if self._settings.openai_api_key:
            kwargs["api_key"] = self._settings.openai_api_key
        if self._settings.openai_api_base:
            kwargs["api_base"] = self._settings.openai_api_base
        return kwargs

@LLMConnectorFactory.register("litellm-openrouter")
class LiteLLMOpenRouterConnector(LiteLLMConnector):
    """OpenRouter (one API key, hundreds of vendors' models) through LiteLLM.

    Uses the dedicated OLLEN_RAG_OPENROUTER_* settings so credentials stay in their own
    namespace. The model string is "<vendor>/<model>" (e.g. "google/gemini-2.5-flash"); the
    "openrouter/" prefix is added unless the caller already included it. Setting
    OLLEN_RAG_OPENROUTER_API_BASE overrides OpenRouter's default endpoint.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        if not self._settings.openrouter_model:
            raise ValueError("OLLEN_RAG_OPENROUTER_MODEL must be set when OLLEN_RAG_LLM_PROVIDER=litellm-openrouter")
        raw = self._settings.openrouter_model
        self.model_name = raw if raw.startswith("openrouter/") else f"openrouter/{raw}"
        self.max_new_tokens = self._settings.openrouter_max_new_tokens
        self.temperature = self._settings.openrouter_temperature

    def _call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model_name}
        if self._settings.openrouter_api_key:
            kwargs["api_key"] = self._settings.openrouter_api_key
        if self._settings.openrouter_api_base:
            kwargs["api_base"] = self._settings.openrouter_api_base
        return kwargs

@LLMConnectorFactory.register("litellm-ollama")
class LiteLLMOllamaConnector(LiteLLMConnector):
    """Local Ollama through LiteLLM. Needs only an api_base -- no credentials at all.

    Reuses the generic OLLEN_RAG_LITELLM_MAX_NEW_TOKENS / _TEMPERATURE rather than introducing
    Ollama-specific twins, and sends no repetition_penalty (that is a watsonx-specific param).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.model_name = f"ollama/{self._settings.ollama_model}"
        self.max_new_tokens = self._settings.litellm_max_new_tokens
        self.temperature = self._settings.litellm_temperature
        # Pull the chat model on first use if the local Ollama doesn't have it yet, so a model
        # chosen after startup (e.g. via the wizard) never fails with 'model not found'.
        from src.providers.ollama import ensure_model
        ensure_model(self._settings.ollama_api_base, self._settings.ollama_model)

    def _call_kwargs(self) -> dict[str, Any]:
        """Ollama is unauthenticated; the api_base is the only thing it needs."""
        return {"model": self.model_name, "api_base": self._settings.ollama_api_base}