"""Provider-agnostic generation LLM: each provider (in src/providers/) implements
LLMConnector.complete() and self-registers with LLMConnectorFactory; ConnectorLLM
adapts whichever connector is configured to the llama_index LLM interface that
CitationQueryEngine needs.
"""
from abc import ABC, abstractmethod
from typing import Any
from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from src.settings import Settings, get_settings

class LLMConnector(ABC):
    """Provider-agnostic LLM interface: implement complete() and register with LLMConnectorFactory.

    Concrete connectors override model_name/max_new_tokens (class attributes or in __init__)
    so create_llm() can build accurate llamaindex metadata without provider-specific settings.
    """
    model_name: str = "connector-llm"
    max_new_tokens: int = 512

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send *prompt* to the LLM and return the response as plain text."""

class LLMConnectorFactory:
    """Registry mapping a provider name to its LLMConnector class.

        @LLMConnectorFactory.register("myprovider")
        class MyConnector(LLMConnector): ...

        connector = LLMConnectorFactory.create("myprovider", settings=settings)
    """
    _registry: dict[str, type[LLMConnector]] = {}

    @classmethod
    def register(cls, provider: str):
        """Class decorator registering *provider*'s connector class."""
        def decorator(connector_cls: type[LLMConnector]) -> type[LLMConnector]:
            cls._registry[provider] = connector_cls
            return connector_cls
        return decorator

    @classmethod
    def create(cls, provider: str, **kwargs: Any) -> LLMConnector:
        """Instantiate the connector for *provider*, or raise listing known providers."""
        if not provider:
            raise ValueError("No LLM provider configured. Finish setup at /ui/, or set OLLEN_RAG_LLM_PROVIDER.")
        if provider not in cls._registry:
            raise ValueError(f"Unknown LLM provider '{provider}'. Available providers: {sorted(cls._registry)}")
        return cls._registry[provider](**kwargs)

def _strip_role_labels(prompt: str) -> str:
    """Undo llama_index's default messages_to_prompt(), which wraps our whole templated
    text as 'user: <text>\\nassistant: ' before it reaches here — those literal role
    labels aren't part of the actual answer and shouldn't leak into the chat message."""
    prompt = prompt.strip()
    if prompt.startswith("user: "):
        prompt = prompt[len("user: "):]
    for suffix in ("\nassistant: ", "\nassistant:"):
        if prompt.endswith(suffix):
            prompt = prompt[: -len(suffix)]
            break
    return prompt

class ConnectorLLM(CustomLLM):
    """Adapts any LLMConnector to llama_index's LLM interface, so CitationQueryEngine
    stays provider-blind: only this class knows about llama_index's calling convention."""
    connector: LLMConnector
    max_new_tokens: int
    model_name: str = "connector-llm"

    @property
    def metadata(self) -> LLMMetadata:
        """llamaindex metadata derived from whatever connector is plugged in."""
        return LLMMetadata(num_output=self.max_new_tokens, model_name=self.model_name, is_chat_model=True)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Forward the (unwrapped) prompt to the connector and wrap its text answer."""
        text = self.connector.complete(_strip_role_labels(prompt))
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Fake streaming for llamaindex compatibility: yield the whole completion once."""
        response = self.complete(prompt, **kwargs)
        def gen() -> CompletionResponseGen:
            yield response
        return gen()

def create_llm(settings: Settings | None = None) -> LLM:
    """Return the configured provider's connector, adapted to the llama_index LLM interface."""
    settings = settings or get_settings()
    # Local import: providers import this module for the registry, so a module-level
    # import here would be circular. This triggers provider self-registration once.
    import src.providers  # noqa: F401
    connector = LLMConnectorFactory.create(settings.llm_provider, settings=settings)
    return ConnectorLLM(connector=connector, max_new_tokens=connector.max_new_tokens, model_name=connector.model_name)
