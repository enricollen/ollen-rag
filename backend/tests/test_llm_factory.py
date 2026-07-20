"""Tests for the provider-agnostic LLM layer: connector registry, the llama_index
adapter, create_llm() wiring, and the watsonx connector (now in src.providers)."""
from types import SimpleNamespace
import pytest
import src.providers.llm.litellm as lite_mod
import src.providers.llm.watsonx as wx_mod
from src.exceptions import GenerationError
from src.factories import llm as llm_mod
from src.settings import Settings

class _FakeConnector(llm_mod.LLMConnector):
    """Records the prompt it was called with; stands in for any registered provider."""
    def __init__(self):
        self.last_prompt = None
    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "risposta"

def test_factory_registers_watsonx():
    assert "watsonx" in llm_mod.LLMConnectorFactory._registry

def test_factory_create_unknown_provider_raises():
    with pytest.raises(ValueError):
        llm_mod.LLMConnectorFactory.create("banana")

def test_factory_create_dispatches_kwargs():
    captured = {}
    @llm_mod.LLMConnectorFactory.register("fake")
    class _Registered(llm_mod.LLMConnector):
        def __init__(self, marker=None):
            captured["marker"] = marker
        def complete(self, prompt: str) -> str:
            return prompt
    llm_mod.LLMConnectorFactory.create("fake", marker="x")
    assert captured["marker"] == "x"

def test_watsonx_connector_calls_chat_endpoint(monkeypatch):
    captured = {}
    class _StubModel:
        def chat(self, messages, params):
            captured["messages"] = messages
            captured["params"] = params
            return {"choices": [{"message": {"content": "ciao"}}]}
    monkeypatch.setattr(wx_mod, "ModelInference", lambda **kwargs: _StubModel())
    monkeypatch.setattr(wx_mod, "Credentials", lambda **kwargs: kwargs)

    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    connector = wx_mod.WatsonxConnector(settings=s)
    result = connector.complete("Domanda?")

    assert result == "ciao"
    assert captured["messages"] == [{"role": "user", "content": "Domanda?"}]
    assert captured["params"]["max_tokens"] == s.watsonx_max_new_tokens

def test_watsonx_connector_exposes_model_metadata():
    """The connector, not settings, is what create_llm() reads model metadata from."""
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    connector = wx_mod.WatsonxConnector(settings=s)
    assert connector.model_name == s.watsonx_llm_model_id
    assert connector.max_new_tokens == s.watsonx_max_new_tokens

def test_strip_role_labels_removes_wrapper():
    wrapped = "user: Sei un assistente.\nRisposta: ciao\nassistant: "
    assert llm_mod._strip_role_labels(wrapped) == "Sei un assistente.\nRisposta: ciao"

def test_strip_role_labels_leaves_plain_prompt_untouched():
    assert llm_mod._strip_role_labels("Sei un assistente. Rispondi.") == "Sei un assistente. Rispondi."

def test_connector_llm_delegates_to_any_connector():
    connector = _FakeConnector()
    llm = llm_mod.ConnectorLLM(connector=connector, max_new_tokens=100, model_name="fake-model")
    response = llm.complete("user: Domanda?\nassistant: ")
    assert response.text == "risposta"
    assert connector.last_prompt == "Domanda?"
    assert llm.metadata.is_chat_model is True

def test_create_llm_wires_watsonx_connector():
    s = Settings(_env_file=None, llm_provider="watsonx", watsonx_apikey="k", watsonx_project_id="p")
    model = llm_mod.create_llm(s)
    assert isinstance(model, llm_mod.ConnectorLLM)
    assert isinstance(model.connector, wx_mod.WatsonxConnector)
    assert model.model_name == s.watsonx_llm_model_id
    assert model.max_new_tokens == s.watsonx_max_new_tokens
    assert model.metadata.is_chat_model is True

def test_create_llm_uses_connector_metadata():
    """Extensibility: create_llm() takes model metadata from the connector, not settings."""
    @llm_mod.LLMConnectorFactory.register("fake-meta")
    class _MetaConnector(llm_mod.LLMConnector):
        model_name = "fake-model"
        max_new_tokens = 42
        def __init__(self, settings=None):
            pass
        def complete(self, prompt: str) -> str:
            return "ok"
    model = llm_mod.create_llm(Settings(_env_file=None, llm_provider="fake-meta"))
    assert model.model_name == "fake-model"
    assert model.max_new_tokens == 42

def test_unknown_llm_provider_raises():
    with pytest.raises(ValueError):
        llm_mod.create_llm(Settings(_env_file=None, llm_provider="banana"))

def test_litellm_import_does_not_load_dotenv():
    """litellm calls dotenv.load_dotenv() at import when LITELLM_MODE=DEV (its default), which would
    splice this project's .env into os.environ -- leaking OLLEN_RAG_WATSONX_APIKEY to subprocesses
    and breaking Settings(_env_file=None) isolation. Importing the connector module must opt out."""
    import os
    assert os.environ["LITELLM_MODE"] == "PRODUCTION"
    # .env sets this one and conftest does not, so its presence would prove load_dotenv() ran.
    # (OLLEN_RAG_WATSONX_APIKEY is unusable as a probe: conftest's autouse fixture sets it.)
    assert "OLLEN_RAG_HYBRID_DENSE_WEIGHT" not in os.environ

def _stub_completion(captured: dict, text: str = "risposta"):
    """Build a litellm.completion stub that records its kwargs and returns a ModelResponse-shaped object."""
    def _completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    return _completion

def test_factory_registers_litellm_providers():
    for key in ("litellm", "litellm-watsonx", "litellm-ollama"):
        assert key in llm_mod.LLMConnectorFactory._registry

def test_generic_litellm_passes_model_string_through(monkeypatch):
    """The generic connector forwards OLLEN_RAG_LITELLM_MODEL to litellm untouched."""
    captured = {}
    monkeypatch.setattr(lite_mod, "completion", _stub_completion(captured))
    s = Settings(_env_file=None, litellm_model="openai/gpt-4o", litellm_api_key="sk-x")
    connector = lite_mod.GenericLiteLLMConnector(settings=s)

    assert connector.complete("Domanda?") == "risposta"
    assert captured["model"] == "openai/gpt-4o"
    assert captured["api_key"] == "sk-x"
    assert captured["messages"] == [{"role": "user", "content": "Domanda?"}]
    assert captured["max_tokens"] == 800
    assert captured["temperature"] == 0.1
    # No api_base was configured, so it must not be sent at all
    assert "api_base" not in captured

def test_generic_litellm_omits_empty_credentials(monkeypatch):
    """Empty api_key/api_base must be dropped, not sent as empty strings."""
    captured = {}
    monkeypatch.setattr(lite_mod, "completion", _stub_completion(captured))
    s = Settings(_env_file=None, litellm_model="ollama/llama3.1")
    lite_mod.GenericLiteLLMConnector(settings=s).complete("ciao")
    assert "api_key" not in captured
    assert "api_base" not in captured

def test_generic_litellm_requires_model():
    """An unconfigured generic provider fails at construction (400), not at query time (502)."""
    with pytest.raises(ValueError, match="OLLEN_RAG_LITELLM_MODEL"):
        lite_mod.GenericLiteLLMConnector(settings=Settings(_env_file=None, litellm_model=""))

def test_litellm_completion_failure_becomes_generation_error(monkeypatch):
    """Any vendor SDK failure surfaces as GenerationError -> HTTP 502."""
    def _boom(**kwargs):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(lite_mod, "completion", _boom)
    s = Settings(_env_file=None, litellm_model="ollama/llama3.1")
    connector = lite_mod.GenericLiteLLMConnector(settings=s)
    with pytest.raises(GenerationError, match="ollama/llama3.1"):
        connector.complete("Domanda?")

def test_create_llm_wires_generic_litellm_connector():
    s = Settings(_env_file=None, llm_provider="litellm", litellm_model="openai/gpt-4o")
    model = llm_mod.create_llm(s)
    assert isinstance(model.connector, lite_mod.GenericLiteLLMConnector)
    assert model.model_name == "openai/gpt-4o"
    assert model.max_new_tokens == 800

def test_litellm_watsonx_maps_existing_watsonx_settings(monkeypatch):
    """litellm-watsonx reuses OLLEN_RAG_WATSONX_*, so switching providers is a one-line .env change."""
    captured = {}
    monkeypatch.setattr(lite_mod, "completion", _stub_completion(captured, text="ciao"))
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    connector = lite_mod.LiteLLMWatsonxConnector(settings=s)

    assert connector.complete("Domanda?") == "ciao"
    assert captured["model"] == f"watsonx/{s.watsonx_llm_model_id}"
    assert captured["api_key"] == "k"
    assert captured["api_base"] == s.watsonx_url
    assert captured["project_id"] == "p"
    assert captured["max_tokens"] == s.watsonx_max_new_tokens
    assert captured["temperature"] == s.watsonx_temperature

def test_litellm_watsonx_omits_repetition_penalty(monkeypatch):
    """repetition_penalty must NOT be forwarded. The "watsonx/" prefix routes to /ml/v1/text/chat,
    and LiteLLM hard-errors there rather than dropping the param:

        litellm.APIConnectionError: LiteLLM now defaults to Watsonx's `/text/chat` endpoint.
        Please use the `watsonx_text` provider instead, to call the `/text/generation` endpoint.
        Param: repetition_penalty

    The native WatsonxConnector does send it, because IBM's SDK tolerates it; LiteLLM is stricter.
    Nothing is lost: the chat template's stop token is what ends generation.
    """
    captured = {}
    monkeypatch.setattr(lite_mod, "completion", _stub_completion(captured))
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    lite_mod.LiteLLMWatsonxConnector(settings=s).complete("Domanda?")
    assert "repetition_penalty" not in captured

def test_litellm_watsonx_requires_project_id():
    """Missing project_id is a config error (400), not a runtime generation failure (502)."""
    with pytest.raises(ValueError, match="OLLEN_RAG_WATSONX_PROJECT_ID"):
        lite_mod.LiteLLMWatsonxConnector(settings=Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id=""))

def test_litellm_watsonx_exposes_prefixed_model_metadata():
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
    connector = lite_mod.LiteLLMWatsonxConnector(settings=s)
    assert connector.model_name == f"watsonx/{s.watsonx_llm_model_id}"
    assert connector.max_new_tokens == s.watsonx_max_new_tokens

def test_create_llm_wires_litellm_watsonx_connector():
    s = Settings(_env_file=None, llm_provider="litellm-watsonx", watsonx_apikey="k", watsonx_project_id="p")
    model = llm_mod.create_llm(s)
    assert isinstance(model.connector, lite_mod.LiteLLMWatsonxConnector)
    assert model.metadata.is_chat_model is True

def test_litellm_ollama_prefixes_model_and_sends_api_base(monkeypatch):
    captured = {}
    monkeypatch.setattr(lite_mod, "completion", _stub_completion(captured))
    s = Settings(_env_file=None, ollama_model="mistral", ollama_api_base="http://ollama:11434")
    connector = lite_mod.LiteLLMOllamaConnector(settings=s)

    assert connector.complete("Domanda?") == "risposta"
    assert captured["model"] == "ollama/mistral"
    assert captured["api_base"] == "http://ollama:11434"

def test_litellm_ollama_omits_repetition_penalty(monkeypatch):
    """repetition_penalty is a watsonx concern; sending it to Ollama would land in its request body."""
    captured = {}
    monkeypatch.setattr(lite_mod, "completion", _stub_completion(captured))
    lite_mod.LiteLLMOllamaConnector(settings=Settings(_env_file=None)).complete("Domanda?")
    assert "repetition_penalty" not in captured
    assert "project_id" not in captured

def test_litellm_ollama_exposes_prefixed_model_metadata():
    s = Settings(_env_file=None, ollama_model="llama3.1")
    connector = lite_mod.LiteLLMOllamaConnector(settings=s)
    assert connector.model_name == "ollama/llama3.1"
    assert connector.max_new_tokens == s.litellm_max_new_tokens

def test_create_llm_wires_litellm_ollama_connector():
    model = llm_mod.create_llm(Settings(_env_file=None, llm_provider="litellm-ollama"))
    assert isinstance(model.connector, lite_mod.LiteLLMOllamaConnector)
    assert model.model_name == "ollama/llama3.1"
