"""Tests for the provider-agnostic LLM layer: connector registry, the llama_index
adapter, create_llm() wiring, and the watsonx connector (now in src.providers)."""
import pytest
import src.providers.llm.watsonx as wx_mod
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
    s = Settings(_env_file=None, watsonx_apikey="k", watsonx_project_id="p")
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
