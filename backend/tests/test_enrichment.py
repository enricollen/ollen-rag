"""Tests for the LLM keyword enrichment transform (LLM mocked)."""
import pytest
from llama_index.core import PromptTemplate
from llama_index.core.schema import TextNode
from src.rag.enrichment import KeywordEnricher, parse_keywords


class _FakeLLM:
    """Stands in for a llamaindex LLM: records predict calls, returns a canned string."""
    def __init__(self, response="triage, codici colore, priorita"):
        self.response = response
        self.calls = []
    def predict(self, prompt, **kwargs):
        self.calls.append(kwargs)
        return self.response


PROMPT = PromptTemplate("Keywords for: {chunk_text}")


def test_parse_keywords_cleans_messy_llm_output():
    raw = '1. "triage"\n2) codici colore\n- priorita ,  , priorita\n* pronto soccorso'
    # numbering/bullets/quotes stripped, split on newlines+commas, empties and dupes dropped
    assert parse_keywords(raw) == "triage, codici colore, priorita, pronto soccorso"


def test_parse_keywords_caps_at_15():
    raw = ", ".join(f"kw{i}" for i in range(30))
    assert parse_keywords(raw) == ", ".join(f"kw{i}" for i in range(15))


def test_parse_keywords_empty_input():
    assert parse_keywords("   \n , , \n") == ""


def test_enricher_sets_keywords_metadata_and_llm_exclusion():
    llm = _FakeLLM()
    nodes = [TextNode(text="Il triage assegna codici colore."), TextNode(text="Altro chunk.")]
    out = KeywordEnricher(llm=llm, prompt=PROMPT)(nodes)
    assert len(llm.calls) == 2
    assert llm.calls[0] == {"chunk_text": "Il triage assegna codici colore."}
    for node in out:
        assert node.metadata["keywords"] == "triage, codici colore, priorita"
        # embedded yes (that's the point), but never injected into generation LLM context
        assert "keywords" in node.excluded_llm_metadata_keys
        assert "keywords" not in node.excluded_embed_metadata_keys


def test_enricher_skips_metadata_when_no_keywords_parsed():
    llm = _FakeLLM(response="   ")
    nodes = [TextNode(text="chunk")]
    out = KeywordEnricher(llm=llm, prompt=PROMPT)(nodes)
    assert "keywords" not in out[0].metadata


def test_enricher_llm_error_propagates():
    class _BoomLLM:
        def predict(self, prompt, **kwargs):
            raise RuntimeError("llm down")
    with pytest.raises(RuntimeError, match="llm down"):
        KeywordEnricher(llm=_BoomLLM(), prompt=PROMPT)([TextNode(text="chunk")])


def test_enricher_reports_progress_fractions():
    llm = _FakeLLM()
    fractions = []
    nodes = [TextNode(text="uno"), TextNode(text="due"), TextNode(text="tre"), TextNode(text="quattro")]
    KeywordEnricher(llm=llm, prompt=PROMPT, progress_cb=fractions.append)(nodes)
    assert fractions == [0.25, 0.5, 0.75, 1.0]
