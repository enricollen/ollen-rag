"""Tests for the chunking-strategy factory."""
import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser, SentenceSplitter,
    SentenceWindowNodeParser, TokenTextSplitter,
)
from src.factories.chunker import CHUNKING_STRATEGIES, create_node_parser
from src.settings import Settings

SETTINGS = Settings(_env_file=None)

def test_strategies_tuple():
    assert CHUNKING_STRATEGIES == ("sentence", "token", "semantic", "window", "llm")

def test_sentence_strategy():
    parser = create_node_parser("sentence", settings=SETTINGS)
    assert isinstance(parser, SentenceSplitter)
    assert parser.chunk_size == SETTINGS.chunk_size

def test_token_strategy():
    assert isinstance(create_node_parser("token", settings=SETTINGS), TokenTextSplitter)

def test_semantic_strategy_requires_embed_model():
    with pytest.raises(ValueError):
        create_node_parser("semantic", settings=SETTINGS)
    parser = create_node_parser("semantic", embed_model=MockEmbedding(embed_dim=8), settings=SETTINGS)
    assert isinstance(parser, SemanticSplitterNodeParser)

def test_window_strategy():
    assert isinstance(create_node_parser("window", settings=SETTINGS), SentenceWindowNodeParser)

def test_llm_strategy_forwards_models():
    # Regression: the factory must forward embed_model — TopicNodeParser.from_defaults
    # falls back to the GLOBAL llamaindex Settings.embed_model (OpenAI) otherwise and
    # raises "llama-index-embeddings-openai package not found" at construction
    from llama_index.core.llms import MockLLM
    embed = MockEmbedding(embed_dim=8)
    llm = MockLLM()
    parser = create_node_parser("llm", embed_model=embed, llm=llm, settings=SETTINGS)
    assert parser.embed_model is embed
    assert parser.llm is llm


def test_llm_strategy_requires_llm():
    with pytest.raises(ValueError):
        create_node_parser("llm", embed_model=MockEmbedding(embed_dim=8), settings=SETTINGS)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        create_node_parser("banana", settings=SETTINGS)
