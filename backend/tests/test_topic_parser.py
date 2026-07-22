"""Tests for OllenTopicNodeParser: progress fractions, print-noise capture, skip accounting."""
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import ChatMessage, ChatResponse, MockLLM
from src.factories.topic_parser import OllenTopicNodeParser


class _JsonLLM(MockLLM):
    """Answers proposition requests with valid JSON and topic checks with 'same topic'."""
    def chat(self, messages, **kwargs):
        content = '["prop one", "prop two"]' if "Decompose" in messages[0].content else "same topic"
        return ChatResponse(message=ChatMessage(role="assistant", content=content))


class _ChattyLLM(MockLLM):
    """Answers conversationally (no JSON) — the failure mode seen with watsonx on junk fragments."""
    def chat(self, messages, **kwargs):
        return ChatResponse(message=ChatMessage(role="assistant", content="Please provide the content."))


def _parser(llm):
    # similarity_method='llm' matches production config; MockEmbedding satisfies the field
    return OllenTopicNodeParser.from_defaults(llm=llm, embed_model=MockEmbedding(embed_dim=8), similarity_method="llm")


def test_progress_fractions_monotonic_ending_at_1():
    parser = _parser(_JsonLLM())
    fractions = []
    parser.progress_cb = fractions.append
    nodes = parser.build_topic_based_nodes_from_documents([Document(text="Para one.\n\nPara two.\n\nPara three.")])
    assert fractions == sorted(fractions)
    assert len(fractions) == 3          # one call per paragraph (proposition cache)
    assert fractions[-1] == 1.0
    assert nodes


def test_library_print_noise_is_captured(capsys):
    parser = _parser(_ChattyLLM())
    parser.build_topic_based_nodes_from_documents([Document(text="Junk fragment.")])
    # The library prints 'No valid JSON found...' — the subclass must swallow it
    assert "No valid JSON" not in capsys.readouterr().out


def test_skipped_fragments_counted():
    parser = _parser(_ChattyLLM())
    parser.build_topic_based_nodes_from_documents([Document(text="Junk one.\n\nJunk two.")])
    assert parser._skipped_fragments == 2
