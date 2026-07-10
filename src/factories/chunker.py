"""Factory mapping chunking-strategy names to llamaindex NodeParser instances."""
from typing import Callable
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.node_parser import (
    NodeParser, SemanticSplitterNodeParser, SentenceSplitter,
    SentenceWindowNodeParser, TokenTextSplitter,
)
from src.settings import Settings, get_settings

# Supported strategies; each maps to a dedicated OpenSearch index ({prefix}_{strategy})
CHUNKING_STRATEGIES: tuple[str, ...] = ("sentence", "token", "semantic", "window", "llm")

def create_node_parser(
    strategy: str,
    embed_model: BaseEmbedding | None = None,
    llm: LLM | None = None,
    settings: Settings | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> NodeParser:
    """Build the NodeParser for the requested strategy using configured chunk parameters."""
    settings = settings or get_settings()
    if strategy == "sentence":
        return SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if strategy == "token":
        return TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if strategy == "semantic":
        # Semantic splitting embeds sentences to find topic breakpoints, so it needs the embed model
        if embed_model is None:
            raise ValueError("Semantic chunking requires an embedding model")
        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=settings.semantic_breakpoint_percentile,
            embed_model=embed_model,
        )
    if strategy == "window":
        return SentenceWindowNodeParser.from_defaults(
            window_size=settings.sentence_window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
    if strategy == "llm":
        # OllenTopicNodeParser groups sentences by topic coherence using the LLM as judge,
        # adding progress reporting and capture of the library's print noise.
        from src.factories.topic_parser import OllenTopicNodeParser
        if llm is None:
            raise ValueError("LLM chunking requires an LLM instance")
        parser = OllenTopicNodeParser.from_defaults(
            llm=llm,
            # Not used with similarity_method="llm", but from_defaults falls back to the
            # global llamaindex Settings.embed_model (OpenAI) when omitted and crashes
            embed_model=embed_model,
            max_chunk_size=settings.llm_chunk_max_size,
            similarity_method="llm",
            window_size=settings.llm_chunk_window_size,
        )
        # from_defaults doesn't know the extra field; plain pydantic assignment works
        parser.progress_cb = progress_cb
        return parser
    raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid: {CHUNKING_STRATEGIES}")
