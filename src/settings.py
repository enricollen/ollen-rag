"""Central application configuration loaded from environment variables (.env supported)."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """All runtime configuration; every field overridable via OLLEN_RAG_* env vars."""
    # Read from OLLEN_RAG_*-prefixed env vars (optionally via .env); ignore unknown keys
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OLLEN_RAG_", extra="ignore")
    # watsonx.ai credentials and models
    watsonx_url: str = "https://eu-de.ml.cloud.ibm.com"
    watsonx_apikey: str = ""
    watsonx_project_id: str = ""
    watsonx_llm_model_id: str = "meta-llama/llama-3-3-70b-instruct"
    watsonx_embedding_model_id: str = "ibm/slate-125m-english-rtrvr"
    watsonx_max_new_tokens: int = 800
    watsonx_temperature: float = 0.1
    # Belt-and-braces against repeated tokens on the chat endpoint too; the chat template's
    # own stop token is what actually ends generation (see WatsonxChatLLM).
    watsonx_repetition_penalty: float = 1.15
    # provider selection
    embedding_provider: str = "watsonx"  # watsonx | fastembed
    llm_provider: str = "watsonx"
    fastembed_model_name: str = "BAAI/bge-small-en-v1.5"
    # Local cache for fastembed ONNX model files (coherent with reranker_model under models/);
    # persists downloads so the model is fetched from the hub once, then reused offline.
    fastembed_cache_dir: str = "models/fastembed"
    # OpenSearch connection and hybrid search
    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_verify_certs: bool = True
    opensearch_index_prefix: str = "ollen_rag"
    opensearch_hybrid_pipeline: str = "ollen-rag-hybrid"
    # vector store backend selection (opensearch | chroma | qdrant …)
    vector_store: str = "opensearch"
    hybrid_sparse_weight: float = 0.3
    hybrid_dense_weight: float = 0.7
    # Chroma backend (embedded PersistentClient): on-disk store location
    chroma_path: str = "./chroma_db"
    # chunking
    default_chunking_strategy: str = "sentence"  # sentence | token | semantic | window
    chunk_size: int = 512
    chunk_overlap: int = 64
    semantic_breakpoint_percentile: int = 95
    sentence_window_size: int = 3
    # LLM-based (topic) chunking
    llm_chunk_max_size: int = 1000   # max tokens per chunk produced by TopicNodeParser
    llm_chunk_window_size: int = 2   # sentence window the LLM sees to judge topic boundaries
    # retrieval / rerank
    retrieval_top_k: int = 10
    # Score floor on fused hybrid scores (min_max-normalized 0-1); 0.0 disables the filter
    similarity_threshold: float = 0.0
    rerank_top_n: int = 4
    reranker_model: str = "models/reranker"
    # generation
    citation_chunk_size: int = 512
    prompts_dir: str = "config/prompts"
    default_prompt_name: str = "rag_answer"
    # retrieval eval harness
    eval_dir: str = "config/eval"
    # ingest-time enrichment
    # Default for LLM keyword extraction per chunk; overridable per request (form/MCP param)
    enrich_keywords: bool = False
    # logging
    # DEBUG | INFO | WARNING | ERROR | CRITICAL (case-insensitive); invalid -> INFO
    log_level: str = "DEBUG"

@lru_cache
def get_settings() -> Settings:
    """Singleton accessor so the whole app shares one Settings instance."""
    return Settings()
