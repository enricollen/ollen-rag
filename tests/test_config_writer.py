"""Tests for the in-place .env line-merge writer."""
from pathlib import Path
from src.config.writer import merge_into_env

def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / ".env"
    p.write_text(text)
    return p

def test_replaces_value_in_place(tmp_path):
    p = _write(tmp_path, "# banner\nOLLEN_RAG_CHUNK_SIZE=512\nOLLEN_RAG_LOG_LEVEL=INFO")
    merge_into_env(p, {"OLLEN_RAG_CHUNK_SIZE": "256"})
    assert p.read_text() == "# banner\nOLLEN_RAG_CHUNK_SIZE=256\nOLLEN_RAG_LOG_LEVEL=INFO"

def test_preserves_trailing_inline_comment(tmp_path):
    p = _write(tmp_path, "OLLEN_RAG_WATSONX_APIKEY=old   # [SET]")
    merge_into_env(p, {"OLLEN_RAG_WATSONX_APIKEY": "new"})
    assert p.read_text() == "OLLEN_RAG_WATSONX_APIKEY=new   # [SET]"

def test_leaves_comments_blanks_and_unmatched_lines_untouched(tmp_path):
    p = _write(tmp_path, "# c\n\nOLLEN_RAG_LOG_LEVEL=INFO\nHF_HUB_OFFLINE=1")
    merge_into_env(p, {"OLLEN_RAG_LOG_LEVEL": "DEBUG"})
    assert p.read_text() == "# c\n\nOLLEN_RAG_LOG_LEVEL=DEBUG\nHF_HUB_OFFLINE=1"

def test_appends_missing_key_without_trailing_blank_line(tmp_path):
    p = _write(tmp_path, "OLLEN_RAG_LOG_LEVEL=INFO")
    merge_into_env(p, {"OLLEN_RAG_CHUNK_SIZE": "256"})
    assert p.read_text() == "OLLEN_RAG_LOG_LEVEL=INFO\nOLLEN_RAG_CHUNK_SIZE=256"

def test_merge_into_env_creates_file_when_missing(tmp_path):
    """A fresh install / empty Docker volume has no .env yet; the first save must create it."""
    env = tmp_path / ".env"  # deliberately does not exist
    merge_into_env(env, {"OLLEN_RAG_LLM_PROVIDER": "litellm-ollama"})
    assert env.read_text() == "OLLEN_RAG_LLM_PROVIDER=litellm-ollama"