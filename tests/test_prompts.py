"""Tests for the YAML prompt loader."""
import pytest
from llama_index.core import PromptTemplate
from src.exceptions import PromptNotFoundError
from src.prompts import load_prompt
from src.settings import Settings

def _settings(tmp_path) -> Settings:
    # Build a Settings pointing prompts_dir at a temp folder
    return Settings(_env_file=None, prompts_dir=str(tmp_path))

def test_load_prompt_returns_template(tmp_path):
    (tmp_path / "demo.yaml").write_text(
        "name: demo\ntemplate: |\n  Context: {context_str}\n  Question: {query_str}\n",
        encoding="utf-8",
    )
    prompt = load_prompt("demo", settings=_settings(tmp_path))
    assert isinstance(prompt, PromptTemplate)
    rendered = prompt.format(context_str="CTX", query_str="Q?")
    assert "CTX" in rendered and "Q?" in rendered

def test_load_prompt_missing_raises(tmp_path):
    with pytest.raises(PromptNotFoundError):
        load_prompt("nope", settings=_settings(tmp_path))

def test_default_rag_answer_prompt_exists():
    # The shipped citation prompt must load with default settings
    prompt = load_prompt("rag_answer", settings=Settings(_env_file=None))
    assert "{context_str}" in prompt.template and "{query_str}" in prompt.template
