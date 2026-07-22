"""Loads named YAML prompt templates and wraps them as llamaindex PromptTemplate objects."""
from pathlib import Path
import yaml
from llama_index.core import PromptTemplate
from src.exceptions import PromptNotFoundError
from src.settings import Settings, get_settings

def load_prompt(name: str, settings: Settings | None = None) -> PromptTemplate:
    """Read {prompts_dir}/{name}.yaml and return its 'template' as a PromptTemplate."""
    settings = settings or get_settings()
    path = Path(settings.prompts_dir) / f"{name}.yaml"
    if not path.is_file():
        raise PromptNotFoundError(f"Prompt template not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data.get("template"):
        raise PromptNotFoundError(f"Prompt file {path} has no 'template' key")
    return PromptTemplate(data["template"])
