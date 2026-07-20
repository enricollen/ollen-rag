"""Embedding providers; importing this package registers them with the embedding factory."""
from src.providers.embeddings import fastembed, litellm, watsonx  # noqa: F401
