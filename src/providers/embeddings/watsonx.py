"""watsonx.ai embedding builder, self-registered with the embedding factory on import."""
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ibm import WatsonxEmbeddings
from src.factories.embeddings import EmbeddingFactory
from src.settings import Settings

@EmbeddingFactory.register("watsonx")
def create_watsonx_embedding(settings: Settings) -> BaseEmbedding:
    """Registered builder: watsonx.ai embeddings via the llamaindex IBM connector."""
    return WatsonxEmbeddings(
        model_id=settings.watsonx_embedding_model_id,
        url=settings.watsonx_url,
        apikey=settings.watsonx_apikey,
        project_id=settings.watsonx_project_id,
        truncate_input_tokens=512,
    )