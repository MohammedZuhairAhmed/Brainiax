import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding
from brainiax.paths import models_cache_path
from brainiax.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    """
    This class manages the loading and usage of the embedding model.

    Attributes:
        embedding_model (BaseEmbedding): The loaded embedding model instance.
    """

    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.embedding.mode
        logger.info(f"Initializing the embedding model in mode={embedding_mode}")

        try:
            from llama_index.embeddings.ollama import OllamaEmbedding  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Local dependencies not found, install with `poetry install --extras embeddings-ollama`"
            ) from e

        ollama_settings = settings.ollama
        self.embedding_model = OllamaEmbedding(
            model_name=ollama_settings.embedding_model,
            base_url=ollama_settings.api_base,
        )
