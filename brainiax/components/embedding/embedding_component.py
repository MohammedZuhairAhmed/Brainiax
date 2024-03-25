import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding

from brainiax.paths import models_cache_path
from brainiax.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.embedding.mode
        logger.info("Initializing the embedding model in mode=%s", embedding_mode)
        match embedding_mode:
            case "mock":
                # Not a random number, is the dimensionality used by
                # the default embedding model
                self.embedding_model = MockEmbedding(384)
            
            case "ollama":
                try:
                    from llama_index.embeddings.ollama import (  # type: ignore
                        OllamaEmbedding,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras embeddings-ollama`"
                    ) from e

                ollama_settings = settings.ollama
                self.embedding_model = OllamaEmbedding(
                    model_name=ollama_settings.embedding_model,
                    base_url=ollama_settings.api_base,
                )
