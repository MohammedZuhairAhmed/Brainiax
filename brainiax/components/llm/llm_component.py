import logging

from injector import inject, singleton
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.settings import Settings as LlamaIndexSettings
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer  # type: ignore

from brainiax.paths import models_cache_path, models_path
from brainiax.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
    """
    This class manages the loading and interaction with the large language model (LLM).

    Attributes:
        llm (LLM): The loaded LLM instance.
    """

    llm: LLM

    @inject
    def __init__(self, settings: Settings) -> None:
        llm_mode = settings.llm.mode
        if settings.llm.tokenizer:
            set_global_tokenizer(
                AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=settings.llm.tokenizer,
                    cache_dir=str(models_cache_path),
                )
            )

        logger.info(f"Initializing the LLM in mode={llm_mode}")

        try:
            from llama_index.llms.ollama import Ollama  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Ollama dependencies not found, install with `poetry install --extras llms-ollama`"
            ) from e

        ollama_settings = settings.ollama
        settings_kwargs = {
            "tfs_z": ollama_settings.tfs_z,  # ollama and llama-cpp
            "num_predict": ollama_settings.num_predict,  # ollama only
            "top_k": ollama_settings.top_k,  # ollama and llama-cpp
            "top_p": ollama_settings.top_p,  # ollama and llama-cpp
            "repeat_last_n": ollama_settings.repeat_last_n,  # ollama
            "repeat_penalty": ollama_settings.repeat_penalty,  # ollama llama-cpp
        }

        self.llm = Ollama(
            model=ollama_settings.llm_model,
            base_url=ollama_settings.api_base,
            temperature=settings.llm.temperature,
            context_window=settings.llm.context_window,
            additional_kwargs=settings_kwargs,
            request_timeout=ollama_settings.request_timeout,
        )
