import logging

from injector import inject, singleton
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.settings import Settings as LlamaIndexSettings
from llama_index.core.utils import set_global_tokenizer
from transformers import AutoTokenizer  # type: ignore

from brainiax.components.llm.prompt_helper import get_prompt_style
from brainiax.paths import models_cache_path, models_path
from brainiax.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class LLMComponent:
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

        logger.info("Initializing the LLM in mode=%s", llm_mode)
        match settings.llm.mode:
            case "llamacpp":
                try:
                    from llama_index.llms.llama_cpp import LlamaCPP  # type: ignore
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras llms-llama-cpp`"
                    ) from e

                prompt_style = get_prompt_style(settings.llamacpp.prompt_style)
                settings_kwargs = {
                    "tfs_z": settings.llamacpp.tfs_z,  # ollama and llama-cpp
                    "top_k": settings.llamacpp.top_k,  # ollama and llama-cpp
                    "top_p": settings.llamacpp.top_p,  # ollama and llama-cpp
                    "repeat_penalty": settings.llamacpp.repeat_penalty,  # ollama llama-cpp
                    "n_gpu_layers": -1,
                    "offload_kqv": True,
                }
                self.llm = LlamaCPP(
                    model_path=str(models_path / settings.llamacpp.llm_hf_model_file),
                    temperature=settings.llm.temperature,
                    max_new_tokens=settings.llm.max_new_tokens,
                    context_window=settings.llm.context_window,
                    generate_kwargs={},
                    callback_manager=LlamaIndexSettings.callback_manager,
                    # All to GPU
                    model_kwargs=settings_kwargs,
                    # transform inputs into Llama2 format
                    messages_to_prompt=prompt_style.messages_to_prompt,
                    completion_to_prompt=prompt_style.completion_to_prompt,
                    verbose=True,
                )

            case "ollama":
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

            case "mock":
                self.llm = MockLLM()
