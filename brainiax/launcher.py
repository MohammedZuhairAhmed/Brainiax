"""FastAPI app creation, logger configuration and main API routes."""
import logging

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from injector import Injector
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.global_handlers import create_global_handler
from llama_index.core.settings import Settings as LlamaIndexSettings

from brainiax.server.chat.chat_router import chat_router
from brainiax.server.embeddings.embeddings_router import embeddings_router
from brainiax.server.ingest.ingest_router import ingest_router
from brainiax.settings.settings import Settings

logger = logging.getLogger(__name__)

def create_app(root_injector: Injector) -> FastAPI:

    # Start the API
    async def bind_injector_to_request(request: Request) -> None:
        request.state.injector = root_injector

    app = FastAPI(dependencies=[Depends(bind_injector_to_request)])

    app.include_router(chat_router)
    app.include_router(ingest_router)
    app.include_router(embeddings_router)

    # Add LlamaIndex simple observability
    global_handler = create_global_handler("simple")
    LlamaIndexSettings.callback_manager = CallbackManager([global_handler])

    settings = root_injector.get(Settings)
    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=settings.server.cors.allow_credentials,
            allow_origins=settings.server.cors.allow_origins,
            allow_origin_regex=settings.server.cors.allow_origin_regex,
            allow_methods=settings.server.cors.allow_methods,
            allow_headers=settings.server.cors.allow_headers,
        )

    if settings.ui.enabled:
        logger.debug("Importing the UI module")
        from brainiax.frontend.ui import BrainiaxUi

        ui = root_injector.get(BrainiaxUi)
        ui.mount_in_app(app, settings.ui.path)

    return app
