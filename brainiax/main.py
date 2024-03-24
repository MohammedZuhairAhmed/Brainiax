"""FastAPI app creation, logger configuration and main API routes."""

from brainiax.di import global_injector
from brainiax.launcher import create_app

app = create_app(global_injector)
