"""FAST-API server with uvicorn"""

import uvicorn

from brainiax.main import app
from brainiax.settings.settings import settings

uvicorn.run(app, host="0.0.0.0", port=settings().server.port, log_config=None)
