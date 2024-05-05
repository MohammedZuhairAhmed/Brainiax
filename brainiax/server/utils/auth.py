import logging
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from brainiax.settings.settings import settings

# 401 signify that the request requires authentication.
# 403 signify that the authenticated user is not authorized to perform the operation.
NOT_AUTHENTICATED = HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": 'Basic realm="All the API", charset="UTF-8"'},
)

logger = logging.getLogger(__name__)


def _simple_authentication(authorization: Annotated[str, Header()] = "") -> bool:
    """Check if the request is authenticated."""
    if not secrets.compare_digest(authorization, settings().server.auth.secret):
        # If the "Authorization" header is not the expected one, raise an exception.
        raise NOT_AUTHENTICATED
    return True


if not settings().server.auth.enabled:
    logger.debug(
        "Defining a dummy authentication mechanism for fastapi, always authenticating requests"
    )

    # Define a dummy authentication method that always returns True.
    def authenticated() -> bool:
        """Check if the request is authenticated."""
        return True

else:
    logger.info("Defining the given authentication mechanism for the API")

    # Method to be used as a dependency to check if the request is authenticated.
    def authenticated(
        _simple_authentication: Annotated[bool, Depends(_simple_authentication)]
    ) -> bool:
        """Check if the request is authenticated."""
        assert settings().server.auth.enabled
        if not _simple_authentication:
            raise NOT_AUTHENTICATED
        return True
