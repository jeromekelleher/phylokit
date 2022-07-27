import logging
import os

import numba

logger = logging.getLogger(__name__)

_DISABLE_NUMBA = os.environ.get("PHYLOKIT_DISABLE_NUMBA", "0")

try:
    ENABLE_NUMBA = {"0": True, "1": False}[_DISABLE_NUMBA]
except KeyError as e:  # pragma: no cover
    raise KeyError(
        "Environment variable 'PHYLOKIT_DISABLE_NUMBA' must be '0' or '1'"
    ) from e

# We will mostly be using disable numba for debugging and running tests for
# coverage, so raise a loud warning in case this is being used accidentally.

if not ENABLE_NUMBA:
    logger.warning(
        "numba globally disabled for phylokit; performance will be drastically"
        " reduced."
    )


DEFAULT_NUMBA_ARGS = {
    "nopython": True,
    "cache": True,
}


def numba_njit(func, **kwargs):
    if ENABLE_NUMBA:  # pragma: no cover
        return numba.jit(func, **{**DEFAULT_NUMBA_ARGS, **kwargs})
    else:
        return func
