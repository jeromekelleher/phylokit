import numba

DEFAULT_NUMBA_ARGS = {
    "nopython": True,
    "cache": True,
}


def numba_njit(func, **kwargs):
    return numba.jit(func, **{**DEFAULT_NUMBA_ARGS, **kwargs})
