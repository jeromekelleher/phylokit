import logging
import os

import numba
import xarray

logger = logging.getLogger(__name__)

DIM_NODE = "nodes"
# Following sgkit example, specifically so that we can join on the samples dimension
DIM_SAMPLE = "samples"

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
        "numba globally disabled for phylokit; performance will be drastically reduced."
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


# TODO add some defaults
def create_tree_dataset(
    *,
    parent,
    left_child,
    right_sib,
    samples,
    time=None,
    branch_length=None,
    sample_id=None
):
    data_vars = {
        "node_parent": ([DIM_NODE], parent),
        "node_left_child": ([DIM_NODE], left_child),
        "node_right_sib": ([DIM_NODE], right_sib),
        "sample_node": ([DIM_SAMPLE], samples),
    }
    if time is not None:
        data_vars["node_time"] = ([DIM_NODE], time)
    if branch_length is not None:
        data_vars["node_branch_length"] = ([DIM_NODE], branch_length)
    # TODO should sample_id be a dimension instead so that we support
    # direct indexing on it?
    if sample_id is not None:
        data_vars["sample_id"] = ([DIM_SAMPLE], sample_id)

    return xarray.Dataset(data_vars)
