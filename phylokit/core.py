import xarray

from .traversal import _postorder
from .traversal import _preorder
from .util import _get_node_branch_length

DIM_NODE = "nodes"
DIM_TRAVERSAL = "traversal"
# Following sgkit example, specifically so that we can join on the samples dimension
DIM_SAMPLE = "samples"


# TODO add some defaults
def create_tree_dataset(
    *,
    parent,
    left_child,
    right_sib,
    samples,
    time=None,
    branch_length=None,
    sample_id=None,
    preorder=None,
    postorder=None,
):
    data_vars = {
        "node_parent": ([DIM_NODE], parent),
        "node_left_child": ([DIM_NODE], left_child),
        "node_right_sib": ([DIM_NODE], right_sib),
        "sample_node": ([DIM_SAMPLE], samples),
    }
    if preorder is not None:
        data_vars["traversal_preorder"] = ([DIM_TRAVERSAL], preorder)
    else:
        data_vars["traversal_preorder"] = (
            [DIM_TRAVERSAL],
            _preorder(parent, left_child, right_sib, -1),
        )
    if postorder is not None:
        data_vars["traversal_postorder"] = ([DIM_TRAVERSAL], postorder)
    else:
        data_vars["traversal_postorder"] = (
            [DIM_TRAVERSAL],
            _postorder(left_child, right_sib, -1),
        )
    if time is not None:
        data_vars["node_time"] = ([DIM_NODE], time)
    if branch_length is not None:
        data_vars["node_branch_length"] = ([DIM_NODE], branch_length)
    else:
        if time is not None:
            data_vars["node_branch_length"] = (
                [DIM_NODE],
                _get_node_branch_length(parent, time),
            )
    # TODO should sample_id be a dimension instead so that we support
    # direct indexing on it?
    if sample_id is not None:
        data_vars["sample_id"] = ([DIM_SAMPLE], sample_id)

    return xarray.Dataset(data_vars)
