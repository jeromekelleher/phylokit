import numpy as np

from . import core
from . import jit
from . import util


@jit.numba_njit()
def _permute_node_seq(nodes, ordering, reversed_map):
    ret = np.zeros_like(nodes, dtype=np.int32)
    for u, v in enumerate(ordering):
        old_node = nodes[v]
        if old_node != -1:
            ret[u] = reversed_map[old_node]
        else:
            ret[u] = -1
    return ret


def permute_tree(ds, ordering):
    """
    Returns a new dataset in which the tree nodes have been permuted according
    to the specified ordering such that node u in the new dataset will be
    equivalent to ``ordering[u]``.
    :param xarray.DataSet ds: The tree dataset to permute.
    :param list ordering: The permutation to apply to the nodes.
    :return: A new dataset with the permuted nodes.
    :rtype: xarray.DataSet
    """
    num_nodes = ds.node_left_child.shape[0]
    if len(ordering) != num_nodes:
        raise ValueError(
            "The length of the ordering must be equal to the number of nodes"
        )

    for node in ordering:
        util.check_node_bounds(ds, node)

    reversed_map = np.zeros(num_nodes, dtype=np.int32)
    for u, v in enumerate(ordering):
        reversed_map[v] = u

    return core.create_tree_dataset(
        parent=_permute_node_seq(ds.node_parent.data, ordering, reversed_map),
        left_child=_permute_node_seq(ds.node_left_child.data, ordering, reversed_map),
        right_sib=_permute_node_seq(ds.node_right_sib.data, ordering, reversed_map),
        samples=np.array([reversed_map[s] for s in ds.sample_node.data]),
    )
