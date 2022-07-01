import numba
import xarray
import tskit
import numpy as np

DIM_NODE = "nodes"
# Following sgkit example, specifically so that we can join on the samples dimension
DIM_SAMPLE = "samples"


DEFAULT_NUMBA_ARGS = {
    "nopython": True,
    "cache": True,
}


def numba_njit(func, **kwargs):
    return numba.jit(func, **{**DEFAULT_NUMBA_ARGS, **kwargs})


def create_tree_dataset(parent, time, left_child, right_sib, samples):
    data_vars = {
        "node_parent": ([DIM_NODE], parent),
        "node_time": ([DIM_NODE], time),
        "node_left_child": ([DIM_NODE], left_child),
        "node_right_sib": ([DIM_NODE], right_sib),
        "sample_node": ([DIM_SAMPLE], samples),
    }
    return xarray.Dataset(data_vars)


def tskit_to_dataset(tree):
    ts = tree.tree_sequence
    # NOTE: we need to add an extra element here to keep the arrays in the nodes
    # dimension the same length
    time = np.append(ts.tables.nodes.time, np.inf)
    return create_tree_dataset(
        parent=tree.parent_array,
        time=time,
        left_child=tree.left_child_array,
        right_sib=tree.right_sib_array,
        samples=ts.samples(),
    )


def dataset_to_tskit(ds):
    tables = tskit.TableCollection(1)
    N = ds.sizes["nodes"] - 1
    flags = np.zeros(N, dtype=np.uint32)
    flags[ds.sample_node.to_numpy()] = tskit.NODE_IS_SAMPLE
    tables.nodes.set_columns(flags=flags, time=ds.node_time[:-1])
    # TODO do this more efficiently.
    for u, parent in enumerate(ds.node_parent):
        if parent != -1:
            tables.edges.add_row(0, 1, parent, u)
    tables.sort()
    return tables.tree_sequence().first()
