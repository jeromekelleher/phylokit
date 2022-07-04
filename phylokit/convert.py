import numpy as np
import tskit

from . import core


def from_tskit(tree):
    ts = tree.tree_sequence
    # NOTE: we need to add an extra element here to keep the arrays in the nodes
    # dimension the same length
    time = np.append(ts.tables.nodes.time, np.inf)
    return core.create_tree_dataset(
        parent=tree.parent_array,
        time=time,
        left_child=tree.left_child_array,
        right_sib=tree.right_sib_array,
        samples=ts.samples(),
    )


def to_tskit(ds):
    tables = tskit.TableCollection(1)
    N = ds.sizes[core.DIM_NODE] - 1
    flags = np.zeros(N, dtype=np.uint32)
    flags[ds.sample_node.to_numpy()] = tskit.NODE_IS_SAMPLE
    tables.nodes.set_columns(flags=flags, time=ds.node_time[:-1])
    # TODO do this more efficiently.
    for u, parent in enumerate(ds.node_parent):
        if parent != -1:
            tables.edges.add_row(0, 1, int(parent), u)
    tables.sort()
    return tables.tree_sequence().first()
