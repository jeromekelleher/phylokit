import newick  # TEMPORARY - see from_newick below
import numpy as np
import tskit
import xarray

from . import core


def from_tskit(tree: tskit.Tree) -> xarray.Dataset:
    ts = tree.tree_sequence
    # NOTE: we need to add an extra element here to keep the arrays in the nodes
    # dimension the same length
    # See https://github.com/tskit-dev/tskit/issues/1322 for work on making this
    # more efficient.
    time = np.append(ts.tables.nodes.time, np.inf)
    return core.create_tree_dataset(
        parent=tree.parent_array,
        time=time,
        left_child=tree.left_child_array,
        right_sib=tree.right_sib_array,
        samples=ts.samples(),
    )


def to_tskit(ds: xarray.Dataset) -> tskit.Tree:
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


def from_newick(s: str) -> xarray.Dataset:
    # Using the newick module for now, but we will
    # want to have our own performant version of this, either here or via
    # conversion in tskit.
    trees = newick.loads(s)
    # NOTE: not bothering with error checking for now.
    tree = trees[0]
    id_map = {}
    samples = []
    parent = []
    left_child = []
    right_sib = []
    branch_length = []
    sample_name = []

    # We need to be able to work with trees that do not have meaningful node times,
    # so we store the branch lengths from the newick. We can compute node times
    # later, if they are needed.

    # NOTE: doing this in postorder for convenience, but this may not be the best
    # way to do it when writing our own parser, so we shouldn't assume any particular
    # assignment of IDs here.
    for u, newick_node in enumerate(tree.walk("postorder")):
        id_map[newick_node] = u
        parent.append(-1)
        left_child.append(-1)
        right_sib.append(-1)
        branch_length.append(newick_node.length)
        left_sib = -1
        for newick_child in newick_node.descendants:
            v = id_map[newick_child]
            parent[v] = u
            if left_sib == -1:
                left_child[u] = v
            else:
                right_sib[left_sib] = v
            left_sib = v
        if len(newick_node.descendants) == 0:
            # Assume for now that samples are leaf nodes that always have names
            samples.append(u)
            sample_name.append("" if newick_node.name is None else newick_node.name)
    parent.append(-1)
    branch_length.append(0)
    left_child.append(u)
    right_sib.append(-1)

    return core.create_tree_dataset(
        parent=np.array(parent, dtype=np.int32),
        left_child=np.array(left_child, dtype=np.int32),
        right_sib=np.array(right_sib, dtype=np.int32),
        branch_length=np.array(branch_length),
        samples=np.array(samples, dtype=np.int32),
        sample_id=np.array(sample_name),
    )


# Naive efforts to JIT this failed miserably. Probably we'll have to write
# an iterative version with a pre-allocated output array like the C version
# used in tskit.
def _to_newick(node, left_child, right_sib, branch_length, label):
    child = left_child[node]
    if child == -1:
        s = label[node]
    else:
        s = "("
        while child != -1:
            subtree = _to_newick(
                node=child,
                left_child=left_child,
                right_sib=right_sib,
                branch_length=branch_length,
                label=label,
            )
            subtree += ":" + branch_length[child]
            s += subtree + ","
            child = right_sib[child]
        s = s[:-1] + ")" + label[node]
    return s


def to_newick(ds: xarray.Dataset, *, precision=6) -> str:
    # TODO define node_branch_length as a variable, following pattern in sgkit
    if "node_branch_length" in ds:
        branch_length = ds.node_branch_length.data
    else:
        branch_length = ds.node_time[ds.node_parent] - ds.node_time

    # Convert branch_length and node labels to an array of strings to simplify
    # the conversion process. We'll probably need to do this differently if/when
    # we try to do it in a more efficient way.
    branch_length = ["{0:.{1}g}".format(x, precision) for x in branch_length.data]
    node_label = ["" for _ in branch_length]
    if "sample_id" in ds:
        for u, label in zip(ds.sample_node.data, ds.sample_id.data):
            node_label[u] = label
    else:
        for u in ds.sample_node.data:
            node_label[u] = f"n{u}"
    node_label = np.array(node_label)
    branch_length = np.array(branch_length)
    root = ds.node_left_child.data[-1]
    # TODO deal with multiroots
    s = _to_newick(
        node=root,
        left_child=ds.node_left_child.data,
        right_sib=ds.node_right_sib.data,
        branch_length=branch_length,
        label=node_label,
    )
    return s + ";"
