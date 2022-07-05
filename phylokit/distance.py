# Tree distance metrics.
import numba
import numpy as np


def check_node_bounds(tree, *args):
    """
    Checks that the specified node is within the tree.

    :param tskit.Tree tree: The tree to check.
    :param int `*args`: The node IDs to check.
    :raises ValueError: If any of the nodes are outside the tree.
    """
    num_nodes = tree.parent_array.shape[0] - 1
    for u in args:
        if u < 0 or u > num_nodes:
            raise ValueError(f"Node {u} is not in the tree")


@numba.njit(cache=True)
def _branch_length(parent, time, u):
    ret = 0
    p = parent[u]
    if p != -1:
        ret = time[p] - time[u]
    return ret


def branch_length(tree, u):
    """
    Returns the length of the branch (in units of time) joining the specified
    node to its parent.

    :param int u: The node ID.
    :return : The length of the branch.
    :rtype : float
    """
    check_node_bounds(tree, u)
    return _branch_length(tree.parent_array, tree.tree_sequence.tables.nodes.time, u)


@numba.njit(cache=True)
def _mrca(parent, time, u, v):
    tu = time[u]
    tv = time[v]
    while u != v:
        if tu < tv:
            u = parent[u]
            if u == -1:
                return -1
            tu = time[u]
        else:
            v = parent[v]
            if v == -1:
                return -1
            tv = time[v]
    return u


def mrca(tree, u, v):
    """
    Returns the most recent common ancestor of the specified nodes.

    :param int `*args`: input node IDs, must be at least 2.
    :return: The most recent common ancestor of input nodes.
    :rtype: int
    """
    virtual_root = tree.virtual_root
    # Check if u or v is outside the tree
    check_node_bounds(tree, u, v)
    # Check if u and v are virtual roots
    if u == virtual_root or v == virtual_root:
        return virtual_root
    return _mrca(tree.parent_array, tree.tree_sequence.tables.nodes.time, u, v)


@numba.njit(cache=True)
def _kc_distance(samples, t1, t2):
    # t1 and t2 are tuples of the form (parent_array, time_array, root)
    n = samples.shape[0]
    N = (n * (n - 1)) // 2
    m = [np.zeros(N + n), np.zeros(N + n)]
    M = [np.zeros(N + n), np.zeros(N + n)]
    for tree_index, tree in enumerate([t1, t2]):
        for sample in range(n):
            m[tree_index][N + sample] = 1
            M[tree_index][N + sample] = _branch_length(tree[0], tree[1], sample)

        for n1 in range(n):
            for n2 in range(n1 + 1, n):
                mrca_id = _mrca(tree[0], tree[1], samples[n1], samples[n2])
                depth = 0
                p = tree[0][mrca_id]
                while p != -1:
                    depth += 1
                    p = tree[0][p]
                pair_index = n1 * (n1 - 2 * n + 1) // -2 + n2 - n1 - 1
                m[tree_index][pair_index] = depth
                M[tree_index][pair_index] = tree[1][tree[2]] - tree[1][mrca_id]
    return m, M


def kc_distance(tree1, tree2, lambda_=0.0):
    """
    Returns the Kendall-Colijn distance between the specified pair of trees.
    lambda_ determines weight of topology vs branch lengths in calculating
    the distance. Set lambda_ at 0.0 to only consider topology, set at 1.0 to
    only consider branch lengths.

    .. seealso::
        See `Kendall & Colijn (2016)
        <https://academic.oup.com/mbe/article/33/10/2735/2925548>`_
        for more details.

    :param tskit.Tree tree1: The first tree to compare.
    :param tskit.Tree tree2: The second tree to compare.
    :param float lambda_: The weight of topology in the distance calculation.
    :return : The Kendall-Colijn distance between the trees.
    :rtype : float
    """
    samples = tree1.tree_sequence.samples()
    if not np.array_equal(samples, tree2.tree_sequence.samples()):
        raise ValueError("Trees must have the same samples")
    if tree1.num_roots != 1 or tree2.num_roots != 1:
        raise ValueError("Trees must have a single root")
    for tree in [tree1, tree2]:
        for u in tree.nodes():
            if tree.num_children(u) == 1:
                raise ValueError("Unary nodes are not supported")

    m, M = _kc_distance(
        samples,
        (
            tree1.parent_array,
            tree1.tree_sequence.tables.nodes.time,
            tree1.root,
        ),
        (
            tree2.parent_array,
            tree2.tree_sequence.tables.nodes.time,
            tree2.root,
        ),
    )

    return np.linalg.norm((1 - lambda_) * (m[0] - m[1]) + lambda_ * (M[0] - M[1]))
