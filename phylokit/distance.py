# Tree distance metrics.
import numpy as np

from . import jit
from . import util


@jit.numba_njit()
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


def _sv_mrca(ds, u, v):
    # TODO - Schieber, B. and U. Vishkin (1988).
    # "On Finding Lowest Common Ancestors: Simplification and Parallelization."
    # SIAM Journal on Computing 17(6): 1253-1262.
    pass


def mrca(ds, u, v):
    """
    Returns the most recent common ancestor of the specified nodes.

    :param xarray.DataArray ds: The tree to compare.
    :param int u: The first node ID.
    :param int v: The second node ID.
    :return: The most recent common ancestor of input nodes.
    :rtype: int
    """
    virtual_root = len(ds.node_parent.data) - 1
    # Check if u or v is outside the tree
    util.check_node_bounds(ds, u, v)
    # Check if u and v are virtual roots
    if u == virtual_root or v == virtual_root:
        return virtual_root
    if "sv_tau" in ds:
        return _sv_mrca(ds, u, v)
    else:
        return _mrca(ds.node_parent.data, ds.node_time.data, u, v)


@jit.numba_njit()
def _kc_distance(samples, ds1, ds2):
    # ds1 and ds2 are tuples of the form (parent_array, time_array, branch_length, root)
    n = samples.shape[0]
    N = (n * (n - 1)) // 2
    m = [np.zeros(N + n), np.zeros(N + n)]
    M = [np.zeros(N + n), np.zeros(N + n)]
    for tree_index, tree in enumerate([ds1, ds2]):
        for sample in range(n):
            m[tree_index][N + sample] = 1
            M[tree_index][N + sample] = tree[2][sample]

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
                M[tree_index][pair_index] = tree[1][tree[3]] - tree[1][mrca_id]
    return m, M


def kc_distance(ds1, ds2, lambda_=0.0):
    """
    Returns the Kendall-Colijn distance between the specified pair of trees.
    lambda_ determines weight of topology vs branch lengths in calculating
    the distance. Set lambda_ at 0.0 to only consider topology, set at 1.0 to
    only consider branch lengths.

    .. seealso::
        See `Kendall & Colijn (2016)
        <https://academic.oup.com/mbe/article/33/10/2735/2925548>`_
        for more details.

    :param xarray.DataArray ds1: The first tree to compare.
    :param xarray.DataArray ds2: The second tree to compare.
    :param float lambda_: The weight of topology in the distance calculation.
    :return : The Kendall-Colijn distance between the trees.
    :rtype : float
    """
    samples = ds1.sample_node.data
    if not np.array_equal(samples, ds2.sample_node.data):
        raise ValueError("Trees must have the same samples")
    if util.get_num_roots(ds1) != 1 or util.get_num_roots(ds2) != 1:
        raise ValueError("Trees must have a single root")
    for tree in [ds1, ds2]:
        if util.is_unary(tree):
            raise ValueError("Unary nodes are not supported")

    m, M = _kc_distance(
        samples,
        (
            ds1.node_parent.data,
            ds1.node_time.data[:-1],
            ds1.node_branch_length.data,
            ds1.node_left_child.data[-1],
        ),
        (
            ds2.node_parent.data,
            ds2.node_time.data[:-1],
            ds2.node_branch_length.data,
            ds2.node_left_child.data[-1],
        ),
    )

    return np.linalg.norm((1 - lambda_) * (m[0] - m[1]) + lambda_ * (M[0] - M[1]))


def get_node_partition(postorder, left_child, right_sib):
    """
    This is a naive implementation of encoding a node's partition
    using python's set operations to calculate the unweighted robinson
    foulds distance (symmetric distance)
    NOTE: This pure python implementation is faster than the jitted implementation,
    the speed of traversing the tree is faster in jitted version, but it took to
    much time calculating bitwise_or, therefore we use the naive implementation for
    now.
    NOTE: A potential optimization is using bit shifting rather than bitwise_or
    """
    tree_sets = [set()] * (len(left_child) - 1)
    for u in postorder:
        v = left_child[u]
        if v == -1:
            tree_sets[u] = {u}
        else:
            while v != -1:
                tree_sets[u] = tree_sets[u] | tree_sets[v]
                v = right_sib[v]

    return tree_sets


def rf_distance(ds1, ds2):
    """
    Returns the Robinson-Foulds distance between the specified pair of trees.

    .. seealso::
        See `Robinson & Foulds (1981)
        <https://doi.org/10.1016/0025-5564(81)90043-2>`_ for more details.

    :param xarray.DataArray ds1: The first tree to compare.
    :param xarray.DataArray ds2: The second tree to compare.
    :return : The Robinson-Foulds distance between the trees.
    :rtype : int
    """
    if util.get_num_roots(ds1) != 1 or util.get_num_roots(ds2) != 1:
        raise ValueError("Trees must have a single root")

    b1 = get_node_partition(
        ds1.traversal_postorder.data,
        ds1.node_left_child.data,
        ds1.node_right_sib.data,
    )
    b2 = get_node_partition(
        ds2.traversal_postorder.data,
        ds2.node_left_child.data,
        ds2.node_right_sib.data,
    )

    s1 = {frozenset(x) for x in b1}
    s2 = {frozenset(x) for x in b2}

    return len(s1.symmetric_difference(s2))
