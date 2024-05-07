# Tree balance/imbalance metrics.
import math

import numpy as np

from . import jit
from . import util


@jit.numba_njit()
def _sackin_index(virtual_root, left_child, right_sib):
    stack = []
    root = left_child[virtual_root]
    while root != -1:
        stack.append((root, 0))
        root = right_sib[root]
    total_depth = 0
    while len(stack) > 0:
        u, depth = stack.pop()
        v = left_child[u]
        if v == -1:
            total_depth += depth
        else:
            depth += 1
            while v != -1:
                stack.append((v, depth))
                v = right_sib[v]
    return total_depth


def sackin_index(ds):
    """
    Returns the Sackin imbalance index for this tree.

    .. seealso::
        See `Shao and Sokal (1990) <https://www.jstor.org/stable/2992186>`_ for more
        details.

    :param xarray.DataSet ds: The tree dataset to compute the Sackin index of.
    :return : The Sackin index of the tree.
    :rtype : float
    """
    return _sackin_index(-1, ds.node_left_child.data, ds.node_right_sib.data)


@jit.numba_njit()
def _colless_index(postorder, left_child, right_sib):
    num_leaves = np.zeros_like(left_child)
    total = 0.0
    for u in postorder:
        v = left_child[u]
        num_children = 0
        while v != -1:
            num_children += 1
            num_leaves[u] += num_leaves[v]
            v = right_sib[v]
        if num_children == 0:
            num_leaves[u] = 1
        elif num_children == 2:
            v = left_child[u]
            total += abs(num_leaves[right_sib[v]] - num_leaves[v])
        else:
            raise ValueError("Colless index not defined for nonbinary trees")
    return total


def colless_index(ds):
    """
    Returns the Colless imbalance index for this tree.

    .. warning::
        Colless index is only defined for binary trees and is not defined for multiroot
        trees.

    .. seealso::
        See `Shao and Sokal (1990) <https://www.jstor.org/stable/2992186>`_ for more
        details.

    :param xarray.DataSet ds: The tree dataset to compute the Colless index of.
    :return : The Colless index of the tree.
    :rtype : float
    """
    if util.get_num_roots(ds) != 1:
        raise ValueError("Colless index not defined for multiroot trees")
    return _colless_index(
        ds.traversal_postorder.data,
        ds.node_left_child.data,
        ds.node_right_sib.data,
    )


@jit.numba_njit()
def _b1_index(postorder, left_child, right_sib, parent):
    max_path_length = np.zeros_like(postorder)
    total = 0.0
    for u in postorder:
        v = left_child[u]
        if parent[u] != -1 and v != -1:
            max_path_length[u] = (
                max(max_path_length[v], max_path_length[right_sib[v]]) + 1
            )
            total += 1 / max_path_length[u]
    return total


def b1_index(ds):
    """
    Returns the B1 balance index for this tree.

    .. seealso::
        See `Shao and Sokal (1990) <https://www.jstor.org/stable/2992186>`_ for more
        details.

    :param tskit.Tree tree: The tree to compute the B1 index of.
    :return : The B1 index of the tree.
    :rtype : float
    """
    return _b1_index(
        ds.traversal_postorder.data,
        ds.node_left_child.data,
        ds.node_right_sib.data,
        ds.node_parent.data,
    )


@jit.numba_njit()
def general_log(x, base):
    """
    Compute the logarithm of x in base `base`.

    :param x: The number to compute the logarithm of.
    :param base: The base of the logarithm.
    :return: The logarithm of x in base `base`.
    :rtype: float
    """
    return math.log(x) / math.log(base)


@jit.numba_njit()
def _b2_index(virtual_root, left_child, right_sib, base):
    root = left_child[virtual_root]
    stack = [(root, 1)]
    total_proba = 0.0
    while len(stack) > 0:
        u, path_product = stack.pop()
        if left_child[u] == -1:
            total_proba -= path_product * general_log(path_product, base)
        else:
            num_children = 0
            v = left_child[u]
            while v != -1:
                num_children += 1
                v = right_sib[v]
            path_product *= 1 / num_children
            v = left_child[u]
            while v != -1:
                stack.append((v, path_product))
                v = right_sib[v]
    return total_proba


def b2_index(ds, base=10):
    """
    Returns the B2 balance index for this tree.

    .. seealso::
        See `Shao and Sokal (1990) <https://www.jstor.org/stable/2992186>`_ for more
        details.

    :param tskit.Tree tree: The tree to compute the B2 index of.
    :param int base: The base of the logarithm used to compute the B2 index in the
    Shannon entropy computation.
    :return : The B2 index of the tree.
    :rtype : float
    """
    if util.get_num_roots(ds) != 1:
        raise ValueError("B2 index not defined for multiroot trees")
    math.log(10, base)  # Check that base is valid
    return _b2_index(
        -1,
        ds.node_left_child.data,
        ds.node_right_sib.data,
        base,
    )
