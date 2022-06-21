# Tree balance/imbalance metrics.

import numba
import numpy as np


@numba.njit()
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


def sackin_index(tree):
    return _sackin_index(tree.virtual_root, tree.left_child_array, tree.right_sib_array)


@numba.njit
def _colless_index(postorder, left_child, right_sib):
    """
    Returns the Colless imbalance index of the tree.

    :param numpy.ndarray postorder: The postorder traversal of the tree.
    :param numpy.ndarray left_child: The left child array of the tree.
    :param numpy.ndarray right_sib: The right sibling array of the tree.
    :return : The Colless index of the tree.
    :rtype : float
    """
    num_leaves = np.zeros_like(left_child)
    total = 0
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


def colless_index(tree):
    """
    Returns the Colless imbalance index for this tree.

    .. warning::
        Colless index is only defined for binary trees and is not defined for multiroot trees.

    .. seealso::
        See `Shao and Sokal (1990) <https://www.jstor.org/stable/2992186>`_ for more details.

    :param tskit.Tree tree: The tree to compute the Colless index of.
    :return : The Colless index of the tree.
    :rtype : float
    """
    if tree.num_roots != 1:
        raise ValueError("Colless index not defined for multiroot trees")
    # TODO check tree.is_binary, somehow
    # for n in tree.postorder():
    #     if len(tree.children(n)) > 2:
    #         raise ValueError("Colless index not defined for nonbinary trees")

    return _colless_index(tree.postorder(), tree.left_child_array, tree.right_sib_array)


@numba.njit(cached=True)
def _b1_index(postorder, left_child, right_sib, parent):
    """
    Returns the B1 balance index of the tree.

    :param numpy.ndarray postorder: The postorder traversal of the tree.
    :param numpy.ndarray left_child: The left child array of the tree.
    :param numpy.ndarray right_sib: The right sibling array of the tree.
    :param numpy.ndarray parent: The parent array of the tree.
    :return : The B1 index of the tree.
    :rtype : float
    """
    max_path_length = np.zeros_like(postorder)
    total = 0
    for u in postorder:
        v = left_child[u]
        if parent[u] != -1 and v != -1:
            max_path_length[u] = (
                max(max_path_length[v], max_path_length[right_sib[v]]) + 1
            )
            total += 1 / max_path_length[u]
    return total


def b1_index(tree):
    """
    Returns the B1 balance index for this tree.

    .. seealso::
        See `Shao and Sokal (1990) <https://www.jstor.org/stable/2992186>`_ for more details.

    :param tskit.Tree tree: The tree to compute the B1 index of.
    :return : The B1 index of the tree.
    :rtype : float
    """
    return _b1_index(
        tree.postorder(), tree.left_child_array, tree.right_sib_array, tree.parent_array
    )
