import numpy as np

from . import jit


@jit.numba_njit()
def _postorder(left_child, right_sib, root):
    # Another implementation with python stack operations such as `pop`
    # makes the same function about 2X slower.
    root = root if root is not None else -1
    num_nodes = len(left_child) - 1
    ret = np.zeros(num_nodes, dtype=np.int32)
    stack = np.zeros(num_nodes, dtype=np.int32)
    stack_top = 0
    num_roots = 0
    count_node = 0
    if root == -1:
        stack_top = -1
        v = left_child[root]
        while v != -1:
            num_roots += 1
            stack_top += 1
            stack[stack_top] = v
            v = right_sib[v]
    else:
        if root < 0 or root > num_nodes:
            return ret
        stack_top = 0
        stack[stack_top] = root

    while stack_top >= 0:
        u = stack[stack_top]
        stack_top -= 1
        ret[num_nodes - 1] = u
        num_nodes -= 1
        count_node += 1
        v = left_child[u]
        while v != -1:
            stack_top += 1
            stack[stack_top] = v
            v = right_sib[v]

    return ret[-count_node:]


def postorder(ds, root=None):
    """
    Returns the postorder traversal of the tree.

    :param xarray.DataSet ds: The tree dataset to compute the postorder traversal of.
    :param int root: The root node to start the traversal from.
    :return : The postorder traversal of the tree.
    :rtype : numpy.ndarray
    """
    return _postorder(
        ds.node_left_child.data,
        ds.node_right_sib.data,
        root,
    )


@jit.numba_njit()
def _preorder(parent, left_child, right_sib, root):
    # Another implementation with python stack operations such as `pop`
    # makes the same function about 2X slower.
    root = root if root is not None else -1
    num_nodes = len(left_child) - 1
    ret = np.zeros(num_nodes, dtype=np.int32)
    stack = np.zeros(num_nodes, dtype=np.int32)
    stack_top = 0
    num_roots = 0
    count_node = 0
    if root == -1:
        stack_top = -1
        u = left_child[root]
        while u != -1:
            num_roots += 1
            stack_top += 1
            stack[stack_top] = u
            u = right_sib[u]
    else:
        if root < 0 or root > num_nodes:
            return ret
        stack_top = 0
        stack[stack_top] = root

    preorder_parent = -1
    while stack_top >= 0:
        u = stack[stack_top]
        if left_child[u] != -1 and u != preorder_parent:
            v = left_child[u]
            while v != -1:
                stack_top += 1
                stack[stack_top] = v
                v = right_sib[v]
        else:
            stack_top -= 1
            preorder_parent = parent[u]
            ret[num_nodes - 1] = u
            count_node += 1
            num_nodes -= 1

    return ret[-count_node:]


def preorder(ds, root=None):
    """
    Returns the preorder traversal of the tree.

    :param xarray.DataSet ds: The tree dataset to compute the preorder traversal of.
    :param int root: The root node to start the traversal from.
    :return : The preorder traversal of the tree.
    :rtype : numpy.ndarray
    """
    return _preorder(
        ds.node_parent.data,
        ds.node_left_child.data,
        ds.node_right_sib.data,
        root,
    )
