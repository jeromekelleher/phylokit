# Tree balance/imbalance metrics.

import numba


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
