import numpy as np
import pytest
import tskit

import phylokit as pk


class TestPermuteTree:
    def get_reversed_map(self, ds, original_seq):
        reversed_map = np.zeros(len(ds.node_left_child.data), dtype=np.int32)
        for u, v in enumerate(original_seq):
            reversed_map[v] = u
        return reversed_map

    def test_permute_tree(self):
        # before permutation
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        #
        # after permutation
        # 3.00┊   0     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   2   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 1 3 5 6 ┊
        #     0         1
        tsk_tree = tskit.Tree.generate_comb(4)
        pk_tree = pk.from_tskit(tsk_tree)

        ordering = np.append(
            pk_tree.traversal_preorder.data, len(pk_tree.node_parent.data) - 1
        )
        permuted_tree = pk.permute_tree(
            pk_tree,
            ordering=ordering,
        )

        assert np.array_equal(
            permuted_tree.node_parent.data,
            np.array([-1, 0, 0, 2, 2, 4, 4, -1]),
        )

        assert np.array_equal(
            permuted_tree.node_left_child.data,
            np.array([1, -1, 3, -1, 5, -1, -1, 0]),
        )

        assert np.array_equal(
            permuted_tree.node_right_sib.data,
            np.array([-1, 2, -1, 4, -1, 6, -1, -1]),
        )

        assert np.array_equal(
            permuted_tree.sample_node.data,
            np.array([1, 3, 5, 6]),
        )

    def generate_trees():
        for i in range(10, 20):
            yield tskit.Tree.generate_random_binary(10, random_seed=i)

    @pytest.mark.parametrize(("tsk_tree"), generate_trees())
    def test_permute_trees(self, tsk_tree):
        tree = pk.from_tskit(tsk_tree)
        original_ordering = np.append(
            tree.traversal_preorder.data, len(tree.node_parent.data) - 1
        )
        reversed_map = self.get_reversed_map(tree, original_ordering)

        permuted = pk.permute_tree(
            tree,
            np.append(tree.traversal_preorder.data, len(tree.node_parent.data) - 1),
        )
        reversed_permuted = pk.permute_tree(
            permuted,
            reversed_map,
        )
        assert np.array_equal(
            reversed_permuted.node_parent.data,
            tree.node_parent.data,
        )
        assert np.array_equal(
            reversed_permuted.traversal_preorder.data,
            tree.traversal_preorder.data,
        )
        assert np.array_equal(
            reversed_permuted.traversal_postorder.data,
            tree.traversal_postorder.data,
        )
        assert np.array_equal(
            reversed_permuted.sample_node.data,
            tree.sample_node.data,
        )

    @pytest.mark.parametrize(("tsk_tree"), generate_trees())
    def test_permute_tree_error(self, tsk_tree):
        permuted = pk.from_tskit(tsk_tree)
        ordering = permuted.traversal_preorder.data
        with pytest.raises(ValueError):
            pk.permute_tree(permuted, ordering)
