import numpy as np
import pytest
import tskit

import phylokit as pk


class TestPermuteTree:
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
        assert (
            pk.permute_tree(
                pk_tree,
                ordering=ordering,
            ).node_parent.data.all()
            == np.array([-1, 0, 0, 2, 2, 4, 4, -1]).all()
        )

    def generate_trees():
        for i in range(10, 20):
            yield tskit.Tree.generate_random_binary(10, random_seed=i)

    @pytest.mark.parametrize(("tsk_tree"), generate_trees())
    def test_permute_trees(self, tsk_tree):
        tree = pk.from_tskit(tsk_tree)
        permuted = pk.permute_tree(
            tree,
            np.append(tree.traversal_preorder.data, len(tree.node_parent.data) - 1),
        )
        reversed_permuted = pk.permute_tree(
            permuted,
            np.append(
                permuted.traversal_postorder.data, len(permuted.node_parent.data) - 1
            ),
        )
        assert reversed_permuted.node_parent.data.all() == tree.node_parent.data.all()
        assert (
            reversed_permuted.traversal_preorder.data.all()
            == tree.traversal_preorder.data.all()
        )
        assert (
            reversed_permuted.traversal_postorder.data.all()
            == tree.traversal_postorder.data.all()
        )
        assert reversed_permuted.samples.data.all() == tree.samples.data.all()

    @pytest.mark.parametrize(("tsk_tree"), generate_trees())
    def test_permute_tree_error(self, tsk_tree):
        permuted = pk.from_tskit(tsk_tree)
        ordering = permuted.traversal_preorder.data
        with pytest.raises(ValueError):
            pk.permute_tree(permuted, ordering)
