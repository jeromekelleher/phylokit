import numpy as np
import tskit

import phylokit as pk


class TestBranchLengths:
    def tsk_tree(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0      1
        return tskit.Tree.generate_balanced(3)

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_branch_lengths(self):
        assert np.all(pk.get_node_branch_length(self.tree()) == [2, 1, 1, 1, 0, 0])

    def test_time_massive(self):
        for i in range(10, 20):
            tsk_tree = tskit.Tree.generate_random_binary(i, random_seed=i)
            pk_tree = pk.from_tskit(tsk_tree)
            node_branch_length = pk.get_node_branch_length(pk_tree)
            for i in range(node_branch_length.shape[0]):
                assert node_branch_length[i] == tsk_tree.branch_length(i)


class TestTime:
    def tsk_tree(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0      1
        return tskit.Tree.generate_balanced(3)

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_time_equal(self):
        assert np.all(pk.get_node_time(self.tree()) == np.array([0, 0, 0, 1, 2, 0]))

    def test_time_massive(self):
        for i in range(10, 20):
            tsk_tree = tskit.Tree.generate_random_binary(i, random_seed=i)
            pk_tree = pk.from_tskit(tsk_tree)
            assert np.all(
                pk.get_node_time(pk_tree)
                == np.append(tsk_tree.tree_sequence.tables.nodes.time, 0)
            )
