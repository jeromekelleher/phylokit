import numpy as np
import tskit

import phylokit as pk


class TestBalancedBinaryOdd:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0      1

    def tsk_tree(self):
        return tskit.Tree.generate_balanced(3)

    def tree(self):
        return pk.from_tskit(tskit.Tree.generate_balanced(3))

    def test_postorder(self):
        assert np.array_equal(pk.postorder(self.tree()), [0, 1, 2, 3, 4])

    def test_preorder(self):
        assert np.array_equal(pk.preorder(self.tree()), [4, 0, 3, 1, 2])

    def test_postorder_from_node(self):
        assert np.array_equal(pk.postorder(self.tree(), 3), [1, 2, 3])

    def test_preorder_from_node(self):
        assert np.array_equal(pk.preorder(self.tree(), 3), [3, 1, 2])


class TestBalancedBinaryEven:
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1
    def tsk_tree(self):
        return tskit.Tree.generate_balanced(4)

    def tree(self):
        return pk.from_tskit(tskit.Tree.generate_balanced(4))

    def test_postorder(self):
        assert np.array_equal(pk.postorder(self.tree()), [0, 1, 4, 2, 3, 5, 6])

    def test_preorder(self):
        assert np.array_equal(pk.preorder(self.tree()), [6, 4, 0, 1, 5, 2, 3])


class TestBalancedTernary:
    # 2.00┊        12         ┊
    #     ┊   ┏━━━━━╋━━━━━┓   ┊
    # 1.00┊   9    10    11   ┊
    #     ┊ ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    def tsk_tree(self):
        return tskit.Tree.generate_balanced(9, arity=3)

    def tree(self):
        return pk.from_tskit(tskit.Tree.generate_balanced(9, arity=3))

    def test_postorder(self):
        assert np.array_equal(
            pk.postorder(self.tree()), [0, 1, 2, 9, 3, 4, 5, 10, 6, 7, 8, 11, 12]
        )

    def test_preorder(self):
        assert np.array_equal(
            pk.preorder(self.tree()), [12, 9, 0, 1, 2, 10, 3, 4, 5, 11, 6, 7, 8]
        )


class TestStarN10:
    # 1.00┊         10          ┊
    #     ┊ ┏━┳━┳━┳━┳┻┳━┳━┳━┳━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 9 ┊
    #     0                     1
    def tsk_tree(self):
        return tskit.Tree.generate_star(10)

    def tree(self):
        return pk.from_tskit(tskit.Tree.generate_star(10))

    def test_postorder(self):
        assert np.array_equal(
            pk.postorder(self.tree()), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )

    def test_preorder(self):
        assert np.array_equal(
            pk.preorder(self.tree()), [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )


class TestCombN5:
    # 4.00┊   8       ┊
    #     ┊ ┏━┻━┓     ┊
    # 3.00┊ ┃   7     ┊
    #     ┊ ┃ ┏━┻━┓   ┊
    # 2.00┊ ┃ ┃   6   ┊
    #     ┊ ┃ ┃ ┏━┻┓  ┊
    # 1.00┊ ┃ ┃ ┃  5  ┊
    #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 4 ┊
    #     0           1
    def tsk_tree(self):
        return tskit.Tree.generate_comb(5)

    def tree(self):
        return pk.from_tskit(tskit.Tree.generate_comb(5))

    def test_postorder(self):
        assert np.array_equal(pk.postorder(self.tree()), [0, 1, 2, 3, 4, 5, 6, 7, 8])

    def test_preorder(self):
        assert np.array_equal(pk.preorder(self.tree()), [8, 0, 7, 1, 6, 2, 5, 3, 4])


class TestMultiRootBinary:
    # 3.00┊            15     ┊
    #     ┊          ┏━━┻━┓   ┊
    # 2.00┊   11     ┃   14   ┊
    #     ┊  ┏━┻━┓   ┃  ┏━┻┓  ┊
    # 1.00┊  9  10  12  ┃ 13  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    def tsk_tree(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_postorder(self):
        assert np.array_equal(
            pk.postorder(self.tree()),
            [0, 1, 9, 2, 3, 10, 11, 4, 5, 12, 6, 7, 8, 13, 14, 15],
        )

    def test_preorder(self):
        assert np.array_equal(
            pk.preorder(self.tree()),
            [11, 9, 0, 1, 10, 2, 3, 15, 12, 4, 5, 14, 6, 13, 7, 8],
        )


class TestEmpty:
    def tsk_tree(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_postorder(self):
        assert np.array_equal(pk.postorder(self.tree()), [])

    def test_preorder(self):
        assert np.array_equal(pk.preorder(self.tree()), [])


class TestTreeInNullState:
    def tsk_tree(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_postorder(self):
        assert np.array_equal(pk.postorder(self.tree()), [0, 1, 2, 3, 4])

    def test_preorder(self):
        assert np.array_equal(pk.preorder(self.tree()), [0, 1, 2, 3, 4])


class TestAllRootsN5:
    def tsk_tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_postorder(self):
        assert np.array_equal(pk.postorder(self.tree()), [0, 1, 2, 3, 4])

    def test_preorder(self):
        assert np.array_equal(pk.preorder(self.tree()), [0, 1, 2, 3, 4])
