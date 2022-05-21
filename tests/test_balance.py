# Tests for the tree balance/imbalance metrics
import pytest

import tskit

import phylokit as pk


class TestBalancedBinaryOdd:
    # 2.00┊   4   ┊
    #     ┊ ┏━┻┓  ┊
    # 1.00┊ ┃  3  ┊
    #     ┊ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 ┊
    #     0      1
    def tree(self):
        return tskit.Tree.generate_balanced(3)

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 5

    # def test_colless(self):
    #     assert self.tree().colless_index() == 1

    # def test_b1(self):
    #     assert self.tree().b1_index() == 1


class TestBalancedBinaryEven:
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1
    def tree(self):
        return tskit.Tree.generate_balanced(4)

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 8

    # def test_colless(self):
    #     assert self.tree().colless_index() == 0

    # def test_b1(self):
    #     assert self.tree().b1_index() == 2


class TestBalancedTernary:
    # 2.00┊        12         ┊
    #     ┊   ┏━━━━━╋━━━━━┓   ┊
    # 1.00┊   9    10    11   ┊
    #     ┊ ┏━╋━┓ ┏━╋━┓ ┏━╋━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    def tree(self):
        return tskit.Tree.generate_balanced(9, arity=3)

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 18

    # def test_colless(self):
    #     with pytest.raises(ValueError):
    #         self.tree().colless_index()

    # def test_b1(self):
    #     assert self.tree().b1_index() == 3


class TestStarN10:
    # 1.00┊         10          ┊
    #     ┊ ┏━┳━┳━┳━┳┻┳━┳━┳━┳━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 9 ┊
    #     0                     1
    def tree(self):
        return tskit.Tree.generate_star(10)

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 10

    # def test_colless(self):
    #     with pytest.raises(ValueError):
    #         self.tree().colless_index()

    # def test_b1(self):
    #     assert self.tree().b1_index() == 0


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
    def tree(self):
        return tskit.Tree.generate_comb(5)

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 14

    # def test_colless(self):
    #     assert self.tree().colless_index() == 6

    # def test_b1(self):
    #     assert self.tree().b1_index() == pytest.approx(1.833, rel=1e-3)


class TestMultiRootBinary:
    # 3.00┊            15     ┊
    #     ┊          ┏━━┻━┓   ┊
    # 2.00┊   11     ┃   14   ┊
    #     ┊  ┏━┻━┓   ┃  ┏━┻┓  ┊
    # 1.00┊  9  10  12  ┃ 13  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 ┊
    #     0                   1
    def tree(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 20

    # def test_colless(self):
    #     with pytest.raises(ValueError):
    #         self.tree().colless_index()

    # def test_b1(self):
    #     assert self.tree().b1_index() == 4.5


class TestEmpty:
    def tree(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 0

    # def test_colless(self):
    #     with pytest.raises(ValueError):
    #         self.tree().colless_index()

    # def test_b1(self):
    #     assert self.tree().b1_index() == 0


class TestTreeInNullState:
    def tree(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 0

    # def test_colless(self):
    #     with pytest.raises(ValueError):
    #         self.tree().colless_index()

    # def test_b1(self):
    #     assert self.tree().b1_index() == 0


class TestAllRootsN5:
    def tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 0

    # def test_colless(self):
    #     with pytest.raises(ValueError):
    #         self.tree().colless_index()

    # def test_b1(self):
    #     assert self.tree().b1_index() == 0
