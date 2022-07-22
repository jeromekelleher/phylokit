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

    def tsk_tree(self):
        return tskit.Tree.generate_balanced(3)

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 5

    def test_colless(self):
        assert pk.colless_index(self.tree()) == 1

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 1

    def test_b2(self):
        assert pk.b2_index(self.tree()) == pytest.approx(0.4515, rel=1e-3)


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
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 8

    def test_colless(self):
        assert pk.colless_index(self.tree()) == 0

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 2

    def test_b2(self):
        assert pk.b2_index(self.tree()) == pytest.approx(0.602, rel=1e-3)

    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            (2, 2),
            (3, 1.2618595071429148),
            (4, 1.0),
            (5, 0.8613531161467861),
            (10, 0.6020599913279623),
            (100, 0.30102999566398114),
            (1000000, 0.10034333188799373),
            (2.718281828459045, 1.3862943611198906),
        ],
    )
    def test_b2_base(self, base, expected):
        assert pk.b2_index(self.tree(), base) == expected

    @pytest.mark.parametrize("base", [0, -0.001, -1, -1e-6, -1e200])
    def test_b2_bad_base(self, base):
        with pytest.raises(ValueError, match="math domain"):
            pk.b2_index(self.tree(), base=base)

    def test_b2_base1(self):
        with pytest.raises(ZeroDivisionError):
            pk.b2_index(self.tree(), base=1)


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
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 18

    def test_colless(self):
        with pytest.raises(ValueError):
            pk.colless_index(self.tree())

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 3

    def test_b2(self):
        assert pk.b2_index(self.tree()) == pytest.approx(0.954, rel=1e-3)


class TestStarN10:
    # 1.00┊         10          ┊
    #     ┊ ┏━┳━┳━┳━┳┻┳━┳━┳━┳━┓ ┊
    # 0.00┊ 0 1 2 3 4 5 6 7 8 9 ┊
    #     0                     1
    def tsk_tree(self):
        return tskit.Tree.generate_star(10)

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 10

    def test_colless(self):
        with pytest.raises(ValueError):
            pk.colless_index(self.tree())

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 0

    def test_b2(self):
        assert pk.b2_index(self.tree()) == pytest.approx(0.9999, rel=1e-3)


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
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 14

    def test_colless(self):
        assert pk.colless_index(self.tree()) == 6

    def test_b1(self):
        assert pk.b1_index(self.tree()) == pytest.approx(1.833, rel=1e-3)

    def test_b2(self):
        assert pk.b2_index(self.tree(), base=10) == pytest.approx(0.564, rel=1e-3)


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

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 20

    def test_colless(self):
        with pytest.raises(ValueError):
            pk.colless_index(self.tree())

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 4.5

    def test_b2(self):
        with pytest.raises(ValueError):
            pk.b2_index(self.tree())


class TestEmpty:
    def tsk_tree(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 0

    def test_colless(self):
        with pytest.raises(ValueError):
            pk.colless_index(self.tree())

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 0

    def test_b2(self):
        with pytest.raises(ValueError):
            pk.b2_index(self.tree())


class TestTreeInNullState:
    def tsk_tree(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 0

    def test_colless(self):
        with pytest.raises(ValueError):
            pk.colless_index(self.tree())

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 0

    def test_b2(self):
        with pytest.raises(ValueError):
            pk.b2_index(self.tree())


class TestAllRootsN5:
    def tsk_tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_sackin(self):
        assert pk.sackin_index(self.tree()) == 0

    def test_colless(self):
        with pytest.raises(ValueError):
            pk.colless_index(self.tree())

    def test_b1(self):
        assert pk.b1_index(self.tree()) == 0

    def test_b2(self):
        with pytest.raises(ValueError):
            pk.b2_index(self.tree())
