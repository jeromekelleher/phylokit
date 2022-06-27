# Tests for the tree distance metrics
import pytest
import tskit

import phylokit as pk


class TestTreeSameSamples:
    # Tree1
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1
    #
    # Tree2
    # 3.00┊   6     ┊
    #     ┊ ┏━┻━┓   ┊
    # 2.00┊ ┃   5   ┊
    #     ┊ ┃ ┏━┻┓  ┊
    # 1.00┊ ┃ ┃  4  ┊
    #     ┊ ┃ ┃ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         1

    def tree1(self):
        return tskit.Tree.generate_balanced(4)

    def tree2(self):
        return tskit.Tree.generate_comb(4)

    def test_mrca(self):
        assert pk.mrca(self.tree1(), 0, 1) == 4

    def test_mrca_same_node(self):
        assert pk.mrca(self.tree1(), 3, 3) == 3

    def test_mrca_out_of_range(self):
        with pytest.raises(ValueError):
            pk.mrca(self.tree1(), 2, 10)

    def test_branch_length(self):
        assert pk.branch_length(self.tree1(), 0) == 1.0

    def test_kc_distance(self):
        assert pk.kc_distance(self.tree1(), self.tree2(), 1) == 3.0


class TestTreeDifferentSamples:
    # Tree1
    # 2.00┊     6     ┊
    #     ┊   ┏━┻━┓   ┊
    # 1.00┊   4   5   ┊
    #     ┊  ┏┻┓ ┏┻┓  ┊
    # 0.00┊  0 1 2 3  ┊
    #     0           1
    #
    # Tree2
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
    def tree1(self):
        return tskit.Tree.generate_balanced(4)

    def tree2(self):
        return tskit.Tree.generate_comb(5)

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)


class TestTreeMultiRoots:
    # Tree1
    # 4.00┊        15             ┊
    #     ┊     ┏━━━┻━━━┓         ┊
    # 3.00┊     ┃      14         ┊
    #     ┊     ┃     ┏━┻━┓       ┊
    # 2.00┊    12     ┃  13       ┊
    #     ┊   ┏━┻━┓   ┃  ┏┻┓      ┊
    # 1.00┊   9  10   ┃  ┃ 11     ┊
    #     ┊  ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓    ┊
    # 0.00┊  0 1 2 3 4 5 6 7 8    ┊
    #     0                       1
    #
    # Tree2
    # 3.00┊              15       ┊
    #     ┊            ┏━━┻━┓     ┊
    # 2.00┊     11     ┃   14     ┊
    #     ┊    ┏━┻━┓   ┃  ┏━┻┓    ┊
    # 1.00┊    9  10  12  ┃ 13    ┊
    #     ┊   ┏┻┓ ┏┻┓ ┏┻┓ ┃ ┏┻┓   ┊
    # 0.00┊   0 1 2 3 4 5 6 7 8   ┊
    #     0                       1
    def tree1(self):
        return tskit.Tree.generate_balanced(9)

    def tree2(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def test_mrca(self):
        assert pk.mrca(self.tree2(), 0, 8) == -1

    def test_mrca_virtual_root(self):
        assert pk.mrca(self.tree2(), 11, 17) == 17

    def test_branch_length(self):
        assert pk.branch_length(self.tree2(), 0) == 1
        assert pk.branch_length(self.tree2(), 15) == 0

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)


class TestEmpty:
    def tree1(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def tree2(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def test_mrca(self):
        with pytest.raises(ValueError):
            pk.mrca(self.tree1(), 0, 1)

    def test_branch_length_out_of_bounds(self):
        with pytest.raises(ValueError):
            pk.branch_length(self.tree1(), 1)

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)


class TestTreeInNullState:
    def tree1(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def tree2(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)


class TestAllRootsN5:
    def tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def test_mrca(self):
        assert pk.mrca(self.tree(), 0, 1) == -1

    def test_branch_length(self):
        assert pk.branch_length(self.tree(), 5) == 0

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree(), self.tree(), 0)
