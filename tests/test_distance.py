# Tests for the tree distance metrics
import itertools

import dendropy
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

    def tsk_tree1(self):
        return tskit.Tree.generate_balanced(4)

    def tsk_tree2(self):
        return tskit.Tree.generate_comb(4)

    def tree1(self):
        return pk.from_tskit(self.tsk_tree1())

    def tree2(self):
        return pk.from_tskit(self.tsk_tree2())

    def test_mrca(self):
        assert pk.mrca(self.tree1(), 0, 1) == 4

    def test_mrca_same_node(self):
        assert pk.mrca(self.tree1(), 3, 3) == 3

    def test_mrca_out_of_range(self):
        with pytest.raises(ValueError):
            pk.mrca(self.tree1(), 2, 10)

    def test_kc_distance(self):
        assert pk.kc_distance(self.tree1(), self.tree2(), 1) == 3.0

    def test_rf_distance(self):
        assert pk.rf_distance(self.tree1(), self.tree2()) == 2


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

    def tsk_tree1(self):
        return tskit.Tree.generate_balanced(4)

    def tsk_tree2(self):
        return tskit.Tree.generate_comb(5)

    def tree1(self):
        return pk.from_tskit(self.tsk_tree1())

    def tree2(self):
        return pk.from_tskit(self.tsk_tree2())

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)

    def test_rf_distance(self):
        assert pk.rf_distance(self.tree1(), self.tree2()) == 8


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

    def tsk_tree1(self):
        return tskit.Tree.generate_balanced(9)

    def tsk_tree2(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return tables.tree_sequence().first()

    def tree1(self):
        return pk.from_tskit(self.tsk_tree1())

    def tree2(self):
        return pk.from_tskit(self.tsk_tree2())

    def test_mrca(self):
        assert pk.mrca(self.tree2(), 0, 8) == -1

    def test_mrca_virtual_root(self):
        assert pk.mrca(self.tree2(), 11, 17) == 17

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            pk.rf_distance(self.tree1(), self.tree2())


class TestEmpty:
    def tsk_tree1(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def tsk_tree2(self):
        tables = tskit.TableCollection(1)
        return tables.tree_sequence().first()

    def tree1(self):
        return pk.from_tskit(self.tsk_tree1())

    def tree2(self):
        return pk.from_tskit(self.tsk_tree2())

    def test_mrca(self):
        with pytest.raises(ValueError):
            pk.mrca(self.tree1(), 0, 1)

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            pk.rf_distance(self.tree1(), self.tree2())


class TestTreeInNullState:
    def tsk_tree1(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def tsk_tree2(self):
        tree = tskit.Tree.generate_comb(5)
        tree.clear()
        return tree

    def tree1(self):
        return pk.from_tskit(self.tsk_tree1())

    def tree2(self):
        return pk.from_tskit(self.tsk_tree2())

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree1(), self.tree2(), 0)

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            pk.rf_distance(self.tree1(), self.tree2())


class TestAllRootsN5:
    def tsk_tree(self):
        tables = tskit.TableCollection(1)
        for _ in range(5):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        return tables.tree_sequence().first()

    def tree(self):
        return pk.from_tskit(self.tsk_tree())

    def test_mrca(self):
        assert pk.mrca(self.tree(), 0, 1) == -1

    def test_kc_distance(self):
        with pytest.raises(ValueError):
            pk.kc_distance(self.tree(), self.tree(), 0)

    def test_rf_distance(self):
        with pytest.raises(ValueError):
            pk.rf_distance(self.tree(), self.tree())


class TestRFDistance:
    def setup_method(self):
        self.taxon_namespace = dendropy.TaxonNamespace()

    def get_non_consecutive_leaf_tree(self):
        tables = tskit.Tree.generate_balanced(3).tree_sequence.dump_tables()
        tables.nodes.add_row(time=-1)
        tables.nodes.add_row(time=-1)
        edges = tables.edges
        edges.add_row(left=0, right=1, parent=0, child=5)
        edges.add_row(left=0, right=1, parent=0, child=6)
        tables.sort()
        return tables.tree_sequence().first()

    def to_dendropy(self, newick_data):
        return dendropy.Tree.get(
            data=newick_data,
            schema="newick",
            rooting="force-rooted",
            taxon_namespace=self.taxon_namespace,
        )

    def generate_trees():
        for i in range(10, 20):
            yield tskit.Tree.generate_random_binary(10, random_seed=i)

    @pytest.mark.parametrize(
        ("tree1", "tree2"),
        itertools.combinations(generate_trees(), 2),
    )
    def test_rf_distance(self, tree1, tree2):
        pk_tree1 = pk.from_tskit(tree1)
        pk_tree2 = pk.from_tskit(tree2)
        dendropy_tree1 = self.to_dendropy(tree1.as_newick())
        dendropy_tree2 = self.to_dendropy(tree2.as_newick())
        assert pk.rf_distance(
            pk_tree1, pk_tree2
        ) == dendropy.calculate.treecompare.symmetric_difference(
            dendropy_tree1, dendropy_tree2
        )

    def test_rf_leaves_non_consecutive_leaves(self):
        t1 = self.get_non_consecutive_leaf_tree()
        t2 = tskit.Tree.generate_balanced(4)
        assert pk.rf_distance(pk.from_tskit(t1), pk.from_tskit(t2)) == 10
