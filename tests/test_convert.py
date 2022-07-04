import msprime
import numpy.testing
import pytest
import tskit

import phylokit as pk


def assert_tsk_pk_trees_equal(tsk_tree, pk_ds):
    numpy.testing.assert_array_equal(tsk_tree.parent_array, pk_ds.node_parent)
    numpy.testing.assert_array_equal(tsk_tree.left_child_array, pk_ds.node_left_child)
    numpy.testing.assert_array_equal(tsk_tree.right_sib_array, pk_ds.node_right_sib)
    numpy.testing.assert_array_equal(
        tsk_tree.tree_sequence.samples(), pk_ds.sample_node
    )
    time = tsk_tree.tree_sequence.tables.nodes.time
    numpy.testing.assert_array_equal(time, pk_ds.node_time[:-1])


class TestFromTskit:
    @pytest.mark.parametrize("n", [3, 10, 20])
    @pytest.mark.parametrize("arity", [2, 3, 5])
    def test_balanced(self, n, arity):
        tree = tskit.Tree.generate_balanced(n, arity=arity)
        ds = pk.from_tskit(tree)
        assert_tsk_pk_trees_equal(tree, ds)
        tree2 = pk.to_tskit(ds)
        # In this case the round-trip should be identical
        t1 = tree.tree_sequence.tables
        t2 = tree2.tree_sequence.tables
        t1.assert_equals(t2, ignore_provenance=True)

    @pytest.mark.parametrize("n", [3, 10, 20])
    def test_msprime_single_tree(self, n):
        tree = msprime.sim_ancestry(n, ploidy=1, random_seed=2).first()
        ds = pk.from_tskit(tree)
        assert_tsk_pk_trees_equal(tree, ds)

    def test_samples_not_zero(self):
        tables = tskit.TableCollection(1)
        tables.nodes.add_row(time=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)
        tables.edges.add_row(0, 1, 0, 1)
        tables.edges.add_row(0, 1, 0, 2)
        tables.sort()
        tree = tables.tree_sequence().first()

        ds = pk.from_tskit(tree)
        assert_tsk_pk_trees_equal(tree, ds)
        numpy.testing.assert_array_equal(ds.sample_node, [1, 2])


# TODO
class TestToTskit:
    pass
