import msprime
import numpy as np
import pytest
import tskit
from numpy.testing import assert_array_equal

import phylokit as pk


def assert_tsk_pk_trees_equal(tsk_tree, pk_ds):
    assert_array_equal(tsk_tree.parent_array, pk_ds.node_parent)
    assert_array_equal(tsk_tree.left_child_array, pk_ds.node_left_child)
    assert_array_equal(tsk_tree.right_sib_array, pk_ds.node_right_sib)
    assert_array_equal(tsk_tree.tree_sequence.samples(), pk_ds.sample_node)
    time = tsk_tree.tree_sequence.tables.nodes.time
    assert_array_equal(time, pk_ds.node_time[:-1])


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
        assert_array_equal(ds.sample_node, [1, 2])


# TODO
class TestToTskit:
    pass


# Defining this function here for now, but it should be implemented and jitted.
def pk_path_to_root(ds, u):
    path = []
    while u != -1:
        path.append(u)
        u = ds.node_parent[u]
    return np.array(path)


def tsk_path_to_root(tree, u):
    path = []
    while u != -1:
        path.append(u)
        u = tree.parent(u)
    return np.array(path)


class TestFromNewick:
    def validate_tsk_tree(self, tsk_tree):
        ds = pk.from_newick(tsk_tree.as_newick())
        for tsk_u in tsk_tree.samples():
            sample_id = f"n{tsk_u}"
            # FIXME This is a stupid way to do it, but works until we figure out the
            # right way to do indexing by sample name.
            sample_index = np.where(np.array(ds.sample_id) == sample_id)[0]
            pk_u = int(ds.sample_node[sample_index])
            path_pk = pk_path_to_root(ds, pk_u)
            path_tsk = tsk_path_to_root(tsk_tree, tsk_u)
            assert path_pk.shape == path_tsk.shape
            tsk_branch_length = [tsk_tree.branch_length(v) for v in path_tsk]
            assert_array_equal(ds.node_branch_length[path_pk], tsk_branch_length)

    @pytest.mark.parametrize("n", [3, 10, 20])
    @pytest.mark.parametrize("arity", [2, 3, 5])
    def test_balanced(self, n, arity):
        tsk_tree = tskit.Tree.generate_balanced(n, arity=arity)
        self.validate_tsk_tree(tsk_tree)

    @pytest.mark.parametrize("n", [3, 10, 20, 50])
    def test_comb(self, n):
        tsk_tree = tskit.Tree.generate_comb(n)
        self.validate_tsk_tree(tsk_tree)


class TestToNewick:
    @pytest.mark.parametrize("n", [3, 10, 20])
    @pytest.mark.parametrize("arity", [2, 3, 5])
    def test_balanced(self, n, arity):
        tsk_tree = tskit.Tree.generate_balanced(n, arity=arity)
        pk_tree = pk.from_tskit(tsk_tree)
        assert tsk_tree.as_newick() == pk.to_newick(pk_tree)

    @pytest.mark.parametrize("n", [3, 10, 20, 50])
    def test_comb(self, n):
        tsk_tree = tskit.Tree.generate_comb(n)
        pk_tree = pk.from_tskit(tsk_tree)
        assert tsk_tree.as_newick() == pk.to_newick(pk_tree)


@pytest.mark.parametrize(
    "s",
    [
        "(n0:2,(n1:1,n2:1):1);",
        "((n0:1,n1:1):1,(n2:2,(n3:1,n4:1):1):1);",
        "((1:1,2:1,3:1):1,(4:1,5:1,6:1):1,(7:1,8:1,9:1):1);",
        "(((n0:1):1):1);",
        "(:2,(:1,:1):1);",
        "(one:2,(two:1,three:1):1);",
        "(n0:3.5,(n1:1.25,n2:1.25):100);",
    ],
)
def test_newick_round_trip(s):
    ds = pk.from_newick(s)
    s2 = pk.to_newick(ds)
    assert s == s2
