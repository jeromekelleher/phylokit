import msprime
import numpy as np
import pytest
import xarray as xr

import phylokit as pk


class TestCreateTreeDataset:
    def create_test_tree_dataset():
        trees = []
        tsa = msprime.sim_ancestry(100, sequence_length=100, ploidy=1, random_seed=1234)
        ts = msprime.sim_mutations(tsa, rate=0.03, random_seed=1234)
        tsk_tree = ts.first()
        pk_tree_minimal = pk.core.create_tree_dataset(
            parent=tsk_tree.parent_array,
            time=np.append(ts.tables.nodes.time, np.inf),
            left_child=tsk_tree.left_child_array,
            right_sib=tsk_tree.right_sib_array,
            samples=ts.samples(),
        )
        trees.append(pk_tree_minimal)
        trees.append(pk.from_tskit(tsk_tree))
        return trees

    @pytest.mark.parametrize("ds", create_test_tree_dataset())
    def test_save_and_open_dataset(self, ds, tmp_path):
        path = tmp_path / "test.zarr"
        pk.save_dataset(ds, path)
        ds2 = pk.open_dataset(path)
        xr.testing.assert_identical(ds, ds2)

    @pytest.mark.parametrize("ds", create_test_tree_dataset()[:1])
    def test_save_and_open_dataset_str(self, ds, tmp_path):
        path = str(tmp_path / "test.zarr")
        pk.save_dataset(ds, path)
        ds2 = pk.open_dataset(path)
        xr.testing.assert_identical(ds, ds2)
