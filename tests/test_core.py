import numpy as np
import xarray

from phylokit import core


class TestCreateTreeDataset:
    def test_single_node(self):
        ds = core.create_tree_dataset(
            parent=[-1, -1],
            time=[0, np.inf],
            left_child=[-1, 0],
            right_sib=[-1, -1],
            samples=[0],
        )
        assert isinstance(ds, xarray.Dataset)
        assert ds.sizes[core.DIM_NODE] == 2
        assert ds.sizes[core.DIM_SAMPLE] == 1
