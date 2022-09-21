import shutil

import msprime
import pytest
import xarray as xr

import phylokit as pk


class TestCreateTreeDataset:
    def simulate_ts(self, num_samples, sequence_length, seed=1234):
        tsa = msprime.sim_ancestry(
            num_samples, sequence_length=sequence_length, ploidy=1, random_seed=seed
        )
        return msprime.sim_mutations(tsa, rate=0.03, random_seed=seed)

    def create_mutation_tree(self, num_samples, sequence_length, chunk_size, seed=1234):
        ts_in = self.simulate_ts(num_samples, sequence_length, seed=seed)
        pk_mts = pk.ts_to_dataset(ts_in, chunks=chunk_size)
        ds_in = pk.from_tskit(ts_in.first())
        ds = ds_in.merge(pk_mts)
        return ds

    @pytest.fixture
    def setup_method(self):
        self.ds = self.create_mutation_tree(100, 100, 10)
        pk.save_dataset(self.ds, "test.zarr")

    def test_save_and_open_dataset(self, setup_method):
        ds = pk.open_dataset("test.zarr")
        xr.testing.assert_identical(ds, self.ds)

    def teardown_method(self):
        shutil.rmtree("test.zarr")
