import msprime
import numpy as np
import pytest
import xarray.testing as xt

import phylokit as pk


def simulate_ts(num_samples, num_sites, seed=1234):
    tsa = msprime.sim_ancestry(
        num_samples, sequence_length=num_sites, ploidy=1, random_seed=seed
    )
    return msprime.sim_mutations(tsa, rate=0.01, random_seed=seed)


class Test_Hartigan_Parsimony_Vectorised:
    def generate_test_tree_ts():
        trees = []
        for i in range(1, 6):
            trees.append(simulate_ts(1000, 100, seed=i * 88))
        return trees

    @pytest.mark.parametrize("chunk_size", [1, 2, 4, 8])
    @pytest.mark.parametrize("ts_in", generate_test_tree_ts())
    def test_ts(self, ts_in, chunk_size):
        pk_mts = pk.parsimony.hartigan.ts_to_dataset(ts_in, chunk_size)
        ds_in = pk.from_tskit(ts_in.first())
        ds = ds_in.merge(pk_mts)
        tree = ts_in.first()
        parsimony_score = []
        for var in ts_in.variants():
            tree.seek(var.site.position)
            _, mutations = tree.map_mutations(var.genotypes, var.alleles)
            parsimony_score.append(len(mutations))
        assert chunk_size == ds.call_genotype.chunks[0][0]
        np.testing.assert_array_equal(
            np.array(parsimony_score),
            pk.get_hartigan_parsimony_score(ds).compute().squeeze("ploidy"),
        )

    @pytest.mark.parametrize("ts_in", generate_test_tree_ts())
    def test_append_parsimony_score(self, ts_in):
        pk_mts = pk.parsimony.hartigan.ts_to_dataset(ts_in)
        ds_in = pk.from_tskit(ts_in.first())
        ds = ds_in.merge(pk_mts)
        _ds = ds.copy()
        tree = ts_in.first()
        parsimony_score = []
        for var in ts_in.variants():
            tree.seek(var.site.position)
            _, mutations = tree.map_mutations(var.genotypes, var.alleles)
            parsimony_score.append(len(mutations))
        _ds["sites_parsimony_score"] = ("variants",), parsimony_score
        xt.assert_equal(
            pk.append_parsimony_score(ds).compute().squeeze("ploidy"),
            _ds.squeeze("ploidy"),
        )
