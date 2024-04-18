import msprime
import numpy as np
import pytest
import sgkit

import phylokit as pk
from phylokit.maximum_likelihood import felsenstein


class TestFelsensteinML:

    def ts_to_dataset(self, ts, samples=None):
        """
        Convert the specified tskit tree sequence into an sgkit dataset.
        Note this just generates haploids for now - see the note above
        in simulate_ts.
        """
        if samples is None:
            samples = ts.samples()
        tables = ts.dump_tables()
        alleles = []
        genotypes = []
        max_alleles = 0
        for var in ts.variants(samples=samples):
            alleles.append(var.alleles)
            max_alleles = max(max_alleles, len(var.alleles))
            genotypes.append(var.genotypes)
        padded_alleles = [
            list(site_alleles) + [""] * (max_alleles - len(site_alleles))
            for site_alleles in alleles
        ]
        alleles = np.array(padded_alleles).astype("S")
        genotypes = np.expand_dims(genotypes, axis=2)

        ds = sgkit.create_genotype_call_dataset(
            variant_contig_names=["1"],
            variant_contig=np.zeros(len(tables.sites), dtype=int),
            variant_position=tables.sites.position.astype(int),
            variant_allele=alleles,
            sample_id=np.array([f"tsk_{u}" for u in samples]).astype("U"),
            call_genotype=genotypes,
        )
        return ds

    def simulate_ts(self, num_samples, sequence_length, mutation_rate, seed=1234):
        tsa = msprime.sim_ancestry(
            num_samples,
            recombination_rate=0,
            sequence_length=sequence_length,
            ploidy=1,
            random_seed=seed,
        )
        return msprime.sim_mutations(tsa, mutation_rate, random_seed=seed)

    def create_mutation_tree(self, ts_in):
        pk_mts = self.ts_to_dataset(ts_in)
        ds_in = pk.from_tskit(ts_in.first())
        ds = ds_in.merge(pk_mts)
        return ds

    def test_same_base_transition_probability(self):
        assert felsenstein._transition_probability(
            0, 0, 1, 1, np.full((4, 4), 0.25, dtype=np.float64)
        ) == pytest.approx(0.5259095808785818)

    def test_diff_base_transition_probability(self):
        assert felsenstein._transition_probability(
            0, 1, 1, 1, np.full((4, 4), 0.25, dtype=np.float64)
        ) == pytest.approx(0.15803013970713942)

    def test_naive_calculate_likelihood(self):

        likelihood = np.array(
            [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float64
        )
        node_branch_length = np.array([0.035433, 0.035433, 0.0], dtype=np.float64)
        pi = np.full((4, 4), 0.25, dtype=np.float64)

        felsenstein._naive_calculate_likelihood(
            2, likelihood, 0, 1, node_branch_length, 0.005, pi
        )

        assert np.allclose(
            np.array(
                [
                    0.9997342928667493,
                    1.961379492146412e-09,
                    1.961379492146412e-09,
                    1.961379492146412e-09,
                ],
                dtype=np.float64,
            ),
            likelihood[2],
        )

    def test_felsenstein_likelihood(self):
        """
        This test is manually calculated from the tree below
               6
             ┏━┻━┓
             5   ┃
            ┏┻┓  ┃
            ┃ ┃  4
            ┃ ┃ ┏┻┓
            0 3 1 2

        Likelihoods:
            0.0011235557375395503,
            0.0011235557375395503,
            0.00018600192239752195,
            0.0011235557375395503,
        for each variant.
        """
        msprime_tree = self.simulate_ts(4, 100, 0.005, seed=1234)
        pk_mts = self.create_mutation_tree(msprime_tree)

        assert np.log(
            felsenstein.likelihood_felsenstein(pk_mts, 0.005)
        ) == pytest.approx(-28.963524119660995)

    def test_felsenstein_likelihood_no_data(self):
        """
        This test is manually calculated from the tree below
               6
             ┏━┻━┓
             5   ┃
            ┏┻┓  ┃
            ┃ ┃  4
            ┃ ┃ ┏┻┓
            0 3 1 2

        With the first variant at sample 0 having no data.
        which gives the sample 0 with a equal probability
        of 0.25 for each base

        Likelihoods:
            0.00032830706033402403,
            0.0011235557375395503,
            0.00018600192239752195,
            0.0011235557375395503,
        for each variant.
        """
        msprime_tree = self.simulate_ts(4, 100, 0.005, seed=1234)
        pk_mts = self.create_mutation_tree(msprime_tree)

        # Set the first variant at sample 0 to have no data
        pk_mts.call_genotype[0][0][0] = -1

        assert np.log(
            felsenstein.naive_likelihood_felsenstein(pk_mts, 0.005)
        ) == pytest.approx(-30.193828490667887)

        assert np.log(
            felsenstein.likelihood_felsenstein(pk_mts, 0.005)
        ) == pytest.approx(-30.193828490667887)

    @pytest.mark.parametrize("mutation_rate", [0.01, 0.02, 0.03])
    def test_felsenstein(self, mutation_rate):
        msprime_tree = self.simulate_ts(100, 100, mutation_rate, seed=1234)
        pk_tree = self.create_mutation_tree(msprime_tree)
        assert felsenstein.naive_likelihood_felsenstein(
            pk_tree, mutation_rate
        ) == pytest.approx(felsenstein.likelihood_felsenstein(pk_tree, mutation_rate))

    def test_felsenstein_error(self):
        msprime_tree = self.simulate_ts(100, 100, 0.01, seed=1234)
        pk_tree = self.create_mutation_tree(msprime_tree)

        with pytest.raises(ValueError):
            felsenstein.likelihood_felsenstein(
                pk_tree,
                0.01,
                np.full((4, 3), 0.25, dtype=np.float64),
            )
