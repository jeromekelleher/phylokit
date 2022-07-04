import msprime
import numpy as np
import sgkit

import phylokit as pk
from phylokit import inference


# Borrowed from sgkit's test_popgen file.
def ts_to_dataset(ts, chunks=None, samples=None):
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
    if chunks is not None:
        ds = ds.chunk(dict(zip(["variants", "samples"], chunks)))
    return ds


def simulate_ts(num_samples, num_sites, seed=1234):
    tsa = msprime.sim_ancestry(
        num_samples, sequence_length=num_sites, ploidy=1, random_seed=seed
    )
    return msprime.sim_mutations(tsa, rate=0.1, random_seed=seed)


def test_things():
    ts_in = simulate_ts(8, 40)
    ds = ts_to_dataset(ts_in)
    ds_tree = inference.upgma(ds)
    ds_merged = ds_tree.merge(ds)
    ts_out = pk.to_tskit(ds_merged)
    # print(ds_merged)
    # print(ts_in.draw_text())
    # print(ds_merged)
    assert ts_in.num_samples == ts_out.tree_sequence.num_samples
    # TODO figure out how to test some things
