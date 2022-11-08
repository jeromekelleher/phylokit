import numba
import numpy as np
import sgkit
import xarray as xr
from numba import int32


@numba.njit()
def _hartigan_preorder_vectorised(parent, optimal_set, left_child, right_sib):
    num_sites, num_alleles = optimal_set.shape[1:]

    allele_count = np.zeros((num_sites, num_alleles), dtype=np.int32)
    child = left_child[parent]
    while child != -1:
        _hartigan_preorder_vectorised(child, optimal_set, left_child, right_sib)
        allele_count += optimal_set[child]
        child = right_sib[child]

    if left_child[parent] != -1:
        for j in range(num_sites):
            site_allele_count = allele_count[j]
            # max_allele_count = np.max(site_allele_count)
            max_allele_count = 0
            for k in range(num_alleles):
                if site_allele_count[k] > max_allele_count:
                    max_allele_count = site_allele_count[k]
            for k in range(num_alleles):
                if site_allele_count[k] == max_allele_count:
                    optimal_set[parent, j, k] = 1


@numba.njit()
def _hartigan_postorder_vectorised(node, state, optimal_set, left_child, right_sib):
    num_sites, num_alleles = optimal_set.shape[1:]

    mutations = np.zeros(num_sites, dtype=np.int32)
    # Strictly speaking we only need to do this if we mutate it. Might be worth
    # keeping track of - but then that would complicate the inner loop, which
    # could hurt vectorisation/pipelining/etc.
    state = state.copy()
    for j in range(num_sites):
        site_optimal_set = optimal_set[node, j]
        if site_optimal_set[state[j]] == 0:
            # state[j] = np.argmax(site_optimal_set)
            maxval = -1
            argmax = -1
            for k in range(num_alleles):
                if site_optimal_set[k] > maxval:
                    maxval = site_optimal_set[k]
                    argmax = k
            state[j] = argmax
            mutations[j] = 1

    v = left_child[node]
    while v != -1:
        v_muts = _hartigan_postorder_vectorised(
            v, state, optimal_set, left_child, right_sib
        )
        mutations += v_muts
        v = right_sib[v]
    return mutations


@numba.jit()
def _hartigan_initialise_vectorised(optimal_set, genotypes, samples):
    for k, site_genotypes in enumerate(genotypes):
        for j, u in enumerate(samples):
            optimal_set[u, k, site_genotypes[j]] = 1


@numba.guvectorize(
    [(int32[:], int32[:], int32[:], int32[:, :], int32[:])],
    "(n),(n),(s),(v, s) -> (v)",
)
def numba_hartigan_parsimony_vectorised(
    left_child, right_sib, samples, genotypes, score
):
    """
    Calculate the parsimony score for each site in the dataset.

    NOTE: The dimensions for each of the parameters should be fixed as follows:
          (n),(n),(s),(v, s) -> (v)
          where:
            n = number of nodes
            s = number of samples
            v = number of variants

    .. seealso::
        See `Numba docs <https://numba.pydata.org/numba-doc/dev/user/vectorize.html>`_
        for more details.

    :param left_child: (dim: n) The left child of each node.
    :param right_sib: (dim: n) The right sibling of each node.
    :param samples: (dim: s) The samples in the tree.
    :param genotypes: (dim: v, s) The genotype of each sample at each site.
    :param score: (dim: v) The parsimony score for each site.
    """

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_sites = genotypes.shape[0]
    num_nodes = left_child.shape[0] - 1

    optimal_set = np.zeros((num_nodes, num_sites, num_alleles), dtype=np.int8)
    _hartigan_initialise_vectorised(optimal_set, genotypes, samples)
    _hartigan_preorder_vectorised(-1, optimal_set, left_child, right_sib)
    ancestral_state = np.argmax(optimal_set[-1], axis=1)
    score[:] = _hartigan_postorder_vectorised(
        -1, ancestral_state, optimal_set, left_child, right_sib
    )


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
        ds = ds.chunk({"variants": chunks})
    return ds


def get_hartigan_parsimony_score(ds):
    """
    Calculate the parsimony score for each site in the dataset.

    :param ds: The dataset to calculate the parsimony score for.
    :return: The parsimony score for each site in the dataset.
    :rtype: xarray.DataArray
    """
    return xr.apply_ufunc(
        numba_hartigan_parsimony_vectorised,
        ds.node_left_child,
        ds.node_right_sib,
        ds.sample_node,
        ds.call_genotype,
        input_core_dims=[
            ["nodes"],
            ["nodes"],
            ["samples"],
            ["samples"],
        ],
        dask="parallelized",
        output_dtypes=[np.int32],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


def append_parsimony_score(ds):
    """
    Append the parsimony score to the dataset.

    :param ds: The dataset to append the parsimony score to.
    :return: The dataset with the parsimony score appended.
    :rtype: xarray.Dataset
    """
    ds["sites_parsimony_score"] = get_hartigan_parsimony_score(ds)
    return ds
