import numpy as np
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
import sgkit

from . import core


def linkage_matrix_to_dataset(Z):
    """
    Scipy's hierarchy methods return a (n-1, 4) linkage matrix describing the clustering
    of the n observations. Each row in this matrix corresponds to an internal node
    in the tree.
    """
    n = Z.shape[0] + 1
    N = 2 * n
    parent = np.full(N, -1, dtype=np.int32)
    time = np.full(N, 0, dtype=np.float64)
    left_child = np.full(N, -1, dtype=np.int32)
    right_sib = np.full(N, -1, dtype=np.int32)
    # TODO Make a jitted function to do this
    for j, row in enumerate(Z):
        u = n + j
        time[u] = j + 1
        lc = int(row[0])
        rc = int(row[1])
        parent[lc] = u
        parent[rc] = u
        left_child[u] = lc
        right_sib[lc] = rc
    left_child[-1] = N - 2
    time[-1] = np.inf
    return core.create_tree_dataset(
        parent, time, left_child, right_sib, np.arange(n, dtype=np.int32)
    )


# This is a bad name - we should do something to group distance matrix
# based methods together and give a method argument to decide what the
# method is.
def upgma(ds):
    # TODO do something more sensible later with ploidy
    assert ds.sizes["ploidy"] == 1
    ds = ds.squeeze("ploidy")
    # TODO add option to keep this distance matrix, like we do in sgkit for expensive
    # intermediate calculations.
    # TODO not sure if this is doing anything sensible!
    D = sgkit.pairwise_distance(ds.call_genotype.T)
    D = D.compute()
    # Convert to condensed vector form.
    v = dist.squareform(D)
    # This performs UPGMA clustering
    Z = hier.average(v)
    ds_tree = linkage_matrix_to_dataset(Z)
    # TODO add options to merge back into the original like we do in sgkit,
    # in conditional_merge_dataset
    return ds_tree
