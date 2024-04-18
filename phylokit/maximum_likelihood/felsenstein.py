import numpy as np
from numba import prange

from .. import jit
from .. import util


@jit.numba_njit()
def _transition_probability(i, j, t, mu, pi=None):
    """
    Transition probability from state i to j with branch length t
    under the Jukes-Cantor model with mutation rate mu.

    $P_{i j}(t)=e^{-u t} \\delta_{i j}+\\left(1-e^{-u t}\right) \\pi_j$

    :param i: int, initial state
    :param j: int, final state
    :param t: float, branch length
    :param mu: float, mutation rate
    :param pi: np.array, transition probability matrix
    with shape (4, 4)
    :return: float, transition probability
    """
    delta_ij = 1.0 if i == j else 0.0
    return np.exp(-mu * t) * delta_ij + (np.float64(1.0) - np.exp(-mu * t)) * pi[i][j]


def _naive_calculate_likelihood(
    node,
    likelihood,
    left_child,
    right_sib,
    node_branch_length,
    rate,
    pi,
):
    for start_state in range(4):
        left_child_prob = 0
        right_sib_prob = 0
        for end_state in range(4):
            left_child_prob += likelihood[left_child][
                end_state
            ] * _transition_probability(
                start_state,
                end_state,
                node_branch_length[left_child],
                rate,
                pi,
            )
            right_sib_prob += likelihood[right_sib][
                end_state
            ] * _transition_probability(
                start_state,
                end_state,
                node_branch_length[right_sib],
                rate,
                pi,
            )
        likelihood[node][start_state] = left_child_prob * right_sib_prob


def _naive_likelihood_felsenstein(
    node_left_child,
    node_right_sib,
    call_genotype,
    num_nodes,
    traversal_postorder,
    sample_nodes,
    variant_allele,
    node_branch_length,
    rate,
    pi,
):
    ret = np.zeros(call_genotype.shape[0], dtype=np.float64)

    for i in range(call_genotype.shape[0]):
        likelihood = np.zeros((num_nodes, 4), dtype=np.float64)
        for j, sample_node in enumerate(sample_nodes):
            if call_genotype[i, j] == -1:
                likelihood[sample_node] = 0.25
            else:
                likelihood[sample_node][variant_allele[i][call_genotype[i][j][0]]] = 1.0
        for node in traversal_postorder:
            if node in sample_nodes:
                continue
            else:
                left_child = node_left_child[node]
                right_sib = node_right_sib[left_child]
                _naive_calculate_likelihood(
                    node,
                    likelihood,
                    left_child,
                    right_sib,
                    node_branch_length,
                    rate,
                    pi,
                )

        ret[i] = np.sum(likelihood[traversal_postorder[-1]] * 0.25)

    return ret


def naive_likelihood_felsenstein(
    ds,
    rate,
    pi=None,
):
    """
    Basic implementation of the likelihood function for the
    Felsenstein Likelihood calculation using the pruning algorithm.

    :param ds: phylokit.DataSet, tree data
    :param rate: float, mutation rate
    :param GENOTYPE_ARRAY: list, genotype mapping,
    :param pi: list, stationary distribution of states
    :return: float, likelihood
    """
    GENOTYPE_ARRAY = np.array([b"A", b"C", b"G", b"T"], dtype="S")

    if pi is None:
        pi = np.full((4, 4), 0.25, dtype=np.float64)

    likelihoods = _naive_likelihood_felsenstein(
        ds.node_left_child.data,
        ds.node_right_sib.data,
        ds.call_genotype.data,
        ds.nodes.shape[0],
        ds.traversal_postorder.data,
        ds.sample_node.data,
        util.base_mapping(ds.variant_allele.data, GENOTYPE_ARRAY),
        ds.node_branch_length.data,
        rate,
        pi,
    )

    ret = np.prod(likelihoods)

    return ret


@jit.numba_njit()
def _calculate_likelihood(
    child,
    node_branch_length,
    rate,
    pi,
    likelihood,
):
    probs = np.zeros(4, dtype=np.float64)
    for start_state in range(4):
        for end_state in range(4):
            probs[start_state] += likelihood[
                child, end_state
            ] * _transition_probability(
                start_state,
                end_state,
                node_branch_length[child],
                rate,
                pi,
            )
    return probs


@jit.numba_njit(parallel=True)
def _likelihood_felsenstein(
    node_left_child,
    node_right_sib,
    call_genotype,
    num_nodes,
    traversal_postorder,
    sample_nodes,
    variant_allele,
    node_branch_length,
    rate,
    pi,
):
    ret = np.zeros(call_genotype.shape[0], dtype=np.float64)

    sample_mask = np.zeros(num_nodes, dtype=np.bool_)
    sample_mask[sample_nodes] = True

    for i in prange(call_genotype.shape[0]):
        likelihood = np.zeros((num_nodes, 4), dtype=np.float64)
        for j, sample_node in enumerate(sample_nodes):
            if call_genotype[i, j] == -1:
                likelihood[sample_node] = 0.25
            else:
                likelihood[sample_node, variant_allele[i, call_genotype[i, j, 0]]] = 1.0
        for node in traversal_postorder:
            if sample_mask[node]:
                continue
            else:
                u = node_left_child[node]
                probs = _calculate_likelihood(
                    u,
                    node_branch_length,
                    rate,
                    pi,
                    likelihood,
                )
                v = node_right_sib[u]
                while v != -1:
                    probs *= _calculate_likelihood(
                        v,
                        node_branch_length,
                        rate,
                        pi,
                        likelihood,
                    )
                    v = node_right_sib[v]

                likelihood[node] = probs

        ret[i] = np.sum(likelihood[traversal_postorder[-1]] * 0.25)

    return np.prod(ret)


def likelihood_felsenstein(
    ds,
    rate,
    pi=None,
):
    """
    Calculate the likelihood of a given tree with `Felsenstein`
    pruning algorithm. This implementation is a parallelized
    version of the `naive_likelihood_felsenstein` function.

    :param ds: phylokit.DataSet, tree data
    :param rate: float, mutation rate
    :param pi: np.array, transtion probability matrix with
    shape (4, 4),
    default is [0.25, 0.25, 0.25, 0.25]
    :return: float, likelihood
    """

    GENOTYPE_ARRAY = np.array([b"A", b"C", b"G", b"T"], dtype="S")

    if pi is None:
        pi = np.full((4, 4), 0.25, dtype=np.float64)
    elif pi.shape != (4, 4):
        raise ValueError("The transition probability matrix must have shape (4, 4)")

    ret = _likelihood_felsenstein(
        ds.node_left_child.data,
        ds.node_right_sib.data,
        ds.call_genotype.data,
        ds.nodes.shape[0],
        ds.traversal_postorder.data,
        ds.sample_node.data,
        util.base_mapping(ds.variant_allele.data, GENOTYPE_ARRAY),
        ds.node_branch_length.data,
        rate,
        pi,
    )

    return ret
