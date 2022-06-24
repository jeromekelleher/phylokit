# Overview

Phylokit is a library of operations for phylogenetic trees using the PyData Ecosystem.
It is based on a simple numerical encoding of topologies used in
[tskit](https://tskit.dev/tskit/docs/stable/data-model.html#tree-structure) where
tree information is represented by a set of arrays. This encoding has a number
of advantages over the standard in-memory description of trees as a set of
linked objects:

- We can use [array oriented computing](https://en.wikipedia.org/wiki/Array_programming) to work
  efficiently with large trees using NumPy
- We can use [numba](https://numba.pydata.org) to compile tree algorithms written in
  Python to fast machine code (including targetting GPUs)
- We can use other parts of the PyData ecosystem such as [xarray](https://docs.xarray.dev/en/stable/)
  and [Dask](https://www.dask.org) to scale

# Design principles

## Loosely coupled to tskit

Although the main source of data for phylokit input will initially be the ``tskit.Tree`` class,
we should make ``phylokit`` as **loosely coupled** as possible to tskit. In practise, this means
that we should assume the smallest possible number of attributes and methods. The minimum that we
need are:

- ``left_child_array``
- ``right_sib_array``
- ``time_array``

We assume the existence of a [virtual root](https://tskit.dev/tskit/docs/stable/data-model.html#the-virtual-root)
like tskit, so that the ``left_child_array`` is one element longer than ``time_array``, and
such that ``left_child_array[-1]`` is the left-most root.

This will means some duplication of functionality between tskit and phylokit for fundamental
operations like ``mrca``, but this is a reasonable tradeoff for long-term flexibility.

The rationale behind minimising the dependence on tskit is to allow more flexible internal
use of the key data structures than building directly on tskit would allow (for example, when
we are inferring trees), and also to hopefully allow other applications to build on this
foundation also. By using the array interface, we open up the possibility of using anything
that implements the numpy array interface, (like Zarr arrays, e.g.) as the underlying
data storage for trees.

## Use PyData upstreams where possible

Don't reinvent the wheel. For example, when we start dealing with input datasets, we use
[sgkit](https://pystatgen.github.io/sgkit/latest/) as the starting point. (Although this may not
be suitable for some types of alignment input, and then we must look at using Xarray directly).
