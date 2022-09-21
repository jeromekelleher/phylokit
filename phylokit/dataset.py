import pathlib

import fsspec
import xarray as xr


def open_dataset(store, **kwargs):
    """
    Open dataset from zarr store

    :param store: zarr store
    :param kwargs: keyword arguments to pass to xarray.open_zarr
    """
    return xr.open_zarr(store, consolidated=True, concat_characters=False, **kwargs)


def save_dataset(ds, store, storage_options=None, **kwargs):
    """
    Save dataset to zarr store

    :param ds: xarray dataset
    :param store: zarr store
    :param kwargs: keyword arguments to pass to ds.to_zarr
    """
    if isinstance(store, str):
        storage_options = storage_options or {}
        store = fsspec.get_mapper(store, **storage_options)
    elif isinstance(store, pathlib.Path):
        store = str(store)
    ds.to_zarr(store, consolidated=True, **kwargs)
