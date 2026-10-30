import pathlib

import xarray as xr
import zarr


def _resolve_store(store, storage_options, read_only):
    """
    Convert a store argument into something zarr can open. zarr only applies
    storage_options to remote protocols, so when options are given we open the
    URL through fsspec ourselves, which also covers file:// and memory:// URLs.
    """
    if isinstance(store, pathlib.Path):
        store = str(store)
    if storage_options and isinstance(store, str):
        store = zarr.storage.FsspecStore.from_url(
            store, storage_options=storage_options, read_only=read_only
        )
    return store


def open_dataset(store, storage_options=None, **kwargs):
    """
    Open dataset from zarr store

    :param store: zarr store, path or fsspec URL
    :param storage_options: options passed to the fsspec filesystem
    :param kwargs: keyword arguments to pass to xarray.open_zarr
    """
    store = _resolve_store(store, storage_options, read_only=True)
    return xr.open_zarr(store, consolidated=True, concat_characters=False, **kwargs)


def save_dataset(ds, store, storage_options=None, **kwargs):
    """
    Save dataset to zarr store

    :param ds: xarray dataset
    :param store: zarr store, path or fsspec URL
    :param storage_options: options passed to the fsspec filesystem
    :param kwargs: keyword arguments to pass to ds.to_zarr
    """
    store = _resolve_store(store, storage_options, read_only=False)
    ds.to_zarr(store, consolidated=True, **kwargs)
