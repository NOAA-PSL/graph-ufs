import logging
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr
import pandas as pd

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

from p1stacked import P1Emulator
from graphufs.log import setup_simple_log
from graphufs.postprocess import interp2pressure, regrid_and_rename, get_valid_initial_conditions

def open_datasets(t0, duration, truth_url):
    """
    Returns:
        predictions, independent_truth_dataset
    """
    # open graphufs, and targets as is
    gds = xr.open_zarr(f"/p1-evaluation/v1/long-forecasts/graphufs.{t0}.{duration}.zarr")
    truth = xr.open_zarr(truth_url, storage_options={"token":"anon"})

    # subsample in space to avoid poles
    if "with_poles" in truth_url:
        truth = truth.sel(latitude=slice(-89, 89))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)
    return gds, truth


if __name__ == "__main__":

    setup_simple_log()
    dask.config.set(scheduler="threads", num_workers=48)

    truth_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    t0 = "2019-01-01T00"
    duration = "8754h"
    gds, truth = open_datasets(t0, duration, truth_url)

    for ds, name in zip(
        [gds],
        ["graphufs"],
    ):
        ds = interp2pressure(ds, [100, 500, 850])
        logging.info(f"Interpolated to pressure levels...")

        # regrid and rename variables
        ds = regrid_and_rename(ds, truth)
        logging.info(f"Done regridding...")

        path = f"/p1-evaluation/v1/long-forecasts/{name}.{t0}.{duration}.postprocessed.zarr"
        ds.to_zarr(path)
        logging.info(f"Done writing to {path}")
