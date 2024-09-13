import logging
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr
import pandas as pd

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

from config import P1Emulator
from graphufs.log import setup_simple_log
from graphufs.postprocess import interp2pressure, regrid_and_rename, get_valid_initial_conditions

def open_datasets(emulator):
    """
    Returns:
        predictions, targets, original_target_dataset, independent_truth_dataset
    """

    duration = emulator.target_lead_time[-1]

    # open graphufs, and targets as is
    gds = xr.open_zarr(f"/gdm-eval/v1/validation/graphufs_gdm.{duration}.zarr")
    truth = xr.open_zarr(emulator.wb2_obs_url, storage_options={"token":"anon"})

    # subsample in space to avoid poles
    if "with_poles" in emulator.wb2_obs_url:
        truth = truth.sel(latitude=slice(-89, 89))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)

    return gds, truth


if __name__ == "__main__":

    setup_simple_log()
    p1, args = P1Emulator.from_parser()
    dask.config.set(scheduler="threads", num_workers=p1.dask_threads)

    duration = p1.target_lead_time[-1]
    gds, truth = open_datasets(p1)

    for ds, name in zip(
        [gds],
        ["graphufs_gdm"],
    ):
        ds = interp2pressure(ds, [100, 500, 850])
        logging.info(f"Interpolated to pressure levels...")

        # regrid and rename variables
        ds = regrid_and_rename(ds, truth)
        logging.info(f"Done regridding...")

        path = f"/gdm-eval/v1/validation/{name}.{duration}.postprocessed.zarr"
        ds.to_zarr(path)
        logging.info(f"Done writing to {path}")
