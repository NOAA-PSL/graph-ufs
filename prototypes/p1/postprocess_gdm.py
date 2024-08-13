import logging
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr
import pandas as pd

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

from p1gdm import P1Emulator
from stacked_preprocess import setup_log
from postprocess_evaluation import interp2pressure, regrid_and_rename, get_valid_initial_conditions

def open_datasets(emulator):
    """
    Returns:
        predictions, targets, original_target_dataset, independent_truth_dataset
    """

    duration = emulator.target_lead_time[-1]

    # open graphufs, and targets as is
    gds = xr.open_zarr(f"/p1-evaluation/gdm-v1/validation/graphufs.{duration}.zarr")
    truth = xr.open_zarr(emulator.wb2_obs_url, storage_options={"token":"anon"})

    # subsample in space to avoid poles
    if "with_poles" in emulator.wb2_obs_url:
        truth = truth.sel(latitude=slice(-89, 89))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)

    ## for replay we also want to get the original data, in order to avoid
    ## excessive interpolation error due to subsampled vertical levels
    #rds = emulator.open_dataset()
    #keep_vars = list(gds.keys())
    #rds = rds[keep_vars]
    #valid_time = gds["time"] + gds["lead_time"]
    #rds = rds.sel(time=slice(gds.time.values[0], valid_time.isel(time=-1, lead_time=-1).values))
    #rds = rds.rename({"pfull": "level", "grid_xt": "lon", "grid_yt": "lat"})
    #time = get_valid_initial_conditions(rds, truth)
    #rds = rds.sel(time=time)
    return gds, truth


if __name__ == "__main__":

    setup_log()
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

        path = f"/p1-evaluation/gdm-v1/validation/{name}.{duration}.postprocessed.zarr"
        ds.to_zarr(path)
        logging.info(f"Done writing to {path}")
