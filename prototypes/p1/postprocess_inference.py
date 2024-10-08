import logging
import numpy as np
import dask
import xarray as xr
import xesmf
import cf_xarray as cfxr
import pandas as pd

from ufs2arco.regrid.gaussian_grid import gaussian_latitudes
from ufs2arco import Layers2Pressure

from graphufs.log import setup_simple_log
from graphufs.postprocess import interp2pressure, regrid_and_rename, get_valid_initial_conditions

from p1stacked import P1Emulator

def open_datasets(emulator):
    """
    Returns:
        predictions, targets, original_target_dataset, independent_truth_dataset
    """

    duration = emulator.target_lead_time[-1]

    # open graphufs, and targets as is
    gds = xr.open_zarr(f"/p1-evaluation/v1/validation/graphufs.{duration}.zarr")
    tds = xr.open_zarr(f"/p1-evaluation/v1/validation/replay.{duration}.zarr")
    truth = xr.open_zarr(emulator.wb2_obs_url, storage_options={"token":"anon"})

    # for 3h dt, subsample to 6h
    if duration == "240h" and len(gds.lead_time) == 80:
        gds = gds.isel(lead_time=slice(1, None, 2))
        tds = tds.isel(lead_time=slice(1, None, 2))

        logging.info("Subsampled lead_time to 6h")
        logging.info(f"New time: {gds.lead_time.values}")

    # subsample in space to avoid poles
    if "with_poles" in emulator.wb2_obs_url:
        truth = truth.sel(latitude=slice(-89, 89))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)
    tds = tds.sel(time=t0)

    # for replay we also want to get the original data, in order to avoid
    # excessive interpolation error due to subsampled vertical levels
    rds = emulator.open_dataset()
    keep_vars = list(gds.keys())
    rds = rds[keep_vars]
    valid_time = gds["time"] + gds["lead_time"]
    rds = rds.sel(time=slice(gds.time.values[0], valid_time.isel(time=-1, lead_time=-1).values))
    rds = rds.rename({"pfull": "level", "grid_xt": "lon", "grid_yt": "lat"})
    time = get_valid_initial_conditions(rds, truth)
    rds = rds.sel(time=time)
    return gds, tds, rds, truth


if __name__ == "__main__":

    setup_simple_log()
    p1, args = P1Emulator.from_parser()
    dask.config.set(scheduler="threads", num_workers=p1.dask_threads)

    duration = p1.target_lead_time[-1]
    gds, tds, rds, truth = open_datasets(p1)

    for ds, name in zip(
        [gds, tds, rds],
        ["graphufs", "replay_targets", "replay"],
    ):
        ds = interp2pressure(ds, [100, 500, 850])
        logging.info(f"Interpolated to pressure levels...")

        # regrid and rename variables
        ds = regrid_and_rename(ds, truth)
        logging.info(f"Done regridding...")

        path = f"/p1-evaluation/v1/validation/{name}.{duration}.postprocessed.zarr"
        ds.to_zarr(path)
        logging.info(f"Done writing to {path}")
