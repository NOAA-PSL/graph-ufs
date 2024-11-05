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
from graphufs.fvemulator import fv_vertical_regrid

from config import P2EvaluationEmulator as Emulator

def open_predictions_and_truth(emulator):

    duration = emulator.target_lead_time[-1]

    # open graphufs, and targets as is
    gds = xr.open_zarr(f"{emulator.local_store_path}/inference/validation/graphufs.{duration}.zarr")

    # add vertical coordinate stuff
    nds = xr.open_zarr(emulator.norm_urls["mean"], storage_options={"token": "anon"})
    gds["ak"] = nds["ak"]
    gds["bk"] = nds["bk"]
    gds = gds.set_coords(["ak", "bk"])

    # add static hgtsfc for geopotential
    if "hgtsfc_static" not in gds:
        gds["hgtsfc_static"] = emulator.open_dataset()["hgtsfc_static"].rename({
            "grid_yt": "lat",
            "grid_xt": "lon",
        }).load()

    # for 3h dt, subsample to 6h
    if duration == "240h" and len(gds.lead_time) == 80:
        gds = gds.isel(lead_time=slice(1, None, 2))

        logging.info("Subsampled lead_time to 6h")
        logging.info(f"New time: {gds.lead_time.values}")

    truth = xr.open_zarr(emulator.wb2_obs_url, storage_options={"token":"anon"})

    # subsample in space to avoid poles
    if "with_poles" in emulator.wb2_obs_url:
        truth = truth.sel(latitude=slice(-89, 89))

    # subsample in time, based on truth we can compare to
    t0 = get_valid_initial_conditions(gds, truth)
    gds = gds.sel(time=t0)
    return gds, truth

def open_targets(emulator, predictions):
    rds = emulator.open_dataset()
    keep_vars = list(predictions.keys()) + ["geopotential", "delz"]
    rds = rds[keep_vars]
    valid_time = predictions["time"] + predictions["lead_time"]
    rds = rds.sel(time=slice(predictions.time.values[0], valid_time.isel(time=-1, lead_time=-1).values))
    time = get_valid_initial_conditions(rds, truth)
    rds = rds.sel(time=time)

    # Now FV vertical
    rds = fv_vertical_regrid(rds, interfaces=list(emulator.interfaces))
    rds = rds.rename({"pfull": "level", "grid_xt": "lon", "grid_yt": "lat"})

    # Store this as is
    replay_path = f"{emulator.local_store_path}/inference/validation/replay.vertical_regrid.zarr"
    logging.info(f"Writing vertically regridded Replay to {replay_path} ...")
    rds.to_zarr(replay_path)
    logging.info(f"Done writing Replay")
    return rds

def main(xds, truth, name, plevels=(250, 500, 850)):

    pds = interp2pressure(
        xds,
        plevels,
        diagnose_geopotential= name!="replay",
    )
    logging.info(f"Done forming interpolation operations...")

    # regrid and rename variables
    pds = regrid_and_rename(pds, truth)
    logging.info(f"Done forming regridding operations...")

    path = f"{emulator.local_store_path}/inference/validation/{name}.{duration}.postprocessed.zarr"
    pds.to_zarr(path)
    logging.info(f"Done writing to {path}")


if __name__ == "__main__":

    setup_simple_log()
    emulator = Emulator()
    dask.config.set(scheduler="threads", num_workers=16)

    duration = emulator.target_lead_time[-1]
    logging.info("Opening Predictions")
    gds, truth = open_predictions_and_truth(emulator)

    logging.info("Postprocessing Predictions")
    main(gds, truth, name="graphufs")
    logging.info("Done postprocessing predictions")

    logging.info("Opening Replay")
    rds = open_targets(emulator, gds)

    logging.info("Postprocessing Replay")
    main(rds, truth, name="replay")

    logging.info("Done with all postprocessing")
