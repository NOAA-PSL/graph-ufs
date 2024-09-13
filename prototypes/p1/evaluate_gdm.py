from functools import partial
import logging
import os
import sys
import jax
import haiku as hk
import numpy as np
import dask
import xarray as xr
from tqdm import tqdm
import gcsfs

from graphcast import rollout, checkpoint

from graphufs import init_devices, construct_wrapped_graphcast
from graphufs.batchloader import ExpandedBatchLoader
from graphufs.datasets import Dataset

from p1gdm import P1Emulator
from evaluate import swap_batch_time_dims, store_container
from long_forecast import load_checkpoint, run_forward, get_all_variables

_norm_urls = {
    "mean": "gs://gdm-noaa-ufs-2024/model_01/stats/mean_by_level.nc",
    "std": "gs://gdm-noaa-ufs-2024/model_01/stats/stddev_by_level.nc",
    "stddiff": "gs://gdm-noaa-ufs-2024/model_01/stats/diffs_stddev_by_level.nc",
}


def load_normalization(task_config):
    """Assumes we want normalization from the 1/4 degree subsampled to ~1 degree data"""

    normalization = dict()
    fs = gcsfs.GCSFileSystem()
    for key in ["mean", "std", "stddiff"]:
        with fs.open(_norm_urls[key], "rb") as f:
            this_norm = xr.load_dataset(f)
        this_norm = this_norm[[x for x in get_all_variables(task_config) if x in this_norm]]
        this_norm = this_norm.sel(level=list(task_config.pressure_levels), method="nearest")
        normalization[key] = this_norm

    # Postprocess normalization stats to handle zero/NaN stddevs
    eps_scale = 1e-6
    for key in ["std", "stddiff"]:
        normalization[key] = normalization[key].fillna(1.)
        normalization[key] = normalization[key].where(normalization[key] > eps_scale, eps_scale)

    return normalization

def predict(
    params,
    state,
    model_config,
    task_config,
    batchloader,
    normalization,
) -> xr.Dataset:

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    def with_configs(fn):
        return partial(
            fn,
            model_config=model_config,
            task_config=task_config,
            normalization=normalization,
        )

    gc = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    hours = int(batchloader.dataset.emulator.forecast_duration.value / 1e9 / 3600)
    pname = f"/p1-evaluation/gdm-v1/{batchloader.dataset.mode}/graphufs_gdm.{hours}h.zarr"
    tname = f"/p1-evaluation/gdm-v1/{batchloader.dataset.mode}/replay_gdm.{hours}h.zarr"

    n_steps = len(batchloader)
    progress_bar = tqdm(total=n_steps, ncols=80, desc="Processing")
    for k in range(n_steps):
        inputs, targets, forcings = batchloader.get_data()

        # retrieve and drop t0
        inittimes = inputs.datetime.isel(time=-1).values
        inputs = inputs.drop_vars("datetime")
        targets = targets.drop_vars("datetime")
        forcings = forcings.drop_vars("datetime")

        predictions = rollout.chunked_prediction(
            gc,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=np.nan * targets,
            forcings=forcings,
        )

        # Add t0 as new variable, and swap out for logical sample/batch index
        predictions = swap_batch_time_dims(predictions, inittimes)
        targets = swap_batch_time_dims(targets, inittimes)

        # Store to zarr one batch at a time
        if k == 0:
            store_container(pname, predictions, time=batchloader.initial_times)
            store_container(tname, targets, time=batchloader.initial_times)

        # Store to zarr
        spatial_region = {k: slice(None, None) for k in predictions.dims if k != "time"}
        region = {
            "time": slice(k*batchloader.batch_size, (k+1)*batchloader.batch_size),
            **spatial_region,
        }
        predictions.to_zarr(pname, region=region)
        targets.to_zarr(tname, region=region)

        progress_bar.update()


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    p1, args = P1Emulator.from_parser()
    init_devices(p1)
    dask.config.set(scheduler="threads", num_workers=p1.dask_threads)

    vds = Dataset(
        p1,
        mode="validation",
        preload_batch=False,
    )

    validator = ExpandedBatchLoader(
        vds,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        max_queue_size=1,
        sample_stride=9,
    )

    # setup weights
    logging.info(f"Reading weights ...")
    params, model_config, task_config = load_checkpoint("results/gdm-v1/model_01.npz")
    state = dict()

    normalization = load_normalization(task_config)

    predict(
        params=params,
        state=state,
        model_config=model_config,
        task_config=task_config,
        batchloader=validator,
        normalization=normalization,
    )
