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

from graphcast import rollout

from graphufs import init_devices, construct_wrapped_graphcast
from graphufs.batchloader import ExpandedBatchLoader
from graphufs.datasets import Dataset

from configeval import LatentTestEmulator

def swap_batch_time_dims(xds, inittimes):

    xds = xds.rename({"time": "lead_time"})

    # create "time" dimension = t0
    xds["time"] = xr.DataArray(
        inittimes,
        coords=xds["batch"].coords,
        dims=xds["batch"].dims,
        attrs={
            "description": "Forecast initialization time, last timestep of initial conditions",
        },
    )

    # swap logical batch for t0
    xds = xds.swap_dims({"batch": "time"}).drop_vars("batch")
    return xds


def store_container(path, xds, time, **kwargs):

    if "time" in xds:
        xds = xds.isel(time=0, drop=True)

    container = xr.Dataset()
    for key in xds.coords:
        container[key] = xds[key].copy()

    for key in xds.data_vars:
        dims = ("time",) + xds[key].dims
        coords = {"time": time, **dict(xds[key].coords)}
        shape = (len(time),) + xds[key].shape
        chunks = (1,) + tuple(-1 for _ in xds[key].dims)

        container[key] = xr.DataArray(
            data=dask.array.zeros(
                shape=shape,
                chunks=chunks,
                dtype=xds[key].dtype,
            ),
            coords=coords,
            dims=dims,
            attrs=xds[key].attrs.copy(),
        )
    container.to_zarr(path, compute=False, **kwargs)
    logging.info(f"Stored container at {path}")

def predict(
    params,
    state,
    emulator,
    batchloader,
) -> xr.Dataset:

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    gc = drop_state(with_params(jax.jit(run_forward.apply)))

    hours = int(emulator.forecast_duration.value / 1e9 / 3600)
    pname = f"{emulator.local_store_path}/evaluation/{batchloader.mode}/graphufs.{hours}h.zarr"
    tname = f"{emulator.local_store_path}/evaluation/{batchloader.mode}/replay.{hours}h.zarr"

    for storename in [pname, tname]:
        thisdir = os.path.dirname(storename)
        if not os.path.isdir(thisdir):
            os.makedirs(thisdir)

    n_steps = len(batchloader)
    progress_bar = tqdm(total=n_steps, ncols=80, desc="Processing")
    for k, (inputs, targets, forcings) in enumerate(batchloader):

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
    p1, args = LatentTestEmulator.from_parser()
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
        sample_stride=p1.sample_stride,
    )

    # setup weights
    logging.info(f"Reading weights ...")
    # TODO: this looks in local_store_path, but we may want to look somewhere else
    ckpt_id = p1.evaluation_checkpoint_id if p1.evaluation_checkpoint_id is not None else p1.num_epochs
    params, state = p1.load_checkpoint(id=ckpt_id)

    predict(
        params=params,
        state=state,
        emulator=p1,
        batchloader=validator,
    )
    validator.shutdown()
