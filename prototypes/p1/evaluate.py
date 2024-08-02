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

from p1stacked import P1Emulator

def swap_batch_time_dims(predictions, targets, inittimes):

    predictions = predictions.rename({"time": "lead_time"})
    targets = targets.rename({"time": "lead_time"})

    # create "time" dimension = t0
    predictions["time"] = xr.DataArray(
        inittimes,
        coords=predictions["batch"].coords,
        dims=predictions["batch"].dims,
        attrs={
            "description": "Forecast initialization time, last timestep of initial conditions",
        },
    )

    targets["time"] = xr.DataArray(
        inittimes,
        coords=targets["batch"].coords,
        dims=targets["batch"].dims,
        attrs={
            "description": "Forecast initialization time, last timestep of initial conditions",
        },
    )

    # swap logical batch for t0
    predictions = predictions.swap_dims({"batch": "time"}).drop_vars("batch")
    targets = targets.swap_dims({"batch": "time"}).drop_vars("batch")

    return predictions, targets


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
    pname = f"/p1-evaluation/v1/{batchloader.dataset.mode}/graphufs.{hours}h.zarr"
    tname = f"/p1-evaluation/v1/{batchloader.dataset.mode}/replay.{hours}h.zarr"

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
        predictions, targets = swap_batch_time_dims(predictions, targets, inittimes)

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
        batch_size=p1.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=p1.num_workers,
        max_queue_size=p1.max_queue_size,
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
