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
from graphufs.inference import swap_batch_time_dims, store_container

from p1stacked import P1Emulator


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
