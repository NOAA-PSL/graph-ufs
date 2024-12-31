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
from graphufs.log import setup_simple_log

from config import P2EvaluationEmulator as Emulator

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
    pname = f"{emulator.local_store_path}/inference/{batchloader.dataset.mode}/graphufs.{hours}h.zarr"
    tname = f"{emulator.local_store_path}/inference/{batchloader.dataset.mode}/replay.{hours}h.zarr"

    n_steps = len(batchloader)
    progress_bar = tqdm(total=n_steps, ncols=80, desc="Processing")
    for k, (inputs, targets, forcings) in enumerate(batchloader):

        # retrieve and drop t0
        inittimes = inputs.datetime.isel(time=-1).values
        inputs = inputs.drop_vars("datetime")
        targets = targets.drop_vars("datetime")
        forcings = forcings.drop_vars("datetime")

        # predictions have dims [batch, time (aka forecast_time), level, lat, lon]
        predictions = rollout.chunked_prediction(
            gc,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=np.nan * targets,
            forcings=forcings,
        )

        # perform output transform, e.g. exp( log_spfh )
        for key, mapping in emulator.output_transforms.items():
            with xr.set_options(keep_attrs=True):
                predictions[key] = mapping(predictions[key])

        # subsample to 6 hours
        predictions = predictions.isel(time=slice(1, None, 2))

        # Add t0 as new variable, and swap out for logical sample/batch index
        # swap dims to be [time (aka initial condition time), lead_time (aka forecast_time), level, lat, lon]
        predictions = swap_batch_time_dims(predictions, inittimes)

        # Store to zarr one batch at a time
        if k == 0:
            store_container(pname, predictions, time=batchloader.initial_times)

        # Store to zarr
        spatial_region = {k: slice(None, None) for k in predictions.dims if k != "time"}
        region = {
            "time": slice(k*batchloader.batch_size, (k+1)*batchloader.batch_size),
            **spatial_region,
        }
        predictions.to_zarr(pname, region=region)

        progress_bar.update()


if __name__ == "__main__":

    setup_simple_log()
    emulator = Emulator()
    init_devices(emulator)
    dask.config.set(scheduler="threads", num_workers=emulator.dask_threads)

    vds = Dataset(
        emulator,
        mode="validation",
        preload_batch=False,
    )

    validator = ExpandedBatchLoader(
        vds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        max_queue_size=1,
        sample_stride=emulator.sample_stride,
    )

    # setup weights
    logging.info(f"Reading weights ...")
    # TODO: this looks in local_store_path, but we may want to look somewhere else
    ckpt_id = emulator.evaluation_checkpoint_id if emulator.evaluation_checkpoint_id is not None else emulator.num_epochs
    params, state = emulator.load_checkpoint(id=ckpt_id)

    predict(
        params=params,
        state=state,
        emulator=emulator,
        batchloader=validator,
    )

    validator.shutdown()
