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

from graphufs.training import construct_wrapped_graphcast
from graphufs.batchloader import MPIExpandedBatchLoader
from graphufs.datasets import Dataset
from graphufs.inference import swap_batch_time_dims, store_container
from graphufs.mpi import MPITopology
from graphufs import diagnostics
from graphufs import fvemulator

def predict(
    params,
    state,
    emulator,
    batchloader,
    mpi_topo,
) -> xr.Dataset:

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    logging.info("JIT Compiling Predict")
    gc = drop_state(with_params(jax.jit(run_forward.apply)))
    logging.info("Done Compiling Predict")

    hours = int(emulator.forecast_duration.value / 1e9 / 3600)
    pname = f"{emulator.local_store_path}/inference/{batchloader.dataset.mode}/graphufs.{hours}h.zarr"
    tname = f"{emulator.local_store_path}/inference/{batchloader.dataset.mode}/replay.{hours}h.zarr"

    # prepare diagnostics if desired
    diagnostic_mappings = dict()
    if emulator.diagnostics is not None:
        diagnostic_mappings = diagnostics.prepare_diagnostic_functions(emulator.diagnostics)


    n_steps = len(batchloader)
    with open(mpi_topo.progress_file, "a") as f:
        progress_bar = tqdm(total=n_steps, ncols=80, desc="Processing", file=f)
        for k, (inputs, targets, forcings) in enumerate(batchloader):

            if inputs is not None:
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
                        targets[key] = mapping(targets[key])

                # subsample to 6 hours
                predictions = predictions.isel(time=slice(1, None, 2))
                targets = targets.isel(time=slice(1, None, 2))

                # compute diagnostics if desired
                for key, func in diagnostic_mappings["functions"].items():
                    if "ak" not in predictions.data_vars or "ak" not in predictions.coords:
                        cds = fvemulator.get_new_vertical_grid(list(emulator.interfaces))
                        for k2 in ["ak", "bk"]:
                            predictions[k2] = cds[k2]
                            targets[k2] = cds[k2]
                            predictions = predictions.set_coords(k2)
                            targets = targets.set_coords(k2)

                    for var in diagnostic_mappings["required_variables"][key]:
                        if var not in predictions.data_vars:
                            if var in inputs:
                                predictions[var] = inputs[var]
                                targets[var] = inputs[var]
                            else:
                                raise KeyError(f"{__name__}.predict: cannot find required variable {var} for diagnostic {key}")

                    # at last, compute the diagnostics
                    predictions[key] = func(predictions)
                    targets[key] = func(targets)


                # now check for static variables, we don't want these in the resulting datasets
                predictions = predictions[[key for key in list(predictions.coords)+list(predictions.data_vars) if "_static" not in key]]
                targets = targets[[key for key in list(targets.coords)+list(targets.data_vars) if "_static" not in key]]
                # Add t0 as new variable, and swap out for logical sample/batch index
                # swap dims to be [time (aka initial condition time), lead_time (aka forecast_time), level, lat, lon]
                predictions = swap_batch_time_dims(predictions, inittimes)
                targets = swap_batch_time_dims(targets, inittimes)

                # Store to zarr one batch at a time
                if k == 0:
                    if mpi_topo.is_root:
                        store_container(pname, predictions, time=batchloader.initial_times, mode="w")
                        store_container(tname, targets, time=batchloader.initial_times, mode="w")
                    mpi_topo.comm.Barrier()

                # Store to zarr
                spatial_region = {d: slice(None, None) for d in predictions.dims if d != "time"}
                rank_idx = mpi_topo.rank*batchloader.data_per_device
                st = k*batchloader.batch_size + rank_idx
                ed = st + batchloader.data_per_device
                region = {
                    "time": slice(st, ed),
                    **spatial_region,
                }
                predictions.to_zarr(pname, region=region)
                targets.to_zarr(tname, region=region)

            progress_bar.update()
        progress_bar.close()


def inference(Emulator):

    topo = MPITopology(log_dir=f"{Emulator.local_store_path}/logs/inference")
    emulator = Emulator(mpi_rank=topo.rank, mpi_size=topo.size)

    vds = Dataset(
        emulator,
        mode="validation",
        preload_batch=False,
    )

    validator = MPIExpandedBatchLoader(
        vds,
        batch_size=emulator.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        max_queue_size=1,
        sample_stride=emulator.sample_stride,
        mpi_topo=topo,
    )
    assert validator.data_per_device == 1

    # setup weights
    logging.info(f"Reading weights ...")
    ckpt_id = emulator.evaluation_checkpoint_id if emulator.evaluation_checkpoint_id is not None else emulator.num_epochs
    params, state = emulator.load_checkpoint(id=ckpt_id)

    predict(
        params=params,
        state=state,
        emulator=emulator,
        batchloader=validator,
        mpi_topo=topo,
    )

    validator.shutdown()
    logging.info("Done running inference")
