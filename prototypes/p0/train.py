import argparse
import os
import shutil
import threading
from functools import partial

import numpy as np
import optax
import xarray as xr
import xesmf as xe
from graphcast import checkpoint, graphcast
from graphufs import optimize, predict, run_forward
from jax import jit
from jax.random import PRNGKey
from ufs2arco.timer import Timer

from simple_emulator import P0Emulator

"""
Script to train and test graphufs over multiple chunks and epochs

Example usage:

    python3 run_training.py --chunks 10 --batch-size 1 --num-batches 16 --train

    This will train networks over 10 chunks where each chunk goes through 16 steps
    with a batch size of 1. You should get 10 checkpoints after training completes.

    Later, you can evaluate a specific model by specifying model id

    python3 run_training.py --chunks 10 --batch-size 1 --num-batches 16 --test --id 8
"""


def get_chunk_data(data: dict, n_batches: int = 4):
    """Get multiple training batches.

    Args:
        data (List[3]): A list containing the [inputs, targets, forcings]
        n_batches (int): Number of batches we want to read
    """
    print("Preparing Batches from Replay on GCS")

    inputs, targets, forcings = gufs.get_training_batches(
        n_optim_steps=n_batches,
    )

    # load into ram
    inputs.load()
    targets.load()
    forcings.load()

    data.update(
        {
            "inputs": inputs,
            "targets": targets,
            "forcings": forcings,
        }
    )

    print("Finished preparing batches")


def convert_wb2_format(ds, targets) -> xr.Dataset:
    """Convert a dataset into weatherbench2 compatible format. Details can be
    found in: https://weatherbench2.readthedocs.io/en/latest/evaluation.html.

    Args:
        ds (xr.Dataset): the xarray dadatset
        targets (xr.Dataset): a dataset that contains "inititime", forecast
                initialization time
    """

    # regrid to the obs coordinates
    ds_obs = xr.open_zarr(
        gufs.wb2_obs_url,
        storage_options={"token": "anon"},
    )
    ds_out = xr.Dataset(
        {
            "lat": (["lat"], ds_obs["latitude"].values),
            "lon": (["lon"], ds_obs["longitude"].values),
        }
    )
    regridder = xe.Regridder(
        ds,
        ds_out,
        "bilinear",
        periodic=True,
        reuse_weights=False,
        filename="graphufs_regridder",
    )
    ds = regridder(ds)

    # rename variables
    ds = ds.rename_vars(
        {
            "pressfc": "surface_pressure",
            "tmp": "temperature",
            "ugrd10m": "10m_u_component_of_wind",
            "vgrd10m": "10m_v_component_of_wind",
        }
    )

    # fix pressure levels to match obs
    ds["level"] = np.array(list(gufs.pressure_levels), dtype=np.float32)

    # remove batch dimension
    ds = ds.rename({"time": "t", "batch": "b"})
    ds = ds.stack(time=("b", "t"), create_index=False)
    ds = ds.drop_vars(["b", "t"])
    init_times = targets["inittime"].values
    lead_times = targets["time"].values
    ds = ds.assign_coords({"lead_time": lead_times, "time": init_times})
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # transpose the dimensions, and insert lead_time
    ds = ds.transpose("time", ..., "longitude", "latitude")
    for var in ds.data_vars:
        ds[var] = ds[var].expand_dims({"lead_time": ds.lead_time}, axis=1)

    return ds


def get_chunk_in_parallel(
    data: dict, data_0: dict, input_thread, it: int, args: dict
) -> threading.Thread:
    """Get a chunk of data in parallel with optimization/prediction. This keeps
    two big chunks (data and data_0) in RAM.

    Args:
        data (dict): the data being used by optimization/prediction process
        data_0 (dict): the data currently being fetched/processed
        input_thread: the input thread
        it: chunk number, it < 0 indicates first chunk
        args: CLI arguments
    """
    # make sure input thread finishes before copying data_0 to data
    if it >= 0:
        input_thread.join()
        for k, v in data_0.items():
            data[k] = v
    # don't prefetch a chunk on the last iteration
    if it < args.chunks - 1:
        input_thread = threading.Thread(
            target=get_chunk_data,
            args=(data_0, args.num_batches),
        )
        input_thread.start()
    # for first chunk, wait until input thread finishes
    if it < 0:
        input_thread.join()
    return input_thread


def init_model(data: dict):
    """Initialize model with random weights.

    Args:
        data (str): data to be used for initialization?
    """
    localtime.start("Initializing Optimizer and Parameters")

    init_jitted = jit(run_forward.init)
    params, state = init_jitted(
        rng=PRNGKey(gufs.init_rng_seed),
        emulator=gufs,
        inputs=data["inputs"].sel(batch=[0]),
        targets_template=data["targets"].sel(batch=[0]),
        forcings=data["forcings"].sel(batch=[0]),
    )

    localtime.stop()
    return params, state


def load_checkpoint(ckpt_path: str):
    """Load checkpoint.

    Args:
        ckpt_path (str): path to model
    """
    localtime.start("Loading weights")

    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")

    localtime.stop()
    return params, state


def save_checkpoint(gufs, params, ckpt_path: str) -> None:
    """Load checkpoint.

    Args:
        gufs: emulator class
        params: the parameters (weights) of the model
        ckpt_path (str): path to model
    """
    with open(ckpt_path, "wb") as f:
        ckpt = graphcast.CheckPoint(
            params=params,
            model_config=gufs.model_config,
            task_config=gufs.task_config,
            description="GraphCast model trained on UFS data",
            license="Public domain",
        )
        checkpoint.dump(f, ckpt)


def compute_metrics(
    predictions: xr.Dataset, target: xr.Dataset, stats: dict, it: int
) -> None:
    """Compute fast metrics (rmse and bias) between predictions and target.

    Args:
        predictions (xr.Dataset): the forecast
        target (xr.Dataset): the ground trutch
        stats (dict): dictionary containing statistics
        it (int): current chunk iteration id. This is needed for computing a running average
    """
    diff = predictions - targets
    rmse = np.sqrt((diff ** 2).mean())
    bias = diff.mean()

    # compute running average of rmse and bias
    for var_name, _ in rmse.data_vars.items():
        r = rmse[var_name].values
        b = bias[var_name].values
        if var_name in stats.keys():
            rmse_o = stats[var_name][0]
            bias_o = stats[var_name][1]
            r = rmse_o + (r - rmse_o) / (it + 1)
            b = bias_o + (b - bias_o) / (it + 1)
        stats[var_name] = [r, b]


def parse_args():
    """Parse CLI arguments."""

    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        required=False,
        help="Train model",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", required=False, help="Test model"
    )
    parser.add_argument(
        "--chunks",
        dest="chunks",
        required=False,
        type=int,
        default=1,
        help="Number of chunks per epoch.",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        required=False,
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        required=False,
        type=int,
        default=32,
        help="batch size.",
    )
    parser.add_argument(
        "--num-batches",
        "-n",
        dest="num_batches",
        required=False,
        type=int,
        default=32,
        help="Number of batches to read and load into RAM.",
    )
    parser.add_argument(
        "--id",
        "-i",
        dest="id",
        required=False,
        type=int,
        default=0,
        help="ID of neural networks to load.",
    )
    parser.add_argument(
        "--checkpoint-chunks",
        "-c",
        dest="checkpoint_chunks",
        required=False,
        type=int,
        default=1,
        help="Save weights after this many chunks are processed.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        required=False,
        default="nets",
        help="Path to network files",
    )
    parser.add_argument(
        "--latent-size",
        dest="latent_size",
        required=False,
        default=256,
        help="The latent space vector width. Set this lower, e.g. 32, for low RAM",
    )
    args = parser.parse_args()
    return args

def override_options(args, emulator):
    """ Override options in emulator class by those from CLI. """
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            arg_name = arg.replace('-', '_')
            if hasattr(emulator, arg_name):
                stored = getattr(emulator, arg_name)
                if stored is not None:
                    attr_type = type(getattr(emulator, arg_name))
                    value = attr_type(value)
                setattr(emulator, arg_name, value)

if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # initialize emulator and open dataset
    walltime = Timer()
    localtime = Timer()

    # override options
    override_options(args, P0Emulator)

    # initialize emulator
    gufs = P0Emulator()

    # get the first chunk of data
    data = {}
    data_0 = {}
    input_thread = None
    input_thread = get_chunk_in_parallel(data, data_0, input_thread, -1, args)

    # load weights or initialize a random model
    ckpt_id = args.id
    ckpt_path = f"{args.checkpoint_dir}/model_{ckpt_id}.npz"

    if os.path.exists(ckpt_path):
        params, state = load_checkpoint(ckpt_path)
    else:
        params, state = init_model(data_0)

    # training
    if args.train:
        walltime.start("Starting Training")

        # create checkpoint directory
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        optimizer = optax.adam(learning_rate=1e-4)

        # training loop
        for e in range(args.epochs):
            for it in range(args.chunks):

                # get chunk of data in parallel with NN optimization
                input_thread = get_chunk_in_parallel(data, data_0, input_thread, it, args)

                # optimize
                localtime.start("Starting Optimization")

                params, loss = optimize(
                    params=params,
                    state=state,
                    optimizer=optimizer,
                    emulator=gufs,
                    input_batches=data["inputs"],
                    target_batches=data["targets"],
                    forcing_batches=data["forcings"],
                )

                localtime.stop()

                # save weights
                if it % args.checkpoint_chunks == 0:
                    ckpt_id = it // args.checkpoint_chunks
                    ckpt_path = f"{args.checkpoint_dir}/model_{ckpt_id}.npz"
                    save_checkpoint(gufs, params, ckpt_path)

    # testing
    else:
        walltime.start("Starting Testing")

        # create predictions and targets zarr file for WB2
        predictions_zarr_name = f"{gufs.local_store_path}/graphufs_predictions.zarr"
        targets_zarr_name = f"{gufs.local_store_path}/graphufs_targets.zarr"
        if os.path.exists(predictions_zarr_name):
            shutil.rmtree(predictions_zarr_name)
        if os.path.exists(targets_zarr_name):
            shutil.rmtree(targets_zarr_name)

        stats = {}
        for it in range(args.chunks):

            # get chunk of data in parallel with inference
            input_thread = get_chunk_in_parallel(data, data_0, input_thread, it, args)

            # run predictions
            predictions = predict(
                params=params,
                state=state,
                emulator=gufs,
                input_batches=data["inputs"],
                target_batches=data["targets"],
                forcing_batches=data["forcings"],
            )

            # Compute rmse and bias comparing targets and predictions
            targets = data["targets"]
            compute_metrics(predictions, targets, stats, it)

            # write chunk by chunk to avoid storing all of it in memory
            predictions = convert_wb2_format(predictions, targets)
            predictions.to_zarr(predictions_zarr_name, mode="a")

            # write also targets to compute metrics against it with wb2
            targets = convert_wb2_format(targets, targets)
            targets.to_zarr(targets_zarr_name, mode="a")

        print("--------- Statistiscs ---------")
        for k, v in stats.items():
            print(f"{k:32s}: RMSE: {v[0]} BIAS: {v[1]}")

    # total walltime
    walltime.stop("Total Walltime")
