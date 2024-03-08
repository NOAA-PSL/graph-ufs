import argparse
import os
import threading
from functools import partial

import numpy as np
import optax
import xarray as xr
from graphcast import checkpoint, graphcast
from graphufs import optimize, predict, run_forward
from jax import jit
from jax.random import PRNGKey
from ufs2arco.timer import Timer

from simple_emulator import P0Emulator


def get_chunk_data(ds: xr.Dataset, data: list, n_batches: int = 4, batch_size: int = 1):
    """Get multiple training batches
    Args:
        ds (xr.Dataset): xarray dataset
        data (List[3]): A list containing the [inputs, targets, forcings]
        n_batches (int): Number of batches we want to read
        batch_size (int): batch size
    """
    print("Preparing Batches from Replay on GCS")

    inputs, targets, forcings = gufs.get_training_batches(
        xds=ds,
        n_batches=n_batches,
        batch_size=batch_size,
        delta_t="6h",
        target_lead_time="18h",
    )

    # load into ram
    inputs.load()
    targets.load()
    forcings.load()

    data[0] = inputs
    data[1] = targets
    data[2] = forcings

    print("Finished preparing batches")


def parse_args():
    """Parse arguments."""

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
        "--steps",
        dest="steps",
        required=False,
        type=int,
        default=1,
        help="Number of steps (chunks) per epoch.",
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
        "--checkpoint-steps",
        "-c",
        dest="checkpoint_steps",
        required=False,
        type=int,
        default=1,
        help="Save weights after this many steps.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        required=False,
        default="nets",
        help="Path to network files",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # initialize emulator and open dataset
    walltime = Timer()
    localtime = Timer()
    gufs = P0Emulator()
    ds = xr.open_zarr(gufs.data_url, storage_options={"token": "anon"})

    # get the first chunk of data
    data_0 = [None] * 3
    input_thread = threading.Thread(
        target=get_chunk_data, args=(ds, data_0, args.num_batches, args.batch_size)
    )
    input_thread.start()
    input_thread.join()

    # load weights: this doesn't work at the moment
    ckpt_id = args.id
    ckpt_path = f"{args.checkpoint_dir}/model_{ckpt_id}.npz"

    if os.path.exists(ckpt_path):
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

    # initialize random network for testing
    else:
        localtime.start("Initializing Optimizer and Parameters")

        init_jitted = jit(run_forward.init)
        params, state = init_jitted(
            rng=PRNGKey(gufs.init_rng_seed),
            emulator=gufs,
            inputs=data_0[0].sel(batch=[0]),
            targets_template=data_0[1].sel(batch=[0]),
            forcings=data_0[2].sel(batch=[0]),
        )

        localtime.stop()

    # training
    if args.train:
        walltime.start("Starting Training")

        # create checkpoint directory
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        optimizer = optax.adam(learning_rate=1e-4)

        # training loop
        for e in range(args.epochs):
            for it in range(args.steps):

                # get chunk of data in parallel with NN optimization
                input_thread.join()
                data = [data_0[i] for i in range(3)]
                if it < args.steps - 1:
                    data_0 = [None] * 3
                    input_thread = threading.Thread(
                        target=get_chunk_data,
                        args=(ds, data_0, args.num_batches, args.batch_size),
                    )
                    input_thread.start()

                # optimize
                localtime.start("Starting Optimization")

                params, loss, diagnostics, opt_state, grads = optimize(
                    params=params,
                    state=state,
                    optimizer=optimizer,
                    emulator=gufs,
                    input_batches=data[0],
                    target_batches=data[1],
                    forcing_batches=data[2],
                )

                localtime.stop()

                # save weights
                if it % args.checkpoint_steps == 0:
                    ckpt_id = it // args.checkpoint_steps
                    with open(f"{args.checkpoint_dir}/model_{ckpt_id}.npz", "wb") as f:
                        ckpt = graphcast.CheckPoint(
                            params=params,
                            model_config=gufs.model_config,
                            task_config=gufs.task_config,
                            description="GraphCast model trained on UFS data",
                            license="Public domain",
                        )
                        checkpoint.dump(f, ckpt)

    # testing
    else:
        walltime.start("Starting Testing")

        predictions_list = []
        for it in range(args.steps):

            # get chunk of data in parallel with inference
            input_thread.join()
            data = [data_0[i] for i in range(3)]
            if it < args.steps - 1:
                data_0 = [None] * 3
                input_thread = threading.Thread(
                    target=get_chunk_data,
                    args=(ds, data_0, args.num_batches, args.batch_size),
                )
                input_thread.start()

            # run predictions
            predictions = predict(
                params=params,
                state=state,
                emulator=gufs,
                input_batches=data[0],
                target_batches=data[1],
                forcing_batches=data[2],
            )
            # append datasets
            predictions_list.append(predictions)

        ds_o = xr.concat(predictions_list, dim="batch")
        ds_o.to_netcdf("graphufs_predict.nc")

    # total walltime
    walltime.stop("Total Walltime")
