import argparse
import os
import shutil
from functools import partial

import optax
from graphufs import (
    optimize,
    predict,
    get_chunk_data,
    get_chunk_in_parallel,
    init_model,
    load_checkpoint,
    save_checkpoint,
    convert_wb2_format,
    compute_rmse_bias,
)
from ufs2arco.timer import Timer

from simple_emulator import P0Emulator

"""
Script to train and test graphufs over multiple chunks and epochs

Example usage:

    python3 train.py --chunks-per-epoch 2 --batches-per-chunk 1 --latent-size 32

    This will train networks over 2 chunks where each chunk goes through 16 steps
    with a batch size of 1. You should get 10 checkpoints after training completes.

    Later, you can evaluate a specific model by specifying model id

    python3 train.py --chunks-per-epoch 1 --batches-per-chunk 1 --latent-size 32 --test --id 1
"""


def parse_args():
    """Parse CLI arguments."""

    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--test",
        dest="test",
        action="store_true",
        required=False,
        help="Test model specified with --id. Otherwise train model.",
    )
    parser.add_argument(
        "--id",
        "-i",
        dest="id",
        required=False,
        type=int,
        default=0,
        help="ID of neural networks to resume training/testing from.",
    )

    # add options from P0Emulator
    for k, v in vars(P0Emulator).items():
        if not k.startswith("__"):
            name = "--" + k.replace("_", "-")
            parser.add_argument(
                name,
                dest=k,
                required=False,
                type=type(v),
                default=v,
                help=f"{k}: default {v}",
            )

    # parse CLI args
    args = parser.parse_args()

    # override options in emulator class by those from CLI
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            arg_name = arg.replace("-", "_")
            if hasattr(P0Emulator, arg_name):
                stored = getattr(P0Emulator, arg_name)
                if stored is not None:
                    attr_type = type(getattr(P0Emulator, arg_name))
                    value = attr_type(value)
                setattr(P0Emulator, arg_name, value)

    return args


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # initialize emulator and open dataset
    walltime = Timer()
    localtime = Timer()

    # initialize emulator
    gufs = P0Emulator()

    # get the first chunk of data
    data = {}
    data_0 = {}
    input_thread = None
    input_thread = get_chunk_in_parallel(gufs, data, data_0, input_thread, -1, args)

    # load weights or initialize a random model
    ckpt_id = args.id
    ckpt_path = f"{args.checkpoint_dir}/model_{ckpt_id}.npz"

    if os.path.exists(ckpt_path):
        localtime.start("Loading weights")
        params, state = load_checkpoint(ckpt_path)
    else:
        localtime.start("Initializing Optimizer and Parameters")
        params, state = init_model(gufs, data_0)
    localtime.stop()

    # training
    if not args.test:
        walltime.start("Starting Training")

        # create checkpoint directory
        if not os.path.exists(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        optimizer = optax.adam(learning_rate=1e-4)

        # training loop
        for e in range(args.num_epochs):
            for it in range(args.chunks_per_epoch):

                # get chunk of data in parallel with NN optimization
                input_thread = get_chunk_in_parallel(
                    gufs, data, data_0, input_thread, it, args
                )

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
        for it in range(args.chunks_per_epoch):

            # get chunk of data in parallel with inference
            input_thread = get_chunk_in_parallel(
                gufs, data, data_0, input_thread, it, args
            )

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
            inittimes = data["inittimes"]
            compute_rmse_bias(predictions, targets, stats, it)

            # write chunk by chunk to avoid storing all of it in memory
            predictions = convert_wb2_format(gufs, predictions, inittimes)
            predictions.to_zarr(predictions_zarr_name, mode="a")

            # write also targets to compute metrics against it with wb2
            targets = convert_wb2_format(gufs, targets, inittimes)
            targets.to_zarr(targets_zarr_name, mode="a")

        print("--------- Statistiscs ---------")
        for k, v in stats.items():
            print(f"{k:32s}: RMSE: {v[0]} BIAS: {v[1]}")

    # total walltime
    walltime.stop("Total Walltime")
