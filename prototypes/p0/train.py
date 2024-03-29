import argparse
import logging
import os
import shutil
from functools import partial

import optax
from graphufs import (
    optimize,
    predict,
    DataGenerator,
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

Usage:

    python3 -W ignore train.py --num-epochs 2 --chunks-per-epoch 2 --latent-size 32 --training-dates "1994-01-01T00" "1994-01-31T18"

    This will train networks for 2 epochs with 2 chunks per epoch with training dataset of first month of 1994.
    You should get 4 models (checkpoints) after training completes.

    Later, you can evaluate a specific model by specifying model id, and testing dataset range i.e. first month of 1995

    python3 -W ignore train.py --chunks-per-epoch 2 --latent-size 32 --test --id 3 --testing-dates  "1995-01-01T00"  "1995-01-31T18"
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
    # Todo: Handle dictionaries
    for k, v in vars(P0Emulator).items():
        if not k.startswith("__"):
            name = "--" + k.replace("_", "-")
            if v is None:
                parser.add_argument(
                    name,
                    dest=k,
                    required=False,
                    type=int,
                    help=f"{k}: default {v}",
                )
            elif isinstance(v, (tuple, list)) and len(v):
                tp = type(v[0])
                parser.add_argument(
                    name,
                    dest=k,
                    required=False,
                    nargs="+",
                    type=tp,
                    help=f"{k}: default {v}",
                )
            else:
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
                    attr_type = type(stored)
                    value = attr_type(value)
                setattr(P0Emulator, arg_name, value)

    return args


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # turn off absl warnings
    logging.getLogger("absl").setLevel(logging.CRITICAL)

    # initialize emulator and open dataset
    walltime = Timer()
    localtime = Timer()

    # initialize emulator
    gufs = P0Emulator()

    # data generators
    generator = DataGenerator(
        emulator=gufs,
        download_data=True,
        n_optim_steps=gufs.steps_per_chunk,
        mode="testing" if args.test else "training",
    )

    # load weights or initialize a random model
    checkpoint_dir = f"{gufs.local_store_path}/models"
    ckpt_id = args.id
    ckpt_path = f"{checkpoint_dir}/model_{ckpt_id}.npz"

    if os.path.exists(ckpt_path):
        localtime.start(f"Loading weights: {ckpt_path}")
        params, state = load_checkpoint(ckpt_path)
    else:
        localtime.start("Initializing Optimizer and Parameters")
        data = generator.get_data()  # just to figure out shapes
        params, state = init_model(gufs, data)
    localtime.stop()

    # training
    if not args.test:
        walltime.start("Starting Training")

        # create checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        optimizer = optax.adam(learning_rate=1e-4)

        # training loop
        for e in range(gufs.num_epochs):
            for c in range(gufs.chunks_per_epoch):
                print(f"Training on epoch {e} and chunk {c}")

                # get chunk of data in parallel with NN optimization
                generator.generate()
                data = generator.get_data()

                # optimize
                params, loss = optimize(
                    params=params,
                    state=state,
                    optimizer=optimizer,
                    emulator=gufs,
                    input_batches=data["inputs"],
                    target_batches=data["targets"],
                    forcing_batches=data["forcings"],
                )

                # save weights
                if c % gufs.checkpoint_chunks == 0:
                    ckpt_id = (e * gufs.chunks_per_epoch + c) // gufs.checkpoint_chunks
                    ckpt_path = f"{checkpoint_dir}/model_{ckpt_id}.npz"
                    save_checkpoint(gufs, params, ckpt_path)

            # reset generator at the end of an epoch
            if e != gufs.num_epochs - 1:
                generator = DataGenerator(
                    emulator=gufs,
                    download_data=False,
                    n_optim_steps=gufs.steps_per_chunk,
                    mode="training",
                )

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
        for c in range(gufs.chunks_per_epoch):
            print(f"Testing on chunk {c}")

            # get chunk of data in parallel with inference
            generator.generate()
            data = generator.get_data()

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
            compute_rmse_bias(predictions, targets, stats, c)

            # write chunk by chunk to avoid storing all of it in memory
            predictions = convert_wb2_format(gufs, predictions, inittimes)
            predictions.to_zarr(predictions_zarr_name, append_dim="time" if c else None)

            # write also targets to compute metrics against it with wb2
            targets = convert_wb2_format(gufs, targets, inittimes)
            targets.to_zarr(targets_zarr_name, append_dim="time" if c else None)

        print("--------- Statistiscs ---------")
        for k, v in stats.items():
            print(f"{k:32s}: RMSE: {v[0]} BIAS: {v[1]}")

    # total walltime
    walltime.stop("Total Walltime")
