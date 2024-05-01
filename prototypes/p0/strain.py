import argparse
import logging
import os
import sys
import shutil
from functools import partial

import numpy as np
import optax
from graphufs.stacked_training import (
    optimize,
    init_model,
)
from graphufs.utils import get_last_input_mapping
from graphufs import (
    load_checkpoint,
    save_checkpoint,
    convert_wb2_format,
    compute_rmse_bias,
    add_emulator_arguments,
    set_emulator_options,
    GraphUFSDataset,
    init_devices,
)
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import default_collate
import jax

from simple_emulator import P0Emulator
from train import parse_args

if __name__ == "__main__":

    # logging isn't working for me on PSL, no idea why
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
    )

    # parse arguments
    args = parse_args()

    # initialize emulator
    gufs = P0Emulator()

    # for multi-gpu training
    init_devices(gufs)

    # data generators
    training_data = GraphUFSDataset(gufs, mode="training")
    training_data.xds.load();
    generator = TorchDataLoader(
        training_data,
        batch_size=gufs.batch_size,
        shuffle=True,
        collate_fn=lambda batch : jax.tree_util.tree_map(np.asarray, default_collate(batch)),
    )

    # NOTE: I could really clean up
    # - what is an emulator attribute and what is a dataset attribute
    # - how these are provided to the model/optimize

    # compute loss function weights once
    weights = gufs.calc_loss_weights(training_data)

    # this is tricky, because it needs to be "rebuildable" in JAX's eyes
    # so better to just explicitly pass it around
    last_input_channel_mapping = get_last_input_mapping(training_data)

    # load weights or initialize a random model
    checkpoint_dir = f"{gufs.local_store_path}/models"
    ckpt_id = args.id
    ckpt_path = f"{checkpoint_dir}/model_{ckpt_id}.npz"

    if os.path.exists(ckpt_path) and args.id >= 0:
        logging.info(f"Loading weights: {ckpt_path}")
        params, state = load_checkpoint(ckpt_path)
    else:
        logging.info("Initializing Optimizer and Parameters")
        params, state = init_model(gufs, training_data, last_input_channel_mapping)
        loss_name = f"{gufs.local_store_path}/loss.nc"
        if os.path.exists(loss_name):
            os.remove(loss_name)

    # training
    if not args.test:
        logging.info("Starting Training")

        # create checkpoint directory
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        optimizer = optax.adam(learning_rate=1e-4)

        # training loop
        for e in range(gufs.num_epochs):
            logging.info(f"Training on epoch {e}")

            # optimize
            params, loss = optimize(
                params=params,
                state=state,
                optimizer=optimizer,
                emulator=gufs,
                generator=generator,
                weights=weights,
                last_input_channel_mapping=last_input_channel_mapping,
            )

            # save weights
            ckpt_id = e
            ckpt_path = f"{checkpoint_dir}/model_{ckpt_id}.npz"
            save_checkpoint(gufs, params, ckpt_path)

    # testing
    else:
        raise NotImplementedError
