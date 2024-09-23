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
from graphufs.datasets import Dataset
from graphufs.batchloader import BatchLoader

from graphufs.utils import get_last_input_mapping
from graphufs import (
    init_devices,
)
import jax

from config import TP0Emulator

if __name__ == "__main__":

    # logging isn't working for me on PSL, no idea why
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )

    # parse arguments
    gufs, args = TP0Emulator.from_parser()

    # for multi-gpu training
    init_devices(gufs)

    # data generators
    training_data = Dataset(gufs, mode="training")
    validation_data = Dataset(gufs, mode="validation")
    # this loads the data in ... suboptimal I know
    training_data.xds.load();
    validation_data.xds.load();
    trainer = BatchLoader(
        training_data,
        batch_size=gufs.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=gufs.num_workers,
    )
    validator = BatchLoader(
        validation_data,
        batch_size=gufs.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=gufs.num_workers,
    )

    # compute loss function weights once
    weights = gufs.calc_loss_weights(training_data)

    # this is tricky, because it needs to be "rebuildable" in JAX's eyes
    # so better to just explicitly pass it around
    last_input_channel_mapping = get_last_input_mapping(training_data)

    stacked_input_transforms, stacked_output_transforms, stacked_target_transforms = gufs.get_stacked_transforms(training_data)

    # load weights or initialize a random model
    logging.info("Initializing Optimizer and Parameters")
    inputs, _ = trainer.get_data()
    params, state = init_model(
        gufs,
        inputs,
        last_input_channel_mapping,
        stacked_input_transforms,
        stacked_output_transforms,
        stacked_target_transforms,
    )

    loss_name = f"{gufs.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    # training
    opt_state = None
    logging.info("Starting Training")

    optimizer = optax.adam(learning_rate=1e-4)

    # training loop
    for e in range(gufs.num_epochs):
        logging.info(f"Training on epoch {e}")

        # optimize
        params, loss, opt_state = optimize(
            params=params,
            state=state,
            optimizer=optimizer,
            emulator=gufs,
            trainer=trainer,
            validator=validator,
            weights=weights,
            last_input_channel_mapping=last_input_channel_mapping,
            input_transforms=stacked_input_transforms,
            output_transforms=stacked_output_transforms,
            target_transforms=stacked_target_transforms,
            opt_state=opt_state,
        )

        # save weights
        ckpt_id = e
        gufs.save_checkpoint(params, ckpt_id)

    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)
