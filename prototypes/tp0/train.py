import argparse
import logging
import os
import sys
import shutil
from functools import partial

import numpy as np
import pandas as pd
import optax
from graphufs.stacked_training import (
    optimize,
    init_model,
)
from graphufs.datasets import Dataset
from graphufs.batchloader import BatchLoader

from graphufs.log import setup_simple_log
from graphufs.utils import get_last_input_mapping
from graphufs.fvstatistics import FVStatisticsComputer
from graphufs import (
    init_devices,
)
import jax

from config import TP0Emulator

if __name__ == "__main__":

    # logging isn't working for me on PSL, no idea why
    setup_simple_log()

    # This is a bit of a hack to enable testing, for real
    # cases, we want to compute statistics during preprocessing
    # Note we want to do this before initializing emulator object
    # since it tries to pull the statistics there.
    stats_path = os.path.dirname(TP0Emulator.norm_urls["mean"])
    if not os.path.isdir(stats_path):
        logging.info(f"Could not find {stats_path}, computing statistics...")
        fvstats = FVStatisticsComputer(
            path_in=TP0Emulator.data_url,
            path_out=stats_path,
            interfaces=TP0Emulator.interfaces,
            start_date=None,
            end_date=TP0Emulator.training_dates[-1],
            time_skip=None,
            load_full_dataset=True,
            transforms=TP0Emulator.input_transforms,
        )
        all_variables = list(set(
            TP0Emulator.input_variables + TP0Emulator.forcing_variables + TP0Emulator.target_variables
        ))
        all_variables.append("log_spfh")
        all_variables.append("log_spfh2m")
        fvstats(all_variables, integration_period=pd.Timedelta(hours=3))

    # We don't parse arguments since we can't be inconsistent with stats
    # computed above
    gufs = TP0Emulator()

    # for multi-gpu training
    init_devices(gufs)

    # data generators
    training_data = Dataset(gufs, mode="training")
    validation_data = Dataset(gufs, mode="validation")
    # this loads the data in ... suboptimal I know
    logging.info("Loading Training and Validation Datasets")
    training_data.xds.load();
    validation_data.xds.load();
    logging.info("... done loading")

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

    # load weights or initialize a random model
    logging.info("Initializing Optimizer and Parameters")
    inputs, _ = trainer.get_data()
    params, state = init_model(gufs, inputs, last_input_channel_mapping)

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
            opt_state=opt_state,
        )

        # save weights
        ckpt_id = e
        gufs.save_checkpoint(params, ckpt_id)

    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)
