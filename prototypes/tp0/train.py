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
from graphufs.optim import clipped_cosine_adamw

def calc_stats(Emulator):

    # This is a bit of a hack to enable testing, for real
    # cases, we want to compute statistics during preprocessing
    # Note we want to do this before initializing emulator object
    # since it tries to pull the statistics there.
    fvstats = FVStatisticsComputer(
        path_in=Emulator.data_url,
        path_out=os.path.dirname(Emulator.norm_urls["mean"]),
        interfaces=Emulator.interfaces,
        start_date=None,
        end_date=Emulator.training_dates[-1],
        time_skip=None,
        load_full_dataset=True,
        transforms=Emulator.input_transforms,
    )
    all_variables = list(set(
        Emulator.input_variables + Emulator.forcing_variables + Emulator.target_variables
    ))
    all_variables.append("log_spfh")
    all_variables.append("log_spfh2m")
    fvstats(all_variables, integration_period=pd.Timedelta(hours=3))

def train(Emulator):

    # We don't parse arguments since we can't be inconsistent with stats
    # computed above
    gufs = Emulator()

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
    gufs.save_checkpoint(params, id=0)

    loss_name = f"{gufs.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    # setup optimizer
    steps_in_epoch = len(trainer)
    n_total = gufs.num_epochs * steps_in_epoch
    n_linear = max(10, int(len(trainer)/100))
    n_cosine = n_total - n_linear
    optimizer = clipped_cosine_adamw(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=1e-3,
    )

    logging.info(f"Starting Training with:")
    logging.info(f"\t batch_size = {gufs.batch_size}")
    logging.info(f"\t {len(trainer)} training steps per epoch")
    logging.info(f"\t {len(validator)} validation steps per epoch")
    logging.info(f"\t ---")
    logging.info(f"\t {n_linear} linearly increasing LR steps")
    logging.info(f"\t {n_cosine} cosine decay LR steps")
    logging.info(f"\t {n_total} total training steps")

    # training
    opt_state = None
    for e in range(gufs.num_epochs):
        logging.info(f"Starting epoch {e}")

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

        logging.info(f"Done with epoch {e}")

        # save weights
        gufs.save_checkpoint(params, id=e+1)

    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)

if __name__ == "__main__":

    # logging isn't working for me on PSL, no idea why
    setup_simple_log()

    stats_path = os.path.dirname(TP0Emulator.norm_urls["mean"])
    if not os.path.isdir(stats_path):
        logging.info(f"Could not find {stats_path}, computing statistics...")
        calc_stats(TP0Emulator)

    train(TP0Emulator)


