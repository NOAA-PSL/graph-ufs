import os
import sys
import logging
from functools import partial

import dask
import numpy as np
import optax
from graphufs.stacked_training import (
    optimize,
    init_model,
)
from graphufs.datasets import Dataset, PackedDataset
from graphufs.tensorstore import PackedDataset as TSPackedDataset, BatchLoader as TSBatchLoader
from graphufs.batchloader import BatchLoader

from graphufs.utils import get_last_input_mapping
from graphufs import init_devices

from config import LatentTestEmulator

def graphufs_optimizer(
    n_linear,
    n_total,
    peak_value=1e-3,
):

    # define learning rate schedules
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_value,
        warmup_steps=n_linear,
        decay_steps=n_total,
        end_value=0.0,
    )

    # Adam optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(32),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            weight_decay=0.1,
        ),
    )
    return optimizer

if __name__ == "__main__":

    # logging isn't working for me on PSL, no idea why
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )

    # parse arguments
    emulator, args = LatentTestEmulator.from_parser()
    #dask.config.set(scheduler="threads", num_workers=emulator.dask_threads)

    # for multi-gpu training
    init_devices(emulator)

    # data generators
    tds = Dataset(emulator, mode="training")
    #chunks = {"sample": 1, "lat": -1, "lon": -1, "channel": -1}
    #training_data = PackedDataset(emulator, mode="training", chunks=chunks)
    #validation_data = PackedDataset(emulator, mode="validation", chunks=chunks)
    training_data = TSPackedDataset(emulator, mode="training")
    validation_data = TSPackedDataset(emulator, mode="validation")


    trainer = TSBatchLoader(
        training_data,
        batch_size=emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=emulator.num_workers,
        max_queue_size=emulator.max_queue_size,
    )
    validator = TSBatchLoader(
        validation_data,
        batch_size=emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=emulator.num_workers,
        max_queue_size=emulator.max_queue_size,
    )

    logging.info("Initializing Loss Function Weights and Stacked Mappings")
    # compute loss function weights once
    loss_weights = emulator.calc_loss_weights(tds)
    last_input_channel_mapping = get_last_input_mapping(tds)

    # initialize a random model
    logging.info("Initializing Optimizer and Parameters")
    inputs, _ = trainer.get_data()
    params, state = init_model(emulator, inputs, last_input_channel_mapping)
    emulator.save_checkpoint(params, id=0)

    loss_name = f"{emulator.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    # setup optimizer
    steps_in_epoch = len(trainer)
    n_total = emulator.num_epochs * steps_in_epoch
    n_linear = 1_000
    n_cosine = n_total - n_linear
    optimizer = graphufs_optimizer(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=1e-3,
    )

    logging.info(f"Starting Training with:")
    logging.info(f"\t batch_size = {emulator.batch_size}")
    logging.info(f"\t {n_linear} linearly increasing LR steps")
    logging.info(f"\t {n_cosine} cosine decay LR steps")
    logging.info(f"\t {n_total} total training steps")
    logging.info(f"\t {len(validator)} validation steps")

    # training
    opt_state = None
    for e in range(emulator.num_epochs):
        logging.info(f"Training on epoch {e}")

        # optimize
        params, loss, opt_state = optimize(
            params=params,
            state=state,
            optimizer=optimizer,
            emulator=emulator,
            trainer=trainer,
            validator=validator,
            weights=loss_weights,
            last_input_channel_mapping=last_input_channel_mapping,
            opt_state=opt_state,
        )

        # save weights
        emulator.save_checkpoint(params, id=e+1)

    logging.info("Done Training")
    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)
