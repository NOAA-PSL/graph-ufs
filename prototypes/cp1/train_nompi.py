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
from graphufs.log import setup_simple_log

from graphufs.utils import get_last_input_mapping
from graphufs import init_devices

from config import (
    CP1TrainingEmulator as RawCoupledEmulator,
    CP1PreprocessedEmulator as TransformedCoupledEmulator,
)


if __name__ == "__main__":

    # initial setup
    setup_simple_log()
    trans_emulator = TransformedCoupledEmulator()
    raw_emulator = RawCoupledEmulator()
    init_devices(trans_emulator)

    # data generators
    tds = Dataset(raw_emulator, mode="training")
    training_data = TSPackedDataset(trans_emulator, mode="training")
    validation_data = TSPackedDataset(trans_emulator, mode="validation")


    trainer = TSBatchLoader(
        training_data,
        batch_size=trans_emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=trans_emulator.num_workers,
        max_queue_size=trans_emulator.max_queue_size,
    )
    validator = TSBatchLoader(
        validation_data,
        batch_size=trans_emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=trans_emulator.num_workers,
        max_queue_size=trans_emulator.max_queue_size,
    )

    logging.info("Initializing Loss Function Weights and Stacked Mappings")
    # compute loss function weights once
    loss_weights = raw_emulator.calc_loss_weights(tds)
    last_input_channel_mapping = get_last_input_mapping(tds)

    # initialize a random model
    logging.info("Initializing Optimizer and Parameters")
    inputs, _ = trainer.get_data()
    params, state = init_model(trans_emulator, inputs, last_input_channel_mapping)
    trans_emulator.save_checkpoint(params, id=0)

    loss_name = f"{trans_emulator.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    # setup optimizer
    steps_in_epoch = len(trainer)
    n_total = trans_emulator.num_epochs * steps_in_epoch
    n_linear = 1000
    n_cosine = n_total - n_linear
    optimizer = graphufs_optimizer(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=1e-3,
    )

    logging.info(f"Starting Training with:")
    logging.info(f"\t batch_size = {trans_emulator.batch_size}")
    logging.info(f"\t {len(trainer)} training steps per epoch")
    logging.info(f"\t {len(validator)} validation steps per epoch")
    logging.info(f"\t ---")
    logging.info(f"\t {n_linear} linearly increasing LR steps")
    logging.info(f"\t {n_cosine} cosine decay LR steps")
    logging.info(f"\t {n_total} total training steps")

    # training
    opt_state = None
    for e in range(trans_emulator.num_epochs):
        logging.info(f"Starting epoch {e}")

        # optimize
        params, loss, opt_state = optimize(
            params=params,
            state=state,
            optimizer=optimizer,
            emulator=trans_emulator,
            trainer=trainer,
            validator=validator,
            weights=loss_weights,
            last_input_channel_mapping=last_input_channel_mapping,
            opt_state=opt_state,
        )

        # save weights
        logging.info(f"Done with epoch {e}")
        trans_emulator.save_checkpoint(params, id=e+1)

    logging.info("Done Training")
    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)
