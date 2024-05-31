"""Notes
inputs size / sample = 55 MB
targets size / sample = 24 MB
"""
import logging
import os
import sys
import dask

from graphufs import init_devices
from graphufs.utils import get_last_input_mapping
from graphufs.datasets import Dataset, PackedDataset
from graphufs.batchloader import BatchLoader
from graphufs.stacked_training import init_model, optimize

from ufs2arco import Timer

from p1stacked import P1Emulator
from train import graphufs_optimizer


if __name__ == "__main__":

    timer1 = Timer()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )

    # parse arguments
    p1, args = P1Emulator.from_parser()
    init_devices(p1)
    dask.config.set(scheduler="threads", num_workers=16)

    tds = Dataset(
        p1,
        mode="training",
        preload_batch=True,
    )
    training_data = PackedDataset(
        p1,
        mode="training",
    )
    valid_data = PackedDataset(
        p1,
        mode="validation",
    )
    trainer = BatchLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=p1.num_workers,
        max_queue_size=p1.max_queue_size,
    )
    validator = BatchLoader(
        valid_data,
        batch_size=p1.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=p1.num_workers,
        max_queue_size=p1.max_queue_size,
    )

    # setup
    logging.info(f"Initial Setup")
    loss_weights = p1.calc_loss_weights(tds)
    last_input_channel_mapping = get_last_input_mapping(tds)

    inputs, _ = trainer.get_data()
    params, state = init_model(p1, inputs, last_input_channel_mapping)

    loss_name = f"{p1.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    # setup optimizer
    steps_in_epoch = len(trainer)
    n_total = p1.num_epochs * steps_in_epoch
    n_linear = max( n_total // 100, steps_in_epoch )
    n_cosine = n_total - n_linear
    optimizer = graphufs_optimizer(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=1e-3,
    )

    # training loop
    logging.info(f"Starting Training with:")
    logging.info(f"\t {n_linear} linearly increasing LR steps")
    logging.info(f"\t {n_cosine} cosine decay LR steps")
    logging.info(f"\t {n_total} total training steps")
    logging.info(f"\t {len(validator)} validation steps")
    opt_state = None
    for e in range(p1.num_epochs):
        timer1.start()
        logging.info(f"Training on epoch {e+1}")

        # optimize
        params, loss, opt_state = optimize(
            params=params,
            state=state,
            optimizer=optimizer,
            emulator=p1,
            trainer=trainer,
            validator=validator,
            weights=loss_weights,
            last_input_channel_mapping=last_input_channel_mapping,
            opt_state=opt_state,
        )

        # save weights every epoch
        p1.save_checkpoint(params, id=e+1)
        timer1.stop(f"Done with epoch {e+1}")

    trainer.shutdown()
    validator.shutdown()
    logging.info("Done Training")
