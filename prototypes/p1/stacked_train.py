import logging
import os

from graphufs import (
    init_model,
    save_checkpoint,
)
from graphufs.torch import Dataset, LocalDataset, DataLoader

from ufs2arco import Timer

from p1 import P1Emulator
from train import graphufs_optimizer


if __name__ == "__main__":

    timer1 = Timer()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )

    # parse arguments
    p1, args = P1Emulator.from_parser()

    tds = Dataset(
        p1,
        mode="training",
    )
    training_data = LocalDataset(
        p1,
        mode="training",
    )
    valid_data = LocalDataset(
        p1,
        mode="validation",
    )
    trainer = DataLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
    )
    validator = DataLoader(
        valid_data,
        batch_size=p1.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # setup
    logging.info(f"Initial Setup")
    loss_weights = p1.calc_loss_weights(tds)
    last_input_channel_mapping = get_last_input_mapping(tds)

    params, state = p1.load_checkpoint(0)

    loss_name = f"{p1.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    # setup optimizer
    n_linear = 100
    n_total = len(trainer)
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
            opt_state=opt_state,
            last_input_channel_mapping=last_input_channel_mapping,
        )

        # save weights every epoch
        p1.save_checkpoint(params, id=e+1)
        timer1.stop(f"Done with epoch {e+1}")

    logging.info("Done Training")
