import os
import logging

import optax
from graphufs import (
    optimize,
    DataGenerator,
    init_devices,
    init_model,
    get_approximate_memory_usage,
)

from p1 import P1Emulator
from ufs2arco import Timer


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

    timer1 = Timer()
    timer2 = Timer()

    # parse arguments
    p1, args = P1Emulator.from_parser()

    # for multi-gpu training
    init_devices(p1)

    # data generators
    trainer = DataGenerator(
        emulator=p1,
        n_optim_steps=p1.steps_per_chunk,
        max_queue_size=p1.max_queue_size,
        mode="testing" if args.test else "training",
    )

    validator = DataGenerator(
        emulator=p1,
        n_optim_steps=p1.steps_per_chunk,
        max_queue_size=p1.max_queue_size,
        mode="validation",
    )

    # compute approximate RAM usage and warn the user
    mem_usage = get_approximate_memory_usage([trainer, validator], p1.max_queue_size)
    logging.info("*****************************************************")
    logging.info(f"**     Total approximate memory usage {mem_usage:.0f} Gbs     ***")
    logging.info("** Make sure you have RAM safely above this value ***")
    logging.info("*****************************************************")

    # load weights or initialize a random model
    logging.info(f"Initializing weights: {0}")
    data = trainer.get_data()
    params, state = init_model(p1, data)
    loss_name = f"{p1.local_store_path}/loss.nc"
    if os.path.exists(loss_name):
        os.remove(loss_name)

    # have to divide steps by num gpus so that LR progresses
    # with the number of parallel optimization steps
    steps_in_epoch = p1.chunks_per_epoch * p1.steps_per_chunk
    n_total = p1.num_epochs * steps_in_epoch
    # use maximum of 1% of total steps or one epoch as warmup
    # Note that: graphcast uses 1/299 = 0.33% for warmup
    n_linear = max(n_total // 100, steps_in_epoch)
    n_linear = n_linear // p1.num_gpus
    n_total = n_total // p1.num_gpus
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
        for c in range(p1.chunks_per_epoch):
            logging.info(f"Training on epoch {e+1} and chunk {c+1}")
            timer2.start()

            # get chunk of data in parallel with NN optimization
            if p1.chunks_per_epoch > 1 and not (
                e == 0 and c == 0 and p1.max_queue_size == 1
            ):
                trainer.generate()
                validator.generate()

            data = trainer.get_data()
            data_valid = validator.get_data()

            # optimize
            params, loss, opt_state = optimize(
                params=params,
                state=state,
                optimizer=optimizer,
                emulator=p1,
                training_data=data,
                validation_data=data_valid,
                opt_state=opt_state,
            )
            timer2.stop(f"Done with chunk {c+1}")

        # save weights every epoch
        p1.save_checkpoint(params, id=e+1)
        timer1.stop(f"Done with epoch {e+1}")

    logging.info("Done Training")
