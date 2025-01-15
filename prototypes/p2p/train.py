import os
import logging

from mpi4py import MPI

from graphufs.datasets import Dataset
from graphufs.tensorstore import PackedDataset as TSPackedDataset, MPIBatchLoader as TSBatchLoader
from graphufs.mpi import MPITopology

from graphufs.stacked_mpi_training import (
    init_model,
    optimize,
)

from graphufs.optim import clipped_cosine_adamw
from graphufs import utils
from graphufs import stacked_diagnostics

def train(RemoteEmulator, PackedEmulator):

    # initial setup
    topo = MPITopology(log_dir=f"{RemoteEmulator.local_store_path}/logs/training")
    emulator = PackedEmulator(mpi_rank=topo.rank, mpi_size=topo.size)
    remote_emulator = RemoteEmulator(mpi_rank=topo.rank, mpi_size=topo.size)

    # data generators
    tds = Dataset(remote_emulator, mode="training")
    training_data = TSPackedDataset(emulator, mode="training")
    validation_data = TSPackedDataset(emulator, mode="validation")

    trainer = TSBatchLoader(
        training_data,
        batch_size=emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=emulator.num_workers,
        max_queue_size=emulator.max_queue_size,
        mpi_topo=topo,
        rng_seed=10,
    )
    validator = TSBatchLoader(
        validation_data,
        batch_size=emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=emulator.num_workers,
        max_queue_size=emulator.max_queue_size,
        mpi_topo=topo,
        rng_seed=11,
    )

    logging.info("Initializing Loss Function Weights and Stacked Mappings")
    # compute loss function weights once
    loss_weights = remote_emulator.calc_loss_weights(tds)
    last_input_channel_mapping = utils.get_last_input_mapping(tds)


    # diagnostics
    diagnostic_mappings = None
    if emulator.diagnostics is not None:
        logging.info("Preparing Diagnostic Mappings")
        xinputs, xtargets, _ = tds.get_xarrays(0)
        input_meta = utils.get_channel_index(xinputs)
        output_meta = utils.get_channel_index(xtargets)
        diagnostic_mappings = stacked_diagnostics.prepare_diagnostic_functions(
            input_meta=input_meta,
            output_meta=output_meta,
            function_names=emulator.diagnostics,
            extra={
                "ak": emulator.ak,
                "bk": emulator.bk,
                "input_transforms": emulator.compilable_input_transforms,
                "output_transforms": emulator.compilable_output_transforms,
            },
        )

    # initialize a random model
    logging.info("Initializing Optimizer and Parameters")
    inputs, _ = trainer.get_data()
    params, state = init_model(
        emulator=emulator,
        inputs=inputs,
        last_input_channel_mapping=last_input_channel_mapping,
        mpi_topo=topo,
        diagnostic_mappings=diagnostic_mappings,
    )

    loss_name = f"{emulator.local_store_path}/loss.nc"
    if topo.is_root:
        emulator.save_checkpoint(params, id=0)
        if os.path.exists(loss_name):
            os.remove(loss_name)

    # setup optimizer
    steps_in_epoch = len(trainer)
    n_total = emulator.num_epochs * steps_in_epoch
    n_linear = 1_000
    n_cosine = n_total - n_linear
    optimizer = clipped_cosine_adamw(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=1e-3,
    )

    logging.info(f"Starting Training with:")
    logging.info(f"\t batch_size = {emulator.batch_size}")
    logging.info(f"\t {len(trainer)} training steps per epoch")
    logging.info(f"\t {len(validator)} validation steps per epoch")
    logging.info(f"\t ---")
    logging.info(f"\t {n_linear} linearly increasing LR steps")
    logging.info(f"\t {n_cosine} cosine decay LR steps")
    logging.info(f"\t {n_total} total training steps")

    # training
    opt_state = None
    for e in range(emulator.num_epochs):
        logging.info(f"Starting epoch {e+1}")

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
            mpi_topo=topo,
            diagnostic_mappings=diagnostic_mappings,
        )

        # save weights
        logging.info(f"Done with epoch {e+1}")
        if topo.is_root:
            emulator.save_checkpoint(params, id=e+1)

    logging.info("Done Training")
    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)
