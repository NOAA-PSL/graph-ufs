import os
import sys
import logging
import json
from mpi4py import MPI

from graphufs.datasets import Dataset
from graphufs.tensorstore import PackedDataset as TSPackedDataset, MPIBatchLoader as TSBatchLoader

from graphufs.stacked_mpi_training import (
    init_model,
    optimize,
)

from graphufs.optim import clipped_cosine_adamw
from graphufs.utils import get_last_input_mapping
from graphufs.stacked_utils import get_channel_index

def train(remote_emulator, emulator, topo, missing_samples=None,):
    
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,)

    # data generators
    tds = Dataset(remote_emulator, mode="training")

    # get the training and target meta data
    logging.info("Getting metadata for inputs and targets")
    xinputs, xtargets, xforcing = tds.get_xarrays(0)
    meta_xinputs = get_channel_index(xinputs)
    meta_xtargets = get_channel_index(xtargets)
    meta_xforcing = get_channel_index(xforcing)

    # get meta data for stacked inputs
    meta_xforcing_copy = {}
    for key, value in meta_xforcing.items():
        meta_xforcing_copy[key+len(meta_xinputs)] = value
    meta_sinputs = {**meta_xinputs, **meta_xforcing_copy}

    # get training and validation data
    training_data = TSPackedDataset(emulator, mode="training", missing_samples=missing_samples, 
                                    meta_inputs=meta_sinputs, meta_targets=meta_xtargets)
    validation_data = TSPackedDataset(emulator, mode="validation", meta_inputs=meta_sinputs,
                                      meta_targets=meta_xtargets)

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

    # get the training and target meta data
    #logging.info("Getting metadata for inputs and targets")
    #xinputs, xtargets, _ = tds.get_xarrays(0)
    #meta_targets = get_channel_index(xtargets)
    #meta_inputs = get_channel_index(xinputs)

    logging.info("Initializing Loss Function Weights and Stacked Mappings")
    # compute loss function weights once
    loss_weights = remote_emulator.calc_loss_weights(tds)
    last_input_channel_mapping = get_last_input_mapping(tds)

    # initialize a random model
    logging.info("Initializing Optimizer and Parameters")
    inputs, _ = trainer.get_data()
    params, state = init_model(
        emulator=emulator,
        inputs=inputs,
        last_input_channel_mapping=last_input_channel_mapping,
        mpi_topo=topo,
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
    peak_value = remote_emulator.lr_peak_value if hasattr(remote_emulator, "lr_peak_value") else 1e-3
    clip_grad_global_norm= remote_emulator.clip_grad_global_norm if hasattr(remote_emulator, "clip_grad_global_norm") else 32.
    weight_decay = remote_emulator.weight_decay if hasattr(remote_emulator, "weight_decay") else 0.1

    optimizer = clipped_cosine_adamw(
        n_linear=n_linear,
        n_total=n_total,
        peak_value=peak_value,
        clip_grad_global_norm=clip_grad_global_norm,
        weight_decay=weight_decay,
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
            meta_inputs = meta_xinputs,
            meta_targets = meta_xtargets,
        )
	
        # save weights
        logging.info(f"Done with epoch {e+1}")
        if topo.is_root:
            emulator.save_checkpoint(params, id=e+1)
            if emulator.log_tensorboard:
                emulator.log_all_metrics(loss, e)
	    	
    logging.info("Done Training")
    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)

    if emulator.log_tensorboard:
        emulator.finalize()
