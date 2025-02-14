import argparse
import logging
import os
import sys
import shutil
from functools import partial

import numpy as np
import xarray as xr
import pandas as pd
import optax
from graphufs.stacked_training import init_model, optimize
#from graphufs.stacked_parallel_training import optimize
from graphufs.datasets import Dataset
from graphufs.batchloader import BatchLoader

from graphufs.log import setup_simple_log
from graphufs.utils import get_last_input_mapping
from graphufs.fvstatistics import FVStatisticsComputer
from graphufs.training import (
    init_devices,
)
import jax
from graphufs.stacked_utils import get_channel_index
from config import StackedCP0Emulator
from graphufs.optim import clipped_cosine_adamw

def calc_stats(Emulator, comp="atm"):

    # This is a bit of a hack to enable testing, for real
    # cases, we want to compute statistics during preprocessing
    # Note we want to do this before initializing emulator object
    # since it tries to pull the statistics there.
    if comp in ["atm", "ice", "land"]:
        time_skip = 2 # everything is in 3 hour time steps in fv3

    elif comp.lower() == "ocn".lower():
        time_skip = 1 # time step is 6 hour for the oceans  
    
    else:
        raise ValueError("comp values can only be atm, ocn, ice, or land")
   
    open_zarr_kwargs = {
        "storage_options": {"token": "anon"},
    }

    to_zarr_kwargs = {
        "mode": "a",
    }

    fvstats = FVStatisticsComputer(
        path_in=Emulator.data_url[comp],
        path_out=os.path.dirname(Emulator.norm_urls[comp]["mean"]),
        interfaces=Emulator.interfaces[comp],
        comp=comp,
        start_date=None,
        end_date=Emulator.training_dates[-1],
        time_skip=time_skip,
        load_full_dataset=True,
        transforms=Emulator.input_transforms,
        open_zarr_kwargs=open_zarr_kwargs,
        to_zarr_kwargs=to_zarr_kwargs
    )

    tisr_args = {}

    if comp.lower() == "atm":
        all_variables = list(set(
            Emulator.atm_input_variables + Emulator.atm_forcing_variables + Emulator.atm_target_variables
        )) 
        all_variables.append("log_spfh")
        all_variables.append("log_spfh2m")
        tisr_args["integration_period"] = pd.Timedelta(hours=6)
    
    elif comp.lower() == "ocn":
        all_variables = list(set(
            Emulator.ocn_input_variables + Emulator.ocn_forcing_variables + Emulator.ocn_target_variables
        ))
    
    elif comp.lower() == "land": 
        all_variables = list(set(
            Emulator.land_input_variables + Emulator.land_forcing_variables + Emulator.land_target_variables
        ))
    
    elif comp.lower() == "ice": 
        all_variables = list(set(
            Emulator.ice_input_variables + Emulator.ice_forcing_variables + Emulator.ice_target_variables
        ))
    else: 
        raise ValueError(f"{comp} not supported")

    fvstats(all_variables, **tisr_args)

def train(Emulator):

    # We don't parse arguments since we can't be inconsistent with stats
    # computed above
    gufs = Emulator()

    # for multi-gpu training
    init_devices(gufs)

    # data generators
    training_data = Dataset(gufs, mode="training")
    sample_batch_inputs, sample_batch_targets = training_data.__getitem__(0)
    print("training data:", training_data.__getitem__(0))
    validation_data = Dataset(gufs, mode="validation")
    print("validation data:", validation_data.__getitem__(0))
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
    # same is true for channel to var mapping for targets as well, which is
    # required for masking purposes.
    xinputs, xtargets, xforcing = training_data.get_xarrays(0)
    meta_targets = get_channel_index(xtargets)
    meta_inputs = get_channel_index(xinputs)

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
            meta_inputs = meta_inputs,
            meta_targets = meta_targets,
        )

        logging.info(f"Done with epoch {e}")

        # save weights
        gufs.save_checkpoint(params, id=e+1)

    trainer.shutdown(cancel=True)
    validator.shutdown(cancel=True)

if __name__ == "__main__":

    # logging isn't working for me on PSL, no idea why
    setup_simple_log()

    # there is a central local stats path for all model components
    stats_path = os.path.dirname(StackedCP0Emulator.norm_url["mean"])

    if not os.path.isdir(stats_path):
        logging.info(f"Could not find {stats_path}, computing statistics...")
        for comp in ["atm","ocn","land","ice"]:
            calc_stats(StackedCP0Emulator, comp=comp)

    train(StackedCP0Emulator)
