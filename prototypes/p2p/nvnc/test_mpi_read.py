import os
from mpi4py import MPI
import logging
import sys
import numpy as np
import shutil

import jax
import dask
from distributed import Client

from graphufs.datasets import Dataset, PackedDataset
from graphufs.batchloader import XBatchLoader, BatchLoader
from graphufs.tensorstore import PackedDataset as TSPackedDataset, MPIBatchLoader as TSBatchLoader
from graphufs.log import setup_simple_log
from graphufs.mpi import MPITopology

from ufs2arco import Timer

from config import P2PTrainer as Emulator

def print_time(batch_size, avg_time, work="read", topo=None):

    logging.info(f" --- Time to {work} batch_size = {batch_size} --- ")
    logging.info(f"\tnum_workers\t avg seconds / batch")
    for key, val in avg_time.items():
        logging.info(f"\t{key}\t\t{val}")

def read_tensorstore_test(gufs, num_tries=10, topo=None):
    """Find optimal number of dask worker threads to read a single batch of data"""

    timer1 = Timer()

    training_data = TSPackedDataset(
        gufs,
        mode="training",
    )
    trainer = TSBatchLoader(
        training_data,
        batch_size=gufs.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        mpi_topo=topo,
    )

    # --- What's the optimal number of dask worker threads to read a batch of data?
    iterloader = iter(trainer)
    avg_time = dict()
    num_workers = 1
    timer1.start(f"{num_workers}")

    for k in range(num_tries):
        x,y = next(iterloader)

    elapsed = timer1.stop(f"Time with {num_workers} workers = ")
    avg_time[num_workers] = elapsed / num_tries

    print_time(gufs.batch_size, avg_time, topo=topo)

if __name__ == "__main__":


#    setup_simple_log()
    topo = MPITopology(log_dir="/pscratch/sd/t/timothys/test-mpi-read")
    if topo.is_root:
        total_time = Timer()
        total_time.start()

    # parse arguments
    emulator = Emulator()

    read_tensorstore_test(emulator, num_tries=10, topo=topo)

    if topo.is_root:
        total_time.stop("Total Walltime")

