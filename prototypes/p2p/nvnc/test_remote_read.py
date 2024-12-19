import os
import logging
import sys
import numpy as np
import shutil

import dask
from distributed import Client

from graphufs.datasets import Dataset
from graphufs.batchloader import XBatchLoader, BatchLoader, MPIExpandedBatchLoader
from graphufs.tensorstore import PackedDataset as TSPackedDataset, BatchLoader as TSBatchLoader
from graphufs.log import setup_simple_log
from graphufs.mpi import MPITopology

from ufs2arco import Timer

from config import P2PEvaluator as Emulator
from test_stacked_read import print_time

def read_test(gufs, topo, num_tries=10):
    """Find optimal number of dask worker threads to read a single batch of data"""

    timer1 = Timer()

    training_data = Dataset(
        gufs,
        mode="training",
    )
    trainer = MPIExpandedBatchLoader(
        training_data,
        batch_size=gufs.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        rng_seed=0,
        mpi_topo=topo,
    )

    # --- What's the optimal number of dask worker threads to read a batch of data?
    avg_time = dict()

    # first try without dask
    iterloader = iter(trainer)
    timer1.start(f"no dask")
    for k in range(num_tries):
        inputs, targets, forcings = next(iterloader)
        logging.info(f"{k}")
    elapsed  = timer1.stop(f"Time with 0 workers = ")
    avg_time[0] = elapsed / num_tries

    # now with dask
    for num_workers in [2, 4, 8, 16, 32, 64, 128]:
        iterloader = iter(trainer)
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            timer1.start(f"{num_workers}")
            for k in range(num_tries):
                inputs, targets, forcings = next(iterloader)
                logging.info(f"{k}")

            elapsed = timer1.stop(f"Time with {num_workers} workers = ")
            avg_time[num_workers] = elapsed / num_tries

    print_time(gufs.batch_size, avg_time)

if __name__ == "__main__":

    total_time = Timer()
    total_time.start()

    # parse arguments
    topo = MPITopology(log_dir="/pscratch/sd/t/timothys/test-inference-read")
    dask.config.set(scheduler="processes", num_workers=2)
    logging.info("dask scheduler set to processes")
    dask.config.set(scheduler="threads", num_workers=2)
    logging.info("dask scheduler set to threads")
    emulator = Emulator(mpi_rank=topo.rank, mpi_size=topo.size)

    read_test(emulator, topo, num_tries=5)
    total_time.stop("Total Walltime")
