import os
import logging
import sys
import numpy as np
import shutil

import dask
from distributed import Client

from graphufs.datasets import Dataset, PackedDataset
from graphufs.batchloader import XBatchLoader, BatchLoader
from graphufs.tensorstore import PackedDataset as TSPackedDataset, BatchLoader as TSBatchLoader
from graphufs.log import setup_simple_log

from ufs2arco import Timer

from config import P2TrainingEmulator as Emulator

def print_time(batch_size, avg_time, work="read"):
    print(f" --- Time to {work} batch_size = {batch_size} --- ")
    print(f"\tnum_workers\t avg seconds / batch")
    for key, val in avg_time.items():
        print(f"\t{key}\t\t{val}")

def read_test(gufs, num_tries=10):
    """Find optimal number of dask worker threads to read a single batch of data"""

    timer1 = Timer()

    training_data = PackedDataset(
        gufs,
        mode="training",
        chunks={"sample": 1, "lat":-1, "lon": -1, "channels":-1},
    )
    trainer = BatchLoader(
        training_data,
        batch_size=gufs.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # --- What's the optimal number of dask worker threads to read a batch of data?
    iterloader = iter(trainer)
    avg_time = dict()
    for num_workers in [8, 16, 24, 32]:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            timer1.start(f"{num_workers}")
            for k in range(num_tries):
                x,y = next(iterloader)

            elapsed = timer1.stop(f"Time with {num_workers} workers = ")
            avg_time[num_workers] = elapsed / num_tries

    print_time(gufs.batch_size, avg_time)

def read_tensorstore_test(gufs, num_tries=10):
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

    print_time(gufs.batch_size, avg_time)

if __name__ == "__main__":


    setup_simple_log()

    # parse arguments
    emulator = Emulator()

    read_test(emulator, num_tries=5)
    read_tensorstore_test(emulator, num_tries=5)
