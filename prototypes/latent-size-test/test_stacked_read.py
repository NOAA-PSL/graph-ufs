import os
import logging
import sys
import numpy as np
import shutil

import dask
from distributed import Client

from graphufs.datasets import Dataset, PackedDataset
from graphufs.batchloader import XBatchLoader, BatchLoader
from graphufs.tensorstore import BatchLoader as TSBatchLoader

from ufs2arco import Timer

from config import LatentTestEmulator

class Formatter(logging.Formatter):
    def __init__(self, fmt):
        super().__init__(fmt)

    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def print_time(batch_size, avg_time, work="read"):
    print(f" --- Time to {work} batch_size = {batch_size} --- ")
    print(f"\tnum_workers\t avg seconds / batch")
    for key, val in avg_time.items():
        print(f"\t{key}\t\t{val}")

def remote_read_test(p1, num_tries=10):
    """Find optimal number of dask worker threads to read a single batch of data"""

    timer1 = Timer()

    training_data = Dataset(
        p1,
        mode="training",
        preload_batch=False,
    )
    trainer = BatchLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # --- What's the optimal number of dask worker threads to read a batch of data?
    iterloader = iter(trainer)
    avg_time = dict()
    for num_workers in [4, 8, 16, 24, 32, 48, 64, 96]:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            timer1.start(f"{num_workers}")
            for k in range(num_tries):
                x,y = next(iterloader)

            elapsed = timer1.stop(f"Time with {num_workers} workers = ")
            avg_time[num_workers] = elapsed / num_tries

    print_time(p1.batch_size, avg_time)

def remote_read_tensorstore_test(p1, num_tries=10):
    """Find optimal number of dask worker threads to read a single batch of data"""

    timer1 = Timer()

    training_data = Dataset(
        p1,
        mode="training",
        preload_batch=False,
    )
    trainer = TSBatchLoader(
        training_data,
        batch_size=p1.batch_size,
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

    print_time(p1.batch_size, avg_time)

if __name__ == "__main__":


    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    formatter = Formatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # parse arguments
    emulator = LatentTestEmulator()

    remote_read_test(emulator, num_tries=5)
    #remote_read_tensorstore_test(emulator, num_tries=5)
