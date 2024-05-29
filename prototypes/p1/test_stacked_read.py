"""Notes:

"""
import logging
import os
import sys
import time
import numpy as np

from graphufs import init_devices
from graphufs.utils import get_last_input_mapping
from graphufs.torch import Dataset, LocalDataset, DataLoader, DataGenerator, DaskDataLoader
from graphufs.stacked_training import init_model, optimize

from ufs2arco import Timer

from p1stacked import P1Emulator
from train import graphufs_optimizer

import dask
#from dask.cache import Cache
#cache = Cache(1e10)
#cache.register()

class Formatter(logging.Formatter):
    def __init__(self, fmt):
        super().__init__(fmt)

    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def local_read_test(p1, num_tries=10):
    """Find optimal number of dask worker threads to read a single batch of data"""

    training_data = LocalDataset(
        p1,
        mode="training",
    )
    trainer = DaskDataLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
    )


    # --- What's the optimal number of dask worker threads to read a batch of data?
    iterloader = iter(trainer)
    avg_time = dict()
    for num_workers in [1, 2, 4, 8, 16, 24, 32]:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            timer1.start()
            for _ in range(num_tries):
                next(iterloader)
            elapsed = timer1.stop(f"Time with {num_workers} workers = ")
            avg_time[num_workers] = elapsed / num_tries

    print(f" --- Time to read batch_size = {p1.batch_size} --- ")
    print(f"\tnum_workers\t avg seconds / batch")
    for key, val in avg_time.items():
        print(f"\t{key}\t\t{val}")


if __name__ == "__main__":

    timer1 = Timer()

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    formatter = Formatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # parse arguments
    p1, args = P1Emulator.from_parser()

    local_read_test(p1)
