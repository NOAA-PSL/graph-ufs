"""Notes:
    Time to read a single batch of 4 samples took (note, see basically same testing samples not batch):
        1  worker thread  = 35 sec
        2  worker threads = 18 sec
        4  worker threads = 13 sec
        8  worker threads = 10 sec
        16 worker threads = 10 sec

    Time to read with 10 GB cache and 8 thread workers = 16.4 sec/batch
    vs
    Time to read without cache and 8 thread workers = 10 sec/batch
"""
import logging
import os
import sys

from graphufs import init_devices
from graphufs.utils import get_last_input_mapping
from graphufs.torch import Dataset, LocalDataset, DataLoader, DataGenerator
from graphufs.stacked_training import init_model, optimize

from ufs2arco import Timer

from p1nodwsrf import P1Emulator
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

if __name__ == "__main__":

    dask.config.set(scheduler="threads", num_workers=8)

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

    tds = Dataset(
        p1,
        mode="training",
        preload_batch=True,
    )
    training_data=tds
    trainer = DataLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
    )
    traingen = DataGenerator(
        trainer,
        num_workers=p1.num_workers,
        max_queue_size=p1.max_queue_size,
    )

    ## Does a dask Cache help?
    #num_tries = 10
    #timer1.start()
    #for _ in range(num_tries):
    #    next(iter(trainer))
    #elapsed = timer1.stop("Time = ")
    #print(f" ... avg time/try = {elapsed / num_tries:.1f}")



    ## What's the optimal number of dask worker threads to read a batch of data?
    #num_tries = 3
    #for num_workers in [1, 2, 4, 8, 16]:
    #    dask.config.set(scheduler="threads", num_workers=num_workers)
    #    timer1.start()
    #    for _ in range(num_tries):
    #        next(iter(trainer))
    #    elapsed = timer1.stop(f"Time with {num_workers} workers = ")
    #    print(f" ... avg time/try = {elapsed / num_tries:.1f}")

    # What's the optimal number of dask worker threads to read a sample of data?
    #num_tries = 3
    #k = 0
    #for num_workers in [1, 2, 4, 8, 16]:
    #    dask.config.set(scheduler="threads", num_workers=num_workers)
    #    timer1.start()
    #    for _ in range(num_tries):
    #        tds[k]
    #        k+=1
    #    elapsed = timer1.stop(f"Time with {num_workers} workers = ")
    #    print(f" ... avg time/try = {elapsed / num_tries:.1f}")

    for k in range(len(traingen)):
        x,y = traingen.get_data()
        logging.info(f"{k} / {len(traingen)}, qsize = {traingen.data_queue.qsize()}")

