import os
import logging
import sys
import numpy as np
import shutil

import dask
from distributed import Client

from graphufs.datasets import Dataset, PackedDataset
from graphufs.batchloader import XBatchLoader, BatchLoader


from ufs2arco import Timer

from p1stacked import P1Emulator

# this didn't help anything at all
#from dask.cache import Cache
#cache = Cache(1e10)
#cache.register()

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

def remote_readwrite_test(p1, num_tries=10, write_dir="./test-remote-io", batch_size=None):
    """Find optimal number of dask worker threads to read a single batch of data"""

    timer1 = Timer()
    timer2 = Timer()


    batch_size=p1.batch_size if batch_size is None else batch_size
    training_data = Dataset(
        p1,
        mode="training",
        preload_batch=False,
        chunks={"batch":1, "lat":-1, "lon":-1, "channels":13},
    )
    trainer = XBatchLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    if os.path.isdir(write_dir):
        shutil.rmtree(write_dir)

    # --- What's the optimal number of dask worker threads to read a batch of data?
    iterloader = iter(trainer)
    avg_time = dict()
    avg_write_time = dict()
    for num_workers in [4, 8, 16, 32]:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            timer1.start(f"{num_workers}")
            write_elapsed = 0.
            for k in range(num_tries):
                x,y = next(iterloader)
                timer2.start()
                x=x.chunk(training_data.chunks)
                y=y.chunk(training_data.chunks)
                x.to_dataset(name="inputs").to_zarr(f"{write_dir}/inputs.{num_workers}.{k}.zarr", mode="w")
                y.to_dataset(name="targets").to_zarr(f"{write_dir}/targets.{num_workers}.{k}.zarr", mode="w")
                write_elapsed += timer2.stop(None)

            elapsed = timer1.stop(f"Time with {num_workers} workers = ")
            elapsed -= write_elapsed
            avg_time[num_workers] = elapsed / num_tries
            avg_write_time[num_workers] = write_elapsed / num_tries

    print_time(batch_size, avg_time)
    print_time(batch_size, avg_write_time, work="write")

def remote_read_dask_distributed(p1, num_tries=10):

    timer1 = Timer()

    training_data = Dataset(
        p1,
        mode="training",
        preload_batch=False,
    )
    trainer = XBatchLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    iterloader = iter(trainer)
    avg_time = dict()

    for n_workers in [2, 4, 8, 16]:
        client = Client(n_workers=n_workers)
        timer1.start(f"{n_workers}")
        for _ in range(num_tries):
            next(iterloader)
        elapsed = timer1.stop(f"Time with {n_workers} workers = ")
        avg_time[n_workers] = elapsed / num_tries
        client.close()

    print_time(p1.batch_size, avg_time)


def local_read_test(p1, num_tries=10, read_dir=None, batch_size=None):
    """Find optimal number of dask worker threads to read a single batch of data"""
    timer1 = Timer()

    if read_dir is not None:
        p1.local_store_path = read_dir

    batch_size=p1.batch_size if batch_size is None else batch_size
    training_data = PackedDataset(
        p1,
        mode="training",
        chunks={
            "sample": 1,
            "lat": -1,
            "lon": -1,
            "channels": -1,
        },
    )
    trainer = BatchLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # --- What's the optimal number of dask worker threads to read a batch of data?
    iterloader = iter(trainer)
    avg_time = dict()
    for num_workers in [8, 16, 32]:#[1, 2, 4, 8, 16, 24, 32]:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            timer1.start()
            for _ in range(num_tries):
                next(iterloader)
            elapsed = timer1.stop(f"Time with {num_workers} workers = ")
            avg_time[num_workers] = elapsed / num_tries
        iterloader = iter(trainer)

    if read_dir is not None:
        print(f" Read time on {read_dir} ")
    print_time(batch_size, avg_time)


def local_tensorstore_test(p1, num_tries=10):
    """See notes.md, trying to add num_threads as GDM recommended quadruples the read time
    and I don't see any other way to improve the situation.

    dask is not used, so nothing to change there.
    """

    from graphufs.tensorstore import PackedDataset as TSPackedDataset, BatchLoader as TSBatchLoader

    timer1 = Timer()


    training_data = TSPackedDataset(
        p1,
        mode="training",
    )
    trainer = TSBatchLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    iterloader = iter(trainer)
    avg_time = dict()

    timer1.start()
    for _ in range(num_tries):
        next(iterloader)
    elapsed = timer1.stop(f"Time for tensorstore = ")
    avg_time[0] = elapsed / num_tries

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
    p1, args = P1Emulator.from_parser()

    #remote_read_test(p1, num_tries=5)
   # local_read_test(p1)
    local_tensorstore_test(p1)
   # remote_read_dask_distributed(p1, num_tries=5)
