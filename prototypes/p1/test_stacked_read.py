import logging
import sys
import numpy as np

import dask

from graphufs.datasets import PackedDataset
from graphufs.batchloader import BatchLoader
from graphufs.tensorstore import PackedDataset as TSPackedDataset, BatchLoader as TSBatchLoader

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

def print_time(batch_size, avg_time):
    print(f" --- Time to read batch_size = {p1.batch_size} --- ")
    print(f"\tnum_workers\t avg seconds / batch")
    for key, val in avg_time.items():
        print(f"\t{key}\t\t{val}")

def local_read_test(p1, num_tries=10):
    """Find optimal number of dask worker threads to read a single batch of data"""

    training_data = PackedDataset(
        p1,
        mode="training",
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
    for num_workers in [1, 2, 4, 8, 16, 24, 32]:
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            timer1.start()
            for _ in range(num_tries):
                next(iterloader)
            elapsed = timer1.stop(f"Time with {num_workers} workers = ")
            avg_time[num_workers] = elapsed / num_tries

    print_time(p1.batch_size, avg_time)


def local_tensorstore_test(p1, num_tries=10):
    """See notes.md, trying to add num_threads as GDM recommended quadruples the read time
    and I don't see any other way to improve the situation.

    dask is not used, so nothing to change there.
    """

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
    local_tensorstore_test(p1)
