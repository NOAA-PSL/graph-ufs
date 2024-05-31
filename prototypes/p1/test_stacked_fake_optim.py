import logging
import sys
import time
import numpy as np
import dask

from graphufs.datasets import PackedDataset
from graphufs.batchloader import BatchLoader
from graphufs.torch import Dataset as TorchDataset, DataLoader as TorchDataLoader

from ufs2arco import Timer

from p1stacked import P1Emulator

class Formatter(logging.Formatter):
    def __init__(self, fmt):
        super().__init__(fmt)

    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def test_local_generator(p1, max_iters=30):

    training_data = PackedDataset(
        p1,
        mode="training",
    )
    trainer = BatchLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=p1.num_workers,
        max_queue_size=p1.max_queue_size,
    )

    setup_time = 3 if p1.batch_size > 4 else 1
    setup_time *= p1.max_queue_size
    setup_time += 5
    logging.info(f"Initial Setup Time = {setup_time} sec")
    time.sleep(setup_time)
    logging.info(f"... done, starting the loop with qsize = {trainer.data_queue.qsize()}")

    qsize = []
    alltimes = []
    n_iter = min(max_iters, len(trainer))
    timer1.start()
    for k, (x,y) in zip(range(n_iter), iter(trainer)):
        time.sleep(1.0)
        elapsed = timer1.stop(f"{k} / {n_iter}, qsize = {trainer.data_queue.qsize()}")

        alltimes.append(elapsed)
        qsize.append(trainer.data_queue.qsize())
        timer1.start()

    logging.info(f"num_workers = {p1.num_workers}, max_queue_size = {p1.max_queue_size}")
    logging.info(f"Avg Time per iteration = {np.mean(alltimes)} seconds")
    logging.info(f"First empty queue iteration = {qsize.index(0)}")

    trainer.shutdown()

def test_local_torchloader(p1, prefetch_factor, num_workers, max_iters=30):

    training_data = TorchDataset(
        p1,
        mode="training",
    )
    trainer = TorchDataLoader(
        training_data,
        batch_size=p1.batch_size,
        shuffle=False,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers
    )

    logging.info(f"Initial Setup")
    _ = next(iter(trainer))
    logging.info(f"... done, starting the loop")

    alltimes = []
    n_iter = min(max_iters, len(trainer))
    timer1.start()
    for k, (x,y) in zip(range(n_iter), enumerate(trainer)):
        time.sleep(1.0)
        elapsed = timer1.stop(f"{k} / {n_iter}")
        alltimes.append(elapsed)
        timer1.start()

    timer1.stop()
    logging.info(f"num_workers = {num_workers}, prefetch_factor = {prefetch_factor}")
    logging.info(f"Avg Time per iteration = {np.mean(alltimes)}")


if __name__ == "__main__":

    dask.config.set(
        scheduler="threads",
        num_workers=16,
    )

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

    test_local_generator(p1)
