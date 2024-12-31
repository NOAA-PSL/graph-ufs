import os
import logging

from mpi4py import MPI

from graphufs.datasets import Dataset
from graphufs.tensorstore import PackedDataset as TSPackedDataset, MPIBatchLoader as TSBatchLoader
from graphufs.mpi import MPITopology

from graphufs.stacked_mpi_training import (
    init_model,
    optimize,
)

from graphufs.optim import clipped_cosine_adamw
from graphufs.utils import get_last_input_mapping

from config import P3Trainer, P3Preprocessed
from prototypes.p3.train import train


def localtest(RemoteEmulator, PackedEmulator):

    # initial setup
    # Request thread support (using THREAD_MULTIPLE for full support)
    topo = MPITopology(log_dir=f"{RemoteEmulator.local_store_path}/logs/training")
    emulator = PackedEmulator(mpi_rank=topo.rank, mpi_size=topo.size)

    # data generators
    training_data = TSPackedDataset(emulator, mode="training")

    trainer = TSBatchLoader(
        training_data,
        batch_size=emulator.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=emulator.num_workers,
        max_queue_size=emulator.max_queue_size,
        mpi_topo=topo,
        rng_seed=10,
    )

    logging.info("no finalize")
    logging.info("packing thread query into mpitopo")
    logging.info("query not init")
    logging.info("this is new")
    logging.info("trying to get data...")
    inputs, targets = trainer.get_data()
    logging.info(f"inputs: \n{inputs}\n")
    trainer.shutdown()

if __name__ == "__main__":
    train(P3Trainer, P3Preprocessed)
