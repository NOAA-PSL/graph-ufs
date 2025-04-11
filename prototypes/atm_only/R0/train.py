from mpi4py import MPI
from config import (
    Trainer as RemoteEmulator,
    Preprocessed as PackedEmulator,
)

from scripts.train_mpi import train

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator)
