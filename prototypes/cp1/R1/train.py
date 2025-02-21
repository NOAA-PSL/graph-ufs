from mpi4py import MPI
from config import (
    CP1Trainer as RemoteEmulator,
    CP1Preprocessed as PackedEmulator,
)

from prototypes.cp1.train_mpi import train

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator)
