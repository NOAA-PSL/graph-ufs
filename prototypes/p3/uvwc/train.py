from mpi4py import MPI
from config import (
    P3Trainer as RemoteEmulator,
    P3Preprocessed as PackedEmulator,
)

from prototypes.p3.train import train

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator)
