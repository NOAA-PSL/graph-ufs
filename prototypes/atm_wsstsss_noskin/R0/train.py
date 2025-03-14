from mpi4py import MPI
from config import (
    AtmSSTSSSTrainer as RemoteEmulator,
    AtmSSTSSSPreprocessed as PackedEmulator,
)

from scripts.train_mpi import train

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator)
