from mpi4py import MPI
from config import (
    AtmTrainer as RemoteEmulator,
    AtmPreprocessed as PackedEmulator,
)

from prototypes.atm_only.train_mpi import train

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator)
