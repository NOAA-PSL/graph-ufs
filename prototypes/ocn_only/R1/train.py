from mpi4py import MPI
from config import (
    OcnTrainer as RemoteEmulator,
    OcnPreprocessed as PackedEmulator,
)

from scripts.train_mpi import train

missing_samples = [5861, 17541, 17542, 23377, 23378, 33605, 33606, 35065, 35066, 35069]

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator, missing_samples=missing_samples)
