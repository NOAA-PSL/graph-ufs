from mpi4py import MPI
from config import (
    AtmOcnTrainer as RemoteEmulator,
    AtmOcnPreprocessed as PackedEmulator,
)

from scripts.train_mpi import train

bad_samples = [5860, 5861, 5862, 8761, 8762, 11700, 11701, 11702, 17529,
               17530, 17540, 17541, 17542, 23376, 23377, 23378, 24833, 24834,
               31848, 32137, 32138, 33604, 33605, 33606, 35064, 35065, 35066,
               35068, 35069, 36904, 36905, 36912, 36925, 36936, 36940, 36944,
               36952, 36977, 36980, 36981, 36985]

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator, missing_samples=bad_samples)
