from mpi4py import MPI
from config import (
    P2PTrainer as RemoteEmulator,
    P2PPreprocessed as PackedEmulator,
)

from prototypes.p2p.train import train

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator)
