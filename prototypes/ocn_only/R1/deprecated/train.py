from mpi4py import MPI
from config import (
    OcnTrainer as RemoteEmulator,
    OcnPreprocessed as PackedEmulator,
)
from scripts.train_mpi import train

bad_samples = [2073, 2074,  5508, 5855, 5856,  5857,  5858, 5859, 8756,  8757,  8758,  8759,
               11695, 11696, 11697, 11698, 11699, 17524, 17525, 17526, 17527, 17535, 17536, 17537, 
               17538, 17539, 21105, 21106, 23371, 23372, 23373, 23374, 23375, 24828, 24829, 24830,
               24831, 25718, 32132, 32133, 32134, 32135, 33599, 33600, 33601, 33602, 33603, 35059, 
               35060, 35061, 35062, 35063, 35368]

if __name__ == "__main__":
    train(RemoteEmulator, PackedEmulator, missing_samples=bad_samples)
