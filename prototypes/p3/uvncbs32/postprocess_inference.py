from mpi4py import MPI
from prototypes.p3.postprocess_inference import main

from config import P3Evaluator

if __name__ == "__main__":
    main(P3Evaluator)
