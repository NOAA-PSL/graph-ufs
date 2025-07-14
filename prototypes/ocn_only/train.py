import argparse
import os
import warnings
import yaml
from omegaconf import OmegaConf
from mpi4py import MPI
from emulator import OcnTrainer, OcnPreprocessed
from scripts.train_mpi_no_init import train
from graphufs.mpi import MPITopology

parser = argparse.ArgumentParser(description="Ocn-only Training")
parser.add_argument("--prototype", required=True, help="Prototype Name")  # e.g., "R1"
parser.add_argument("--dt", help="Model time step")  # e.g., 6h

if __name__ == "__main__":
    args = parser.parse_args()
    prototype = args.prototype

    # Build config paths
    config_trainer_path = f"./{prototype}/config.yaml"
    bad_samples_path = f"./{prototype}/bad_samples.yaml"

    # Initialize trainers
    trainer_config = OmegaConf.load(config_trainer_path)
    topo = MPITopology(log_dir=f"{trainer_config.local_store_path}/logs/training")
    remote_emulator = OcnTrainer(config=config_trainer_path, mpi_rank=topo.rank, mpi_size=topo.size)
    emulator = OcnPreprocessed(prototype, mpi_rank=topo.rank, mpi_size=topo.size)

    # Load bad samples list if available
    found = False
    if os.path.exists(bad_samples_path):
        found = True
    elif args.dt and os.path.exists(f"./bad_samples_{args.dt}.yaml"):
        bad_samples_path = f"./bad_samples_{args.dt}.yaml"
        found = True 
    
    # Train
    if found:
        with open(bad_samples_path, "r") as f:
            bad_samples_dict = yaml.safe_load(f)
        bad_samples = bad_samples_dict.get("bad_samples")
        train(remote_emulator, emulator, topo, missing_samples=bad_samples)
    else:
        warnings.warn("No bad samples list found. Continuing without excluding any sample.")
        train(remote_emulator, emulator, topo)
