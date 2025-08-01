from mpi4py import MPI
import time
import logging
import os
import sys
import subprocess
import numpy as np
import dask
import yaml
import argparse

from omegaconf import OmegaConf
from graphufs.batchloader import XBatchLoader, MPIXBatchLoader
from graphufs.datasets import Dataset
from graphufs.log import setup_simple_log
from graphufs.progress import ProgressTracker
from graphufs.mpi import MPITopology

# in the future this could be generalized to where it just takes the following as inputs
from emulator import OcnPreprocessor

def setup(mode, emulator, config, topo, level=logging.INFO):

    if topo.is_root:
        pt = ProgressTracker(json_file_path=f"{topo.log_dir}/restart.{topo.rank:02d}.{topo.size:02d}.json")
    topo.comm.barrier()
    if not topo.is_root:
        pt = ProgressTracker(json_file_path=f"{topo.log_dir}/restart.{topo.rank:02d}.{topo.size:02d}.json")
    start = pt.get_current_iteration()
    tds = Dataset(
        emulator,
        mode=mode,
        preload_batch=False,
        input_chunks={
            "sample": 1,
            "lat": -1,
            "lon": -1,
            "channels": config.input_channel_chunks,
        },
        target_chunks={
            "sample": 1,
            "lat": -1,
            "lon": -1,
            "channels": config.target_channel_chunks,
        },
    )
    loader = MPIXBatchLoader(
        tds,
        batch_size=emulator.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        max_queue_size=1,
        start=start,
        mpi_topo=topo,
    )
    return tds, loader, pt


def submit_slurm_job(config, prototype):

    the_code = \
        f"from preprocess import store_batch_of_samples\n"+\
        f"store_batch_of_samples('training', '{prototype}')\n" +\
        f"store_batch_of_samples('validation', '{prototype}')\n"

    slurm_dir = f"{config.local_store_path}/slurm"
    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J preprocess_ocn_only\n"+\
        f"#SBATCH -o {slurm_dir}/preprocess.%j.out\n"+\
        f"#SBATCH -e {slurm_dir}/preprocess.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks={config.batch_size}\n"+\
        f"#SBATCH --cpus-per-task={config.cpus_per_node // config.batch_size}\n"+\
        f"#SBATCH --qos={config.qos}\n"+\
        f"#SBATCH --account=m4718\n"+\
        f"#SBATCH --constraint=cpu\n"+\
        f"#SBATCH -t {config.walltime}\n\n"+\
        f"conda activate graphufs-mpi\n"+\
        f'srun python -c "{the_code}"'

    script_dir = "job-scripts"
    fname = f"{script_dir}/submit_stacked_preprocess.sh"

    for this_dir in [slurm_dir, script_dir]:
        if not os.path.isdir(this_dir):
            os.makedirs(this_dir)

    with open(fname, "w") as f:
        f.write(txt)

    if config.n_jobs > 1:
        runstr = "`for i in {1.."+f"{config.n_jobs}"+"}; do sbatch --job-name=preproc --dependency=singleton "+fname+"; done`"
    else:
        runstr = f"sbatch {fname}"
    subprocess.run(runstr, shell=True)

def store_batch_of_samples(mode, prototype):

    # Build config paths
    config_trainer_path = f"./{prototype}/config.yaml"
    config_preprocess_path = f"./{prototype}/config_preprocessor.yaml"

    # Load config and initialize the emulator
    trainer_config = OmegaConf.load(config_trainer_path)
    preprocessor_config = OmegaConf.load(config_preprocess_path)

    topo = MPITopology(log_dir=f"{trainer_config.local_store_path}/logs/preprocessing-{mode}")
    emulator = OcnPreprocessor(prototype, mpi_rank=topo.rank, mpi_size=topo.size)
    
    tds, loader, pt = setup(mode, emulator, preprocessor_config, topo)

    start = pt.get_current_iteration()

    logging.info(f"Processing {len(loader)} in batch_size: {emulator.batch_size}")
    logging.info(f"Starting at idx = {start}")
    logging.info(f"loader.counter = {loader.counter}")
    logging.info(f"loader.data_counter = {loader.data_counter}")

    for idx in range(start, len(loader)):

        inputs, targets = next(loader)

        if inputs is not None:
            inputs = inputs.rename({"batch": "sample"}).chunk(tds.input_chunks)
            targets = targets.rename({"batch": "sample"}).chunk(tds.target_chunks)

            idx0 = int(inputs.sample.isel(sample=0))
            idx1 = int(inputs.sample.isel(sample=-1))
            logging.info(f" ... storing sample indices {idx0} - {idx1}")
            spatial_region = {k : slice(None, None) for k in inputs.dims if k != "sample"}
            region = {"sample": slice(idx0, idx1+1), **spatial_region}

            for name, xda, path in zip(
                ["inputs", "targets"],
                [inputs, targets],
                [tds.local_inputs_path, tds.local_targets_path],
            ):

                xda.to_dataset(name=name).to_zarr(path, region=region)

        if idx % 10 == 0:
            logging.info(f"Done with batch {idx} / {len(loader)}")

        pt.update_progress(idx+1)
        logging.info(f"    pt.current_iteration = {pt.current_iteration}")
        logging.info(f"    loader.counter = {loader.counter}")
        logging.info(f"    loader.data_counter = {loader.data_counter}")

    logging.info(f"Done with mode {mode}")

def make_container(mode, emulator, config, topo):

    tds, _, _ = setup(mode, emulator, config, topo)

    tds.store_containers()
    logging.info(f"Stored {mode} container")

parser = argparse.ArgumentParser(description="Ocn-only Preprocessing")
parser.add_argument("--prototype", required=True, help="Prototype Name")  # e.g., "R1"
parser.add_argument("--dt", help="Model time step")  # e.g., 6h

if __name__ == "__main__":

    setup_simple_log()

    args = parser.parse_args()
    prototype = args.prototype

    # Build config paths
    config_trainer_path = f"./{prototype}/config.yaml"
    config_preprocess_path = f"./{prototype}/config_preprocessor.yaml"

    # Initialize the emulator
    trainer_config = OmegaConf.load(config_trainer_path)
    preprocessor_config = OmegaConf.load(config_preprocess_path)
    merged_config = OmegaConf.merge(trainer_config, preprocessor_config)
    
    # create a container zarr store for all the data
    for mode in ["training", "validation"]:
        topo = MPITopology(log_dir=f"{trainer_config.local_store_path}/logs/preprocessing-{mode}")
        emulator = OcnPreprocessor(prototype, mpi_rank=topo.rank, mpi_size=topo.size)
        make_container(mode, emulator, preprocessor_config, topo)

    # Pull the training and validation data and store to data/data.zarr
    # Do this in one slurm job because concurrent I/O on lustre is problematic 
    submit_slurm_job(merged_config, prototype)
