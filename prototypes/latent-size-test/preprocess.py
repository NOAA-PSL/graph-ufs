import time
import logging
import os
import sys
import subprocess
import numpy as np
import dask

from graphufs.batchloader import XBatchLoader
from graphufs.datasets import Dataset
from graphufs.log import setup_simple_log
from config import LatentTestEmulator

_n_cpus = 48
_partition = "cpuD48v3"

def setup(mode, level=logging.INFO):

    setup_simple_log(level=level)

    p1 = LatentTestEmulator()
    tds = Dataset(
        p1,
        mode=mode,
        preload_batch=False,
        input_chunks={
            "sample": 1,
            "lat": -1,
            "lon": -1,
            "channels": 21,
        },
        target_chunks={
            "sample": 1,
            "lat": -1,
            "lon": -1,
            "channels": 22,
        },
    )
    loader = XBatchLoader(
        tds,
        batch_size=p1.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        max_queue_size=1,
    )
    dask.config.set(scheduler="threads", num_workers=p1.dask_threads)
    return p1, tds, loader


def submit_slurm_job():

    the_code = \
        f"from preprocess import store_batch_of_samples\n"+\
        f"store_batch_of_samples('training')\n" +\
        f"store_batch_of_samples('validation')\n"

    slurm_dir = f"slurm"
    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J preproc\n"+\
        f"#SBATCH -o {slurm_dir}/preprocess.%j.out\n"+\
        f"#SBATCH -e {slurm_dir}/preprocess.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks=1\n"+\
        f"#SBATCH --cpus-per-task={_n_cpus}\n"+\
        f"#SBATCH --partition={_partition}\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib2/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate graphufs-cpu2\n"+\
        f'python -c "{the_code}"'

    script_dir = "job-scripts"
    fname = f"{script_dir}/submit_stacked_preprocess.sh"

    for this_dir in [slurm_dir, script_dir]:
        if not os.path.isdir(this_dir):
            os.makedirs(this_dir)

    with open(fname, "w") as f:
        f.write(txt)

    subprocess.run(f"sbatch {fname}", shell=True)

def store_batch_of_samples(mode):

    p1, tds, loader = setup(mode)

    logging.info(f"Processing {len(loader)} in batch_size: {p1.batch_size}")

    for idx, (inputs, targets)  in enumerate(loader):

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

    logging.info(f"Done with mode {mode}")

def make_container(mode):

    p1, tds, _ = setup(mode)

    tds.store_containers()
    logging.info(f"Stored {mode} container")


if __name__ == "__main__":

    setup_simple_log()

    # create a container zarr store for all the data
    for mode in ["training", "validation"]:
        make_container(mode)

    # Pull the training and validation data and store to data/data.zarr
    # Do this in one slurm job because concurrent I/O on lustre is problematic
    submit_slurm_job()
