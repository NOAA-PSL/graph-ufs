import logging
import os
import sys
import subprocess
import numpy as np

from graphufs.datasets import Dataset
from p1 import P1Emulator

from ufs2arco import Timer

_n_cpus = 48
_partition = "cpuD48v3-spot"

class SimpleFormatter(logging.Formatter):
    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def setup(mode):

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    p1 = P1Emulator()
    tds = Dataset(
        p1,
        mode=mode,
        preload_batch=True,
        chunks={
            "sample": 1,
            "lat": -1,
            "lon": -1,
            "channels": 13,
        },
    )
    return p1, tds


def submit_slurm_job(job_id, n_jobs, mode):

    the_code = \
        f"from stacked_preprocess import store_batch_of_samples\n"+\
        f"store_batch_of_samples({job_id}, {n_jobs}, '{mode}')\n"

    slurm_dir = f"slurm/stacked-preprocess/{mode}"
    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J sp{mode[0]}{job_id:03d}\n"+\
        f"#SBATCH -o {slurm_dir}/{job_id:03d}.%j.out\n"+\
        f"#SBATCH -e {slurm_dir}/{job_id:03d}.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks=1\n"+\
        f"#SBATCH --cpus-per-task={_n_cpus}\n"+\
        f"#SBATCH --partition={_partition}\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib2/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate graphufs-cpu\n"+\
        f'python -c "{the_code}"'

    script_dir = "job-scripts"
    fname = f"{script_dir}/submit_sp_{mode}_{job_id:03d}.sh"

    for this_dir in [slurm_dir, script_dir]:
        if not os.path.isdir(this_dir):
            os.makedirs(this_dir)

    with open(fname, "w") as f:
        f.write(txt)

    subprocess.run(f"sbatch {fname}", shell=True)

def store_batch_of_samples(jid, n_jobs, mode):

    p1, tds = setup(mode)

    index_chunks = np.linspace(0, len(tds), n_jobs+1)
    start = int(index_chunks[jid])
    end = int(index_chunks[jid+1])
    logging.info(f"Job = {jid} / {n_jobs}")
    logging.info(f"Processing indices: {start} - {end}")


    datachunk_indices = np.linspace(0, len(tds.xds.datetime), n_jobs+1)
    cst = int(datachunk_indices[jid]-1)
    cst = max(cst, 0)
    ced = int(datachunk_indices[jid+1]+1)
    logging.info(f"Loading data time indices {cst} - {ced}")
    tds.xds.isel(datetime=slice(cst, ced)).load()
    logging.info("Starting my batch...")
    for idx in range(start, end):
        tds.store_sample(idx)
        if idx % 10 == 0:
            logging.info(f"Done with sample {idx}")

    logging.info("Done with my batch")


def make_container(mode):

    p1, tds = setup(mode)

    tds.store_containers()
    logging.info(f"Stored {mode} container")


if __name__ == "__main__":

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    timer = Timer()

    # create a container zarr store for all the data
    for mode in ["training", "validation"]:
        make_container(mode)

    # Pull the training and validation data and store to data/data.zarr
    n_jobs = 26*2 # 2 jobs per year
    for jid in range(n_jobs):
        submit_slurm_job(jid, n_jobs, mode="training")

    n_jobs_valid = 2*2 # 2 jobs per year
    for jid in range(n_jobs_valid):
        submit_slurm_job(jid, n_jobs_valid, mode="validation")
