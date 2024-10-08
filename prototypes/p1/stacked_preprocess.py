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
from p1stacked import P1Emulator

from ufs2arco import Timer

_n_cpus = 48
_partition = "cpuD48v3"

def setup(mode, level=logging.INFO):

    setup_simple_log(level=level)

    p1 = P1Emulator()
    tds = Dataset(
        p1,
        mode=mode,
        preload_batch=False,
        chunks={
            "sample": 1,
            "lat": -1,
            "lon": -1,
            "channels": 13,
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
        f"from stacked_preprocess import store_batch_of_samples\n"+\
        f"store_batch_of_samples('training')\n" +\
        f"store_batch_of_samples('validation')\n"

    slurm_dir = f"slurm/stacked-preprocess"
    txt = "#!/bin/bash\n\n" +\
        f"#SBATCH -J spreproc\n"+\
        f"#SBATCH -o {slurm_dir}/%j.out\n"+\
        f"#SBATCH -e {slurm_dir}/%j.err\n"+\
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

        inputs = inputs.rename({"batch": "sample"}).chunk(tds.chunks)
        targets = targets.rename({"batch": "sample"}).chunk(tds.chunks)

        idx0 = int(inputs.sample.isel(sample=0))
        idx1 = int(inputs.sample.isel(sample=-1))
        logging.info(f" ... storing indices {idx0} - {idx1}")
        spatial_region = {k : slice(None, None) for k in inputs.dims if k != "sample"}
        region = {"sample": slice(idx0, idx1+1), **spatial_region}

        for name, xda, path in zip(
            ["inputs", "targets"],
            [inputs, targets],
            [tds.local_inputs_path, tds.local_targets_path],
        ):

            xda.to_dataset(name=name).to_zarr(path, region=region)

        if idx % 10 == 0:
            logging.info(f"Done with index {idx} / {len(loader)}")

    logging.info(f"Done with mode {mode}")

def store_one_by_one(jid, n_jobs, mode):

    p1, tds, loader = setup(mode)

    index_chunks = np.linspace(0, len(tds), n_jobs+1)
    start = int(index_chunks[jid])
    end = int(index_chunks[jid+1])
    logging.info(f"Job = {jid} / {n_jobs}")
    logging.info(f"Processing indices: {start} - {end}")

    for idx in range(start, end):
        try:
            tds.store_sample(idx)
        except RuntimeError:
            logging.error(f" *** Runtime error with sample {idx} *** ")

        if idx % 10 == 0:
            logging.info(f"Done with sample {idx}")

    logging.info("Done with my batch")

def fill_some_indices():
    """
    Note:
        This was a hack to fill "problem indices" - basically some indices would fail during the initial
        preprocessing, using multiple slurm jobs to run `store_one_by_one` concurrently.
        This function would supposedly fill the rest of the job's indices, but this didn't end up working too well.
    """


    p1, tds, _ = setup("training")

    indices = np.concatenate([
        np.arange(500, 2921), # etc
    ])
    for idx in indices:
        idx_int = int(idx)
        logging.info(f"Storing sample {idx_int}")
        try:
            tds.store_sample(idx_int)
            logging.info(f" ... Done with sample {idx_int}")
        except RuntimeError:
            logging.error(f" *** Runtime error with sample {idx_int} *** ")


def make_container(mode):

    p1, tds, _ = setup(mode)

    tds.store_containers()
    logging.info(f"Stored {mode} container")


if __name__ == "__main__":

    setup_simple_log()
    timer = Timer()

    # create a container zarr store for all the data
    for mode in ["training", "validation"]:
        make_container(mode)

    # Pull the training and validation data and store to data/data.zarr
    # Do this in one slurm job because concurrent I/O on lustre is problematic
    submit_slurm_job()
