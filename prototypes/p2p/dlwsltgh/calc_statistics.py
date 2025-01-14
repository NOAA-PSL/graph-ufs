import logging
import os
import subprocess

import xarray as xr

from graphufs.log import setup_simple_log
from graphufs.fvstatistics import FVStatisticsComputer

from config import P2PTrainer as Emulator


def submit_slurm_job(varname, partition="compute", n_cpus=30):

    logdir = "slurm/fvstats"
    scriptdir = "job-scripts"
    for d in [logdir, scriptdir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    jobscript = f"#!/bin/bash\n\n"+\
        f"#SBATCH -J {varname}_norm\n"+\
        f"#SBATCH -o {logdir}/{varname}.%j.out\n"+\
        f"#SBATCH -e {logdir}/{varname}.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --ntasks=1\n"+\
        f"#SBATCH --cpus-per-task={n_cpus}\n"+\
        f"#SBATCH --partition={partition}\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate graphufs-cpu\n"+\
        f"echo $PYTHONPATH\n"+\
        f"python -c 'from calc_statistics import main ; main(\"{varname}\")'"

    scriptname = f"{scriptdir}/submit_statistics_{varname}.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)

    subprocess.run(f"sbatch {scriptname}", shell=True)

def main(varname):

    setup_simple_log()

    # if it's a surface variable, then read it from existing stats and use that
    # otherwise, 3D, need to calculate the FV version
    path_out = os.path.dirname(Emulator.norm_urls["mean"])
    gcs_stats = lambda prefix : f"gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/{prefix}_by_level.zarr"

    open_zarr_kwargs = {
        "storage_options": {"token": "anon"},
    }

    to_zarr_kwargs = {
        "mode": "a",
        "storage_options": {"token": "/contrib/Tim.Smith/.gcs/replay-service-account.json"},
    }

    ds = xr.open_zarr(gcs_stats("mean"), **open_zarr_kwargs)
    do_fv_calc = True
    if varname in ds:
        if "pfull" not in ds[varname].dims:
            do_fv_calc = False

    if do_fv_calc:
        logging.info(f"Need to calculate statistics for {varname}")
        end_date = "2019-12-31T18"
        logging.warning(f"{__name__}.calc_statistics.main: note a detail here, P2P script used up to Dec 31 2019 @ 18z for stats, but this is just 3h shy of what we use for training. So the end_date is hard coded in this script.\n\nI am keeping this hard coded date so that we can continue to append to that same zarr store without error.")
        fvstats = FVStatisticsComputer(
            path_in=Emulator.data_url,
            path_out=path_out,
            interfaces=Emulator.interfaces,
            start_date=None,
            end_date=end_date,
            time_skip=None,
            load_full_dataset=False,
            transforms=Emulator.input_transforms,
            open_zarr_kwargs=open_zarr_kwargs,
            to_zarr_kwargs=to_zarr_kwargs,
        )

        fvstats(data_vars=list(), diagnostics=varname)

    else:

        store_path = lambda prefix : f"{path_out}/{prefix}_by_level.zarr"
        for prefix in ["mean", "stddev", "diffs_stddev"]:
            ds = xr.open_zarr(gcs_stats(prefix), **open_zarr_kwargs)
            if varname in ds:
                ds = ds[[varname]]
                ds.to_zarr(store_path(prefix), **to_zarr_kwargs)
                logging.info(f"Pulled {varname} {prefix} from {gcs_stats(prefix)} to {store_path(prefix)}")


if __name__ == "__main__":

    for key in Emulator.diagnostics:
        submit_slurm_job(key)
