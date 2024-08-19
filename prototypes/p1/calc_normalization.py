import os
import xarray as xr
import subprocess

from graphufs import StatisticsComputer, add_derived_vars

def main(varname):

    path_in = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    open_zarr_kwargs = {"storage_options": {"token": "anon"}}
    ds = xr.open_zarr(path_in, **open_zarr_kwargs)
    ds = add_derived_vars(ds)
    load_full_dataset = "pfull" not in ds[varname].dims

    normer = StatisticsComputer(
        path_in=path_in,
        path_out="gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019",
        start_date=None, # original start date
        end_date="2019",
        time_skip=None,
        load_full_dataset=load_full_dataset,
        open_zarr_kwargs=open_zarr_kwargs,
        to_zarr_kwargs={
            "mode":"a",
            "storage_options": {"token": "/contrib/Tim.Smith/.gcs/replay-service-account.json"},
        },
    )
    normer(varname)


def submit_slurm_job(varname, partition="compute"):

    logdir = "slurm/normalization"
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
        f"#SBATCH --cpus-per-task=30\n"+\
        f"#SBATCH --partition={partition}\n"+\
        f"#SBATCH -t 120:00:00\n\n"+\
        f"source /contrib/Tim.Smith/miniconda3/etc/profile.d/conda.sh\n"+\
        f"conda activate graphufs-cpu\n"+\
        f"python -c 'from calc_normalization import main ; main(\"{varname}\")'"

    scriptname = f"{scriptdir}/submit_normalization_{varname}.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)

    subprocess.run(f"sbatch {scriptname}", shell=True)

if __name__ == "__main__":

    path_in = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    ds = xr.open_zarr(
        path_in,
        storage_options={"token": "anon"},
    )
    ds = add_derived_vars(ds)

    for key in ds.data_vars:
        submit_slurm_job(key)
