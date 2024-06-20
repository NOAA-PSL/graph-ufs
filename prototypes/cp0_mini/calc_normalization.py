import xarray as xr
import subprocess

from graphufs import StatisticsComputer, add_derived_vars

def main(varname, comp="atm"):

    path_in = "gs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/mom6.zarr"
    open_zarr_kwargs = {"storage_options": {"token": "anon"}}
    ds = xr.open_zarr(path_in, **open_zarr_kwargs)
    ds = add_derived_vars(ds,"ocean")
    load_full_dataset = "z_l" not in ds[varname].dims

    normer = StatisticsComputer(
        path_in=path_in,
        path_out="/home/nagarwal/work/mom6.statistics.1993-2023",
        comp = comp,
        start_date=None, # original start date
        end_date="2023",
        time_skip=1,
        load_full_dataset=load_full_dataset,
        open_zarr_kwargs=open_zarr_kwargs,
        to_zarr_kwargs={
            "mode":"a",
            #"storage_options": {"token": "/contrib/Tim.Smith/.gcs/replay-service-account.json"},
        },
    )
    normer(varname)


def compute_stats(varname, comp="atm"):
    
    # On GCP
    #jobscript = f"#!/bin/bash\n\n"+\
    #    f"#SBATCH -J {varname}_norm\n"+\
    #    f"#SBATCH -o slurm/normalization/{varname}.%j.out\n"+\
    #    f"#SBATCH -e slurm/normalization/{varname}.%j.err\n"+\
    #    f"#SBATCH --nodes=1\n"+\
    #    f"#SBATCH --ntasks=1\n"+\
    #    f"#SBATCH --cpus-per-task=30\n"+\
    #    f"#SBATCH --partition={partition}\n"+\
    #    f"#SBATCH -t 120:00:00\n\n"+\
    #    f"source /contrib/niraj.agarwal/miniconda3/etc/profile.d/conda.sh\n"+\
    #    f"conda activate graphufs\n"+\
    #    f"python -c 'from calc_normalization import main ; main(\"{varname}\")'"
    
    # On PSL Cluster
    jobscript = f"#!/bin/bash\n\n"+\
            f"source /opt/conda/bin/conda.sh\n"+\
            f"conda activate graphufs\n\n"+\
            f"python -c 'from calc_normalization import main ; main(\"{varname}\", \"{comp}\")'"
    
    scriptname = f"job-scripts/compute_normalization_{varname}.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)
    
    subprocess.run(f"chmod a+x {scriptname}", shell=True)
    subprocess.run(f"{scriptname}", shell=True)

if __name__ == "__main__":

    path_in = "gs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/mom6.zarr"
    comp = "ocean"
    ds = xr.open_zarr(
        path_in,
        storage_options={"token": "anon"},
    )
    ds = add_derived_vars(ds, comp)

    for key in ds.data_vars:
        compute_stats(key, comp)
