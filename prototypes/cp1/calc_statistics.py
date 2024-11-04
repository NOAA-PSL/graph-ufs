import logging
import os
import subprocess

import xarray as xr

from graphufs.log import setup_simple_log
from graphufs.fvstatistics import FVStatisticsComputer

from config import CP1Emulator as Emulator


def submit_slurm_job(varname, comp="atm", apartition="compute", n_cpus=30):

    logdir = "slurm/fvstats"
    scriptdir = "job-scripts"
    for d in [logdir, scriptdir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    #jobscript = f"#!/bin/bash\n\n"+\
    #    f"#SBATCH -J {varname}_norm\n"+\
    #    f"#SBATCH -o {logdir}/{varname}.%j.out\n"+\
    #    f"#SBATCH -e {logdir}/{varname}.%j.err\n"+\
    #    f"#SBATCH --nodes=1\n"+\
    #    f"#SBATCH --ntasks=1\n"+\
    #    f"#SBATCH --cpus-per-task={n_cpus}\n"+\
    #    f"#SBATCH --partition={apartition}\n"+\
    #    f"#SBATCH -t 120:00:00\n\n"+\
    #    f"source /contrib/niraj.agarwal/miniconda3/etc/profile.d/conda.sh\n"+\
    #    f"conda activate graphufs\n"+\
    #    f"echo $PYTHONPATH\n"+\
    #    f"python -c 'from calc_statistics import main ; main(\"{varname}\", \"{comp}\")'"

    # On PSL Cluster
    jobscript = f"#!/bin/bash\n\n"+\
            f"source activate base\n"+\
            f"conda activate graphufs\n\n"+\
            f"python -c 'from calc_statistics import main ; main(\"{varname}\", \"{comp}\")'"

    scriptname = f"{scriptdir}/submit_statistics_{varname}.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)

    subprocess.run(f"chmod a+x {scriptname}", shell=True)
    subprocess.run(f"{scriptname}", shell=True)

def main(varname, comp="atm"):

    setup_simple_log()

    # if it's a surface variable, then try reading it from existing stats. If no 
    # stats exist, then compute everything. If 3D, compute the FV version.
    
    #path_out = os.path.dirname(Emulator.norm_urls[comp]["mean"])
    statsdir = "coupled.statistics.1994-2019"
    if not os.path.isdir(statsdir):
        os.makedirs(statsdir)
    path_out = os.path.abspath(statsdir)

    if comp in ["atm", "ice", "land"]:
        gcs_existing_stats = lambda prefix: f"gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.statistics.1993-2019/{prefix}_by_level.zarr"
        time_skip = 2 # everything is in 3 hour time steps in fv3 stats

    elif comp.lower() == "ocn".lower():
        gcs_existing_stats = lambda prefix: f"gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.statistics.1993-2019/{prefix}_by_level.zarr"
        time_skip = 1 # time step is 6 hour for the oceans  
    else:
        raise ValueError("comp values can only be atm, ocn, ice, or land")
    
    open_zarr_kwargs = {
        "storage_options": {"token": "anon"},
    }

    to_zarr_kwargs = {
        "mode": "a",
    }

    does_it_exist = True
    do_fv_calc = True

    # check whether the specified GCS zarr store exists
    try:
        ds = xr.open_zarr(gcs_existing_stats("mean"), **open_zarr_kwargs)
    except:
        does_it_exist = False
    
    if does_it_exist:
        if varname in ds:
            if "pfull" not in ds[varname].dims:
                do_fv_calc = False
        
        if do_fv_calc:
            logging.info(f"Need to calculate statistics for {varname}")
            fvstats = FVStatisticsComputer(
                path_in=Emulator.data_url[comp],
                path_out=path_out,
                comp=comp,
                interfaces=Emulator.interfaces[comp],
                start_date=None,
                end_date=Emulator.training_dates[-1],
                time_skip=time_skip,
                load_full_dataset=False,
                transforms=Emulator.input_transforms,
                open_zarr_kwargs=open_zarr_kwargs,
                to_zarr_kwargs=to_zarr_kwargs
            )

            fvstats(varname)

        else:
            gcs_store_stats = lambda prefix: Emulator.norm_urls[comp][prefix]
            for prefix in ["mean", "stddev", "diffs_stddev"]:
                ds = xr.open_zarr(gcs_existing_stats(prefix), **open_zarr_kwargs)
                if varname in ds:
                    ds = ds[[varname]]
                    ds.to_zarr(gcs_store_stats(prefix), **to_zarr_kwargs)
                    logging.info(f"Pulled {varname} {prefix} from {gcs_existing_stats(prefix)} to {gcs_store_stats(prefix)}")

    else:
        logging.info(f"No prior statistics exist. Computing statistics for {varname}")
        fvstats = FVStatisticsComputer(
                path_in=Emulator.data_url[comp],
                path_out=path_out,
                comp=comp,
                interfaces=Emulator.interfaces[comp],
                start_date=None,
                end_date=Emulator.training_dates[-1],
                time_skip=time_skip,
                load_full_dataset=False,
                transforms=Emulator.input_transforms,
                open_zarr_kwargs=open_zarr_kwargs,
                to_zarr_kwargs=to_zarr_kwargs
        )
        fvstats(varname)


if __name__ == "__main__":
    comp = "land"
    all_variables = list(set(
        # uncomment one of the below lines based on the component chosen. This would be automated in the future.
        #Emulator.atm_input_variables + Emulator.atm_forcing_variables + Emulator.atm_target_variables
        #Emulator.ocn_input_variables + Emulator.ocn_forcing_variables + Emulator.ocn_target_variables
        #Emulator.ice_input_variables + Emulator.ice_forcing_variables + Emulator.ice_target_variables
        Emulator.land_input_variables + Emulator.land_forcing_variables + Emulator.land_target_variables
    ))
    
    if comp.lower() == "atm".lower(): 
        all_variables.append("log_spfh")
        all_variables.append("log_spfh2m")

    for key in all_variables:
        submit_slurm_job(key, comp=comp)
