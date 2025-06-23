import logging
import os
import subprocess

import xarray as xr
import pandas as pd

from graphufs.log import setup_simple_log
from graphufs.fvstatistics import FVStatisticsComputer

from R0.config import AtmOcnSfcLandIceTrainer as Emulator

def submit_slurm_job(varname, comp="atm", apartition="compute", n_cpus=30):

    scriptdir = os.path.join(os.getcwd(), "job-scripts")
    expt = "atm_wocnsfc_land_ice"
    for d in [scriptdir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    # On Perlmutter
    slurm_dir = f"{Emulator.local_store_path}/slurm/fvstats"
    jobscript = "#!/bin/bash\n\n" +\
        f"#SBATCH -J calc_statistics\n"+\
        f"#SBATCH -o {slurm_dir}/calc_stats.%j.out\n"+\
        f"#SBATCH -e {slurm_dir}/calc_stats.%j.err\n"+\
        f"#SBATCH --nodes=1\n"+\
        f"#SBATCH --qos=debug\n"+\
        f"#SBATCH --account=m4718\n"+\
        f"#SBATCH --constraint=cpu\n"+\
        f"#SBATCH -t 00:30:00\n\n"+\
        f"conda activate graphufs-mpi\n"+\
        f"cd /global/homes/n/nagarwal/graph-ufs/prototypes/{expt}\n\n"+\
        f"python -c 'from calc_statistics import main ; main(\"{varname}\", \"{comp}\")'"

    #jobscript = f"#!/bin/bash\n\n"+\
    #        f"source activate base\n"+\
    #        f"conda activate graphufs\n\n"+\
    #        f"python -c 'from calc_statistics import main ; main(\"{varname}\", \"{comp}\")'"

    scriptname = f"{scriptdir}/submit_statistics_{varname}.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)

    #subprocess.run(f"chmod a+x {scriptname}", shell=True)
    #subprocess.run(f"{scriptname}", shell=True)
    #subprocess.run(f"sbatch {scriptname}", shell=True)

def main(varname, comp):

    setup_simple_log()

    # if it's a surface variable, then try reading it from existing stats. If no 
    # stats exist, then compute everything. If 3D, compute the FV version.
    
    statsdir = f"{comp}.fvstatistics.1993-2019"
    if not os.path.isdir(statsdir):
        os.makedirs(statsdir)
    path_out = os.path.abspath(statsdir)

    if comp in ["atm", "ice", "land"]:
    #    gcs_existing_stats = lambda prefix: f"gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.statistics.1993-2019/{prefix}_by_level.zarr"
        time_skip = int(pd.Timedelta(Emulator.delta_t_model)/pd.Timedelta("3h")) # everything is in 3 hour time steps in fv3
        
    elif comp.lower() == "ocn".lower():
    #    gcs_existing_stats = lambda prefix: f"gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.statistics.1993-2019/{prefix}_by_level.zarr"
        time_skip = int(pd.Timedelta(Emulator.delta_t_model)/pd.Timedelta("6h")) # time step is 6 hour for the oceans
    
    else:
        raise ValueError("comp values can only be atm, ocn, ice, or land")
   
    print(f"{comp}, time skip = {time_skip}")

    open_zarr_kwargs = {
        "storage_options": {"token": "anon"},
    }

    to_zarr_kwargs = {
        "mode": "a",
    }

    does_it_exist = False
    do_fv_calc = True

    # check whether the specified GCS zarr store exists
    #try:
    #    ds = xr.open_zarr(gcs_existing_stats("mean"), **open_zarr_kwargs)
    #except:
    #    does_it_exist = False
   
    if comp == "atm".lower():
        vcoord = "pfull"
    elif comp == "ocn".lower():
        vcoord = "z_l"

    if does_it_exist:
        if varname in ds:
            if vcoord not in ds[varname].dims:
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
    if comp == "atm":
        all_variables = Emulator.atm_input_variables + Emulator.atm_forcing_variables + Emulator.atm_target_variables
        all_variables.append("log_spfh")
        all_variables.append("log_spfh2m")
    elif comp == "ocn":
        all_variables = Emulator.ocn_input_variables + Emulator.ocn_forcing_variables + Emulator.ocn_target_variables
    elif comp == "ice":
        all_variables = Emulator.ice_input_variables + Emulator.ice_forcing_variables + Emulator.ice_target_variables
    elif comp == "land":
        all_variables = Emulator.land_input_variables + Emulator.land_forcing_variables + Emulator.land_target_variables
    else:
        raise ValueError("comp can only be atm, ocn, land, or ice")

    for key in all_variables:
        submit_slurm_job(key, comp=comp)
