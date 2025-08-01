import logging
import os
import subprocess
import argparse

import xarray as xr
import pandas as pd

from graphufs.log import setup_simple_log
from graphufs.fvstatistics import FVStatisticsComputer

from emulator import OcnTrainer as Emulator

def submit_slurm_job(emulator, prototype, varname, comp="atm", apartition="compute", n_cpus=30):

    scriptdir = os.path.join(os.getcwd(), "job-scripts")
    for d in [scriptdir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    # On Perlmutter
    slurm_dir = f"{emulator.local_store_path}/slurm/fvstats"
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
        f"cd /global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only\n\n"+\
        f"python -c 'from calc_statistics import main ; main(\"{prototype}\", \"{varname}\", \"{comp}\")'"

    scriptname = f"{scriptdir}/submit_statistics_{varname}.sh"
    with open(scriptname, "w") as f:
        f.write(jobscript)

    subprocess.run(f"sbatch {scriptname}", shell=True)

def main(prototype, varname, comp):

    setup_simple_log()

    # Build config paths
    config_trainer_path = f"./{prototype}/config.yaml"

    # Initialize the emulator
    emulator = Emulator(config_trainer_path,)

    # if it's a surface variable, then try reading it from existing stats. If no 
    # stats exist, then compute everything. If 3D, compute the FV version.
    statsdir = f"statistics/{emulator.delta_t_model}/{comp}.fvstatistics.1993-2019"
    if not os.path.isdir(statsdir):
        os.makedirs(statsdir)
    path_out = os.path.abspath(statsdir)

    if comp in ["atm", "ice", "land"]:
    #    gcs_existing_stats = lambda prefix: f"gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.statistics.1993-2019/{prefix}_by_level.zarr"
        time_skip = int(pd.Timedelta(emulator.delta_t_model)/pd.Timedelta("3h")) # everything is in 3 hour time steps in fv3
        
    elif comp.lower() == "ocn".lower():
    #    gcs_existing_stats = lambda prefix: f"gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.statistics.1993-2019/{prefix}_by_level.zarr"
        time_skip = int(pd.Timedelta(emulator.delta_t_model)/pd.Timedelta("6h")) # time step is 6 hour for the oceans
    
    else:
        raise ValueError("comp values can only be atm, ocn, ice, or land")
   
    print(f"{comp}, time skip = {time_skip}")

    open_zarr_kwargs = {
        "storage_options": {"token": "anon"},
    }

    to_zarr_kwargs = {
        "mode": "a",
    }

    existing_stats = lambda prefix: f"{emulator.norm_urls[comp][prefix]}"
    does_it_exist = False

    # check whether the specified zarr store exists
    try:
        ds = xr.open_zarr(existing_stats("mean"), **open_zarr_kwargs)
        does_it_exist = True
    except:
        does_it_exist = False
   
    if comp == "atm".lower():
        vcoord = "pfull"
    elif comp == "ocn".lower():
        vcoord = "z_l"

    if does_it_exist:
        store_stats = lambda prefix: f"{path_out}/{prefix}_by_level.zarr"
        for prefix in ["mean", "stddev", "diffs_stddev"]:
            ds = xr.open_zarr(existing_stats(prefix), **open_zarr_kwargs)
            if varname in ds:
                ds = ds[[varname]]
                ds.to_zarr(store_stats(prefix), **to_zarr_kwargs)
                logging.info(f"Pulled {varname} {prefix} from {existing_stats(prefix)} to {store_stats(prefix)}")

    else:
        logging.info(f"No prior statistics exist. Computing statistics for {varname}")
        fvstats = FVStatisticsComputer(
                path_in=emulator.data_url[comp],
                path_out=path_out,
                comp=comp,
                interfaces=emulator.interfaces[comp],
                start_date=None,
                end_date=emulator.training_dates[-1],
                time_skip=time_skip,
                spatial_avg=True,
                load_full_dataset=False,
                transforms=emulator.input_transforms,
                open_zarr_kwargs=open_zarr_kwargs,
                to_zarr_kwargs=to_zarr_kwargs
        )
        fvstats(varname)

parser = argparse.ArgumentParser(description="Statistics Computation")
parser.add_argument("--prototype", required=True, help="Prototype Name (e.g.,R1, R2)")
parser.add_argument("--comp", required=True, help="Component for which statistics to be computed (e.g., atm, ocn)")

if __name__ == "__main__":

    args = parser.parse_args()
    prototype = args.prototype
    comp = args.comp
   
    # Build config paths
    config_trainer_path = f"./{prototype}/config.yaml"

    # Initialize the emulator
    emulator = Emulator(config_trainer_path,)
 
    if comp == "atm":
        all_variables = set(emulator.atm_input_variables + emulator.atm_forcing_variables + emulator.atm_target_variables)
        for key in emulator.input_transforms.keys(): 
            if key in all_variables:
                transformed_key = emulator.input_transforms[key] + "_" + key
                all_variables.append(transformed_key)
    elif comp == "ocn":
        all_variables = set(emulator.ocn_input_variables + emulator.ocn_forcing_variables + emulator.ocn_target_variables)
    elif comp == "ice":
        all_variables = set(emulator.ice_input_variables + emulator.ice_forcing_variables + emulator.ice_target_variables)
    elif comp == "land":
        all_variables = set(emulator.land_input_variables + emulator.land_forcing_variables + emulator.land_target_variables)
    else:
        raise ValueError("comp can only be atm, ocn, land, or ice")
    
    for key in all_variables:
        submit_slurm_job(emulator, prototype, key, comp=comp)
