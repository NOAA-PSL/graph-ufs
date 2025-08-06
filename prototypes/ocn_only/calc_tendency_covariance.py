import argparse
import logging
import os
import sys
import warnings
import xarray as xr
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from emulator import OcnTrainer

def cross_covariance(da: xr.DataArray, dim: str = "channels", norm = False):
  """
  Calculates the cross-covariance matrix for all pairs along dim.

  Args:
    da (xr.DataArray): The input DataArray.
    dim (str): The shared dimension along which variables are defined.
    norm (bool): If true, normalization by std will be applied to compute correlation

  Returns:
    xr.DataArray: A DataArray containing the cov/corr matrix.
  """

  channels = da[dim]  # Get the dim values
    
  out_mat = np.zeros((len(channels), len(channels)))
  
  for i in range(len(channels)):
    for j in range(i, len(channels)):

      channel_i = channels[i]
      channel_j = channels[j]

      # Calculate covariance
      if not norm:
          out_mat[i, j] = xr.cov(da.isel({dim:channel_i}), da.isel({dim:channel_j})).compute()
          statistic = "covariance"
      else:
          out_mat[i, j] = xr.corr(da.isel({dim:channel_i}), da.isel({dim:channel_j})).compute()
          statistic = "correlation"
          
      # apply symmetry
      out_mat[j, i] = out_mat[i, j].copy()

  # convert to dataArray
  out_da = xr.DataArray(out_mat, dims=["channels_x","channels_y"], coords={"channels_x":channels.values, "channels_y":channels.values}, 
                        attrs={"description":f"forecast error cross-{statistic} for the R4/T3M ocn-only configuration (trained with correct statistics)"})
  return out_da


parser = argparse.ArgumentParser(description="Forecast Error Covariance Computation")
parser.add_argument("--prototype", required=True, help="Prototype Name")  # e.g., "R1"
parser.add_argument("--num_missing_samples", default=0, 
    help="Number of additional missing samples from targets due to wrong input_overlap value during preprocessing")  # integer

if __name__ == "__main__":
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = parser.parse_args()
    prototype = args.prototype
    num_missing_samples = args.num_missing_samples
    logging.info(f"prototype:{prototype}")
    logging.info(f"Num of missing samples:{num_missing_samples}")

    # get the emulator 
    config_trainer_path = f"./{prototype}/config.yaml"
    trainer_config = OmegaConf.load(config_trainer_path)
    emulator = OcnTrainer(config=trainer_config)
    logging.info("Emulator object created")
 
    # open targets
    fname = "targets.zarr"
    targets = xr.open_zarr(os.path.join(emulator.local_store_path, 
        "training", fname))
    logging.info(f"targets: {targets}")

    # assign the datetime coordinate
    target_lead_time = int(emulator.target_lead_time[:-1])
    delta_t_data = int(emulator.delta_t_data[:-1])
    start = pd.Timestamp(emulator.training_dates[0])
    end = pd.Timestamp(emulator.training_dates[-1])

    targets_time = pd.date_range(
        start = start + pd.Timedelta(target_lead_time, "h"),
        end = end - int(num_missing_samples)*pd.Timedelta(delta_t_data, "h"),
        freq=emulator.delta_t_data,
        inclusive="both"
    ) 
    logging.info(f"targets time stamps:{targets_time}") 
    targets_dated = targets.assign_coords(datetime=("sample", targets_time.values,))
    
    # compute tendency
    tendency_targets = targets_dated.diff(dim="sample")
    logging.info("Targets tendency created")
    
    # remove seasonality
    seasonality = tendency_targets.targets.groupby('datetime.month').mean()
    da_tendency_targets_deseasoned = tendency_targets.targets.groupby('datetime.month') - seasonality
    logging.info("seasonality removed from tendency targets")

    # cross-channel tendency covariance
    tendency_cov = cross_covariance(da_tendency_targets_deseasoned, dim="channels")
    logging.info("covariance computed!!!")
    
    # convert to dataset and write to disk
    ds_tendency_cov = tendency_cov.to_dataset(name="targets")
    output_file = f"./{prototype}/tendency_covariance_ocn_only_{emulator.target_lead_time}.nc"
    ds_tendency_cov.to_netcdf(output_file)
    logging.info("Wrote to the disk. Done...") 
