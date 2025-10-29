import argparse
import logging
import os
import sys
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import flox.xarray

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
    logging.info(f"i = {i}")
    for j in range(i, len(channels)):
      logging.info(f"j = {j}")
      channel_i = channels[i]
      channel_j = channels[j]

      # Calculate covariance
      if not norm:
          out_mat[i, j] = xr.cov(da.isel({dim:channel_i}), 
                                 da.isel({dim:channel_j}),
                                 dim="sample").mean(["lat","lon"], skipna=True)
          statistic = "covariance"
      else:
          out_mat[i, j] = xr.corr(da.isel({dim:channel_i}), 
                                  da.isel({dim:channel_j}),
                                  dim="sample").mean(["lat","lon"], skipna=True)
          statistic = "correlation"
      logging.info(f"value = {out_mat[i, j]}")
      # apply symmetry
      out_mat[j, i] = out_mat[i, j].copy()

  # convert to dataArray
  out_da = xr.DataArray(out_mat, 
                        dims=["channels_x","channels_y"], 
                        coords={"channels_x":channels.values, "channels_y":channels.values}, 
                        attrs={"description":f"tendency {statistic}"})
  return out_da

parser = argparse.ArgumentParser(description=
    "Forecast Covariance Computation")
parser.add_argument("--prototype", required=True, 
    help="Prototype Name")
parser.add_argument("--num_missing_samples", default=0, 
    type=int, help="Number of additional missing samples")
parser.add_argument("--norm", action="store_true", # meaning default=False 
    help="If set, compute correlation instead of covariance")
parser.add_argument("--remove_seasonality", action="store_true",
    help="If set, remove seasonality before computing corr/cov")

if __name__ == "__main__":
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = parser.parse_args()
    prototype = args.prototype
    num_missing_samples = args.num_missing_samples
    norm = args.norm

    logging.info(f"prototype: {prototype}")
    logging.info(f"Num of missing samples: {num_missing_samples}")
    logging.info(f"Computing {'correlation' if norm else 'covariance'} matrix")

    # get the emulator 
    config_trainer_path = f"./{prototype}/config.yaml"
    trainer_config = OmegaConf.load(config_trainer_path)
    emulator = OcnTrainer(config=trainer_config)
    logging.info("Emulator object created")
 
    # open targets
    fname = "targets.zarr"
    ds = xr.open_zarr(os.path.join(emulator.local_store_path, 
        "training", fname))

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
    ds = ds.assign_coords(datetime=("sample", targets_time.values))
   
    # subsample dataset and load into memory
    factor = int(target_lead_time/delta_t_data)
    steps_per_day = int(24/target_lead_time)
    num_years = 6
    end = 365*num_years*factor*steps_per_day if factor<int(4) else None
    ds = ds.isel(sample=slice(None, end, factor)).astype('float32')
    logging.info("targets subsampled")

    # compute tendency
    ds = ds.diff(dim="sample")
    logging.info("Targets tendency created")
    
    if remove_seasonality:
        # remove seasonality
        #month = tendency_targets.coords["datetime"].dt.month
        #seasonality = flox.xarray.groupby_reduce(
        #    tendency_targets.targets,
        #    month,
        #    func="mean",
        #    axis="sample",
        #    method="map-reduce"
        #)
        seasonality = ds.targets.groupby('datetime.month').mean('sample')
        da = ds.targets.groupby('datetime.month') - seasonality
        da.load()
        logging.info("seasonality removed from tendency targets")

        # cross-channel tendency covariance
        tendency_cov = cross_covariance(da, dim="channels", norm=norm)
    
    else:
        tendency_cov = cross_covariance(ds.targets.load(), dim="channels", norm=norm)

    logging.info("covariance computed!!!")
    
    # convert to dataset and write to disk
    name = "correlation" if norm else "covariance"
    ds = tendency_cov.to_dataset(name="targets")
    seasonality_suffix = "rm_seasonality" if remove_seasonality else "seasonality_intact"
    output_file = f"./{prototype}/tendency_{name}_ocn_only_{emulator.target_lead_time}_{seasonality_suffix}.nc"
    ds.to_netcdf(output_file)
    logging.info(f"Saved {name} matrix to {output_file}")
