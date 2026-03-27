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

def cross_covariance_xr(
    da: xr.DataArray,
    dim: str = "channels",
    spatial_avg: bool = True,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    norm: bool = False
) -> xr.DataArray:
    """
    Calculates the cross-covariance or correlation matrix for all pairs
    along a given dimension.

    This function is robust, using xarray's label-based assignment
    to prevent dimension misalignment.

    Args:
        da (xr.DataArray): The input DataArray. Must have a "sample" dimension
                           and the specified `dim`.
        dim (str): The shared dimension along which variables are defined.
        spatial_avg (bool): If True, spatially average the covariance.
        lat_dim (str): Name of the latitude dimension.
        lon_dim (str): Name of the longitude dimension.
        norm (bool): If True, compute correlation instead of covariance.

    Returns:
        xr.DataArray: A DataArray containing the cov/corr matrix.
    """
    
    # Get coordinate values from the dimension
    channel_coords = da[dim]
    
    # Define the output dimensions and coordinates for our empty array
    out_dims = [f"{dim}_x", f"{dim}_y"]
    out_coords = {f"{dim}_x": channel_coords, f"{dim}_y": channel_coords}
    
    # Define the shape of the empty data
    out_shape = (len(channel_coords), len(channel_coords))

    if not spatial_avg:
        # Add spatial dimensions and coordinates if not averaging
        out_dims.extend([lat_dim, lon_dim])
        out_coords[lat_dim] = da[lat_dim]
        out_coords[lon_dim] = da[lon_dim]
        out_shape += (da.sizes[lat_dim], da.sizes[lon_dim])
    
    # Create the empty 'shell' DataArray, filled with NaNs
    # This is the robust way to pre-allocate.
    out_da = xr.DataArray(
        data=np.full(out_shape, np.nan),
        dims=out_dims,
        coords=out_coords
    )

    # Select the correct operation (cov or corr)
    op = xr.corr if norm else xr.cov
    statistic = "correlation" if norm else "covariance"
    
    # Iterate over channel values
    for i in range(len(channel_coords)):
        logging.info(f"i = {i}")
        # Only compute the upper triangle (j >= i)
        for j in range(i, len(channel_coords)):
            logging.info(f"j = {j}")
            
            channel_i_val = channel_coords[i].item()
            channel_j_val = channel_coords[j].item()

            # Select the data for each channel using .sel()
            da_i = da.sel({dim: channel_i_val})
            da_j = da.sel({dim: channel_j_val})

            # Calculate covariance/correlation
            # This result will have 'lat' and 'lon' dims
            out_cov = op(da_i, da_j, dim="sample")
            
            # Create the dictionary for .loc to select the correct "slot"
            loc_ij = {f"{dim}_x": channel_i_val, f"{dim}_y": channel_j_val}
            loc_ji = {f"{dim}_x": channel_j_val, f"{dim}_y": channel_i_val}

            if spatial_avg:
                out_mean = out_cov.mean([lat_dim, lon_dim], skipna=True)
                out_da.loc[loc_ij] = out_mean
                out_da.loc[loc_ji] = out_mean.copy() # Apply symmetry
            else:
                out_da.loc[loc_ij] = out_cov
                out_da.loc[loc_ji] = out_cov.copy() # Apply symmetry

    # Add attributes and return
    out_da.attrs["description"] = f"Cross-{statistic} over 'sample' dim."
    return out_da

def cross_covariance(da: xr.DataArray, dim: str = "channels", 
                     spatial_avg = True, lat_dim = "lat", 
                     lon_dim = "lon", norm = False):
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
  if spatial_avg:
      out_mat = np.zeros((len(channels), len(channels)))
  else:
      out_mat = np.zeros((len(channels), len(channels), 
                          da.sizes[lat_dim], da.sizes[lon_dim]))

  for i in range(len(channels)):
    logging.info(f"i = {i}")
    for j in range(i, len(channels)):
      logging.info(f"j = {j}")
      channel_i = channels[i]
      channel_j = channels[j]

      # Calculate covariance
      if not norm:
          out = xr.cov(da.isel({dim:channel_i}), da.isel({dim:channel_j}), dim="sample")
          statistic = "covariance"
      else:
          out = xr.corr(da.isel({dim:channel_i}), da.isel({dim:channel_j}),
                        dim="sample")
          statistic = "correlation"
      
      if spatial_avg:
          out_mat[i, j] = out.mean(["lat","lon"], skipna=True)
      else:
          out_mat[i, j] = out 
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
