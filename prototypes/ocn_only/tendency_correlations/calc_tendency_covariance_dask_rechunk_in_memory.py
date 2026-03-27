import argparse
import dask
import flox.xarray
import logging
import os
import psutil
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import LocalCluster, Client
from prototypes.ocn_only.emulator import OcnTrainer
from omegaconf import OmegaConf

# Utility functions to understand the cluster
def print_cluster_overview(client: Client):
    info = client.scheduler_info()
    workers = info.get("workers", {})
    print("\n=== Dask Cluster Overview ===")
    print(f"Dashboard: {client.dashboard_link}")
    print(f"Total workers: {len(workers)}")
    total_mem = sum(w["memory_limit"] for w in workers.values()) / 1e9
    total_threads = sum(w["nthreads"] for w in workers.values())
    print(f"Aggregate memory limit: {total_mem:.2f} GB")
    print(f"Aggregate threads: {total_threads}")
    print("-" * 72)
    for wid, w in workers.items():
        host = w["host"]
        mem = w["memory_limit"] / 1e9
        nth = w["nthreads"]
        name = w.get("name", wid[:8])
        print(f"{name:<22} host={host:<20} threads={nth:<3} mem={mem:6.2f} GB")
    print("-" * 72)

def print_worker_runtime_state(client: Client):
    # Query each worker process for PID, RSS, and CPU affinity
    # RSS would tell you how much real memory a Dask worker is using right now.
    # CPU affinity would tell you which CPU cores the current process is running
    # on.
    def _introspect():
        p = psutil.Process(os.getpid())
        rss = p.memory_info().rss / 1e9
        try:
            affinity = p.cpu_affinity()
        except Exception:
            affinity = []
        return {
            "pid": p.pid,
            "rss_gb": rss,
            "cpu_affinity": affinity,
        }
    results = client.run(_introspect)
    print("\n=== Per-worker runtime state (live) ===")
    for k, v in results.items():
        print(f"{k}: pid={v['pid']} rss={v['rss_gb']:.3f} GB cpu_affinity={v['cpu_affinity']}")
    print("-" * 72)

def cross_covariance(da: xr.DataArray, 
                     dim: str = "channels", 
                     spatial_avg: bool = True,
                     lat_dim: str = "lat", 
                     lon_dim: str = "lon",
                     norm: bool = False) -> xr.DataArray:
    """
    Calculates the cross-covariance or cross-correlation matrix for all pairs 
    along dim.

    Args:
        da (xr.DataArray): The input DataArray, which should be loaded into memory.
                           ('sample', 'lat', 'lon', 'channels').
        dim (str): The dimension along which variables are defined (e.g., "channels").
        spatial_avg (bool): If True, spatially average the covariance/correlation.
        lat_dim (str): Name of the latitude dimension.
        lon_dim (str): Name of the longitude dimension.
        norm (bool): If true, compute correlation instead of covariance.

    Returns:
        xr.DataArray: A DataArray containing the full covariance or correlation matrix.
    """

    n_samples = da.sizes["sample"]

    if not norm:
        statistic = "covariance"
        # For covariance, the one-pass demean -> dot product is stable and correct.
        da_demeaned = da - da.mean(dim="sample", skipna=True)
        da1 = da_demeaned.rename({dim: f"{dim}_x"})
        da2 = da_demeaned.rename({dim: f"{dim}_y"})
        cov_map = xr.dot(da1, da2, dims="sample") / (n_samples - 1)
        out_matrix = cov_map #.mean(["lat", "lon"], skipna=True)

    else:
        statistic = "correlation"
        # First z-score the data and then compute the covariance of the result.
        
        # Calculate mean and sample standard deviation (with ddof=1).
        mean = da.mean(dim="sample", skipna=True).persist()
        std = da.std(dim="sample", skipna=True, ddof=1).persist()
        #epsilon = 1e-12 # when turned on, this is leading inconsistencies,
                         # menifested by the diag entries of corr not eq 1. 

        # Z-score the data.
        da_zscored = (da - mean) / std

        # The correlation is the covariance of the z-scored data.
        da1 = da_zscored.rename({dim: f"{dim}_x"})
        da2 = da_zscored.rename({dim: f"{dim}_y"})
        
        # The dot product of z-scored data, divided by (n-1), is the correlation map.
        corr_map = xr.dot(da1, da2, dims="sample") / (n_samples - 1)
        
        # 4. Average the correlation map over spatial dimensions.
        out_matrix = corr_map #.mean(["lat", "lon"], skipna=True)

    if spatial_avg:
        out_matrix = out_matrix.mean([lat_dim, lon_dim], skipna=True)

    out_matrix.attrs = {"description": f"tendency {statistic}"}
    
    return out_matrix

if __name__ == "__main__":
    # --- Dask Setup ---
    # Configure the cluster for Perlmutter CPU node  with 128 cores and 512GB RAM.
    # We use fewer workers (processes) and give each one multiple threads.
    # This is often more memory-efficient for numerical workloads. 
    cluster = LocalCluster(n_workers=32, threads_per_worker=4,)
    client = Client(cluster)
    #logging.info(f"Dask dashboard link: {client.dashboard_link}")
    print_cluster_overview(client)

    # Parse Arguments
    parser = argparse.ArgumentParser(description="Forecast Covariance Computation")                                                                                                                                                
    parser.add_argument("--prototype", required=True, help="Prototype Name")
    parser.add_argument("--num_missing_samples", default=0, type=int, help="Number of additional missing samples")                                                                                                                            
    parser.add_argument("--norm", action="store_true", help="If set, compute correlation instead of covariance")
    parser.add_argument("--remove_seasonality", action="store_true", help="If set, remove seasonality before computing corr/cov")                                                                                                                      
    parser.add_argument("--no_spatial_avg", action="store_true", help="If true, computes the space-dependent statistics")
    args = parser.parse_args()
 
    # Store and log
    prototype = args.prototype
    num_missing_samples = args.num_missing_samples
    norm = args.norm
    spatial_avg = not args.no_spatial_avg 
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(f"prototype: {prototype}")
    logging.info(f"Num of missing samples: {num_missing_samples}")
    logging.info(f"Computing {'correlation' if args.norm else 'covariance'} matrix")

    # Emulator
    home_dir = "/global/homes/n/nagarwal"
    config_trainer_path = f"{home_dir}/graph-ufs/prototypes/ocn_only/{prototype}/config.yaml"
    trainer_config = OmegaConf.load(config_trainer_path)
    emulator = OcnTrainer(config=trainer_config)
    logging.info("Emulator object created")
 
    # Open Targets 
    fname = "targets.zarr"
    ds = xr.open_zarr(os.path.join(emulator.local_store_path, "training", fname), 
                      chunks={"sample": "auto"})

    # Assign datetime coordinate and subsample
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
    ds = ds.assign_coords(datetime=("sample", targets_time.values))
    logging.info(f"Datetime coordinate assigned") 
   
    # subsample
    factor = int(target_lead_time / delta_t_data)
    ds = ds.isel(sample=slice(None, None, factor)).astype("float32")
    logging.info("Targets subsampled")

    # Compute tendency
    ds = ds.diff(dim="sample")
    logging.info("Tendency computed lazily")

    # Select a variable and convert to DataArray
    varname = "targets" if "targets" in ds.variables else list(ds.data_vars)[0]
    da = ds[varname]
    
    # Remove seasonality
    if args.remove_seasonality:
        logging.info("Removing seasonality from tendency targets")
        # Dask can handle groupby operations in a parallel, chunked manner.
        seasonality = da.groupby('datetime.month').mean('sample', skipna=True)
        da = da.groupby('datetime.month') - seasonality
        logging.info("Seasonality removal graph built.")
   
    # --- PERFORMANCE OPTIMIZATION: Rechunk Dask array in memory ---
    # With 512GB of RAM and 32 workers, each worker gets 16GB. We can
    # use much larger chunks to reduce task overhead.
    # A chunk size of 50 samples will be ~500mb, which would mean that some 30
    # chunks can fit into the worker's memory.
    samples_per_chunk = 20
    da = da.chunk({"sample": samples_per_chunk})
    logging.info(f"Rechunked array in memory to have {samples_per_chunk} samples per chunk.") 
    
    logging.info("Building Dask computation graph for covariance...")
    tendency_cov_lazy = cross_covariance(da, dim="channels", spatial_avg=spatial_avg,
                                         lat_dim="lat", lon_dim="lon", norm=norm)
    
    # --- Trigger Computation ---
    # The .compute() method tells Dask to execute the entire graph of tasks.
    logging.info("Executing Dask graph. This may take a while...")
    tendency_cov = tendency_cov_lazy.compute()
    logging.info("Covariance/correlation matrix computed.")
    
    # Convert to dataset and write to disk
    name = "correlation" if norm else "covariance"
    ds_out = tendency_cov.to_dataset(name="targets")
        
    # Save
    seasonality_suffix = "rm_seasonality" if args.remove_seasonality else "seasonality_intact"
    space_suffix = "spatially_averaged" if spatial_avg else "space_dependent" 
    output_file = (
        f"./tendency_{name}_ocn_only_{emulator.target_lead_time}"
        f"_{seasonality_suffix}_{space_suffix}.nc"
    )
    ds_out.to_netcdf(output_file)
    logging.info(f"Saved {name} matrix to {output_file}")

    # Optional: print live worker memory after compute
    print_worker_runtime_state(client)

    # Shut down the Dask cleint
    client.close()
