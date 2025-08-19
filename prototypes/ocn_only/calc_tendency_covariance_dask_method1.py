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
from omegaconf import OmegaConf
from emulator import OcnTrainer

from dask.distributed import LocalCluster, Client

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

def main():
    parser = argparse.ArgumentParser(description=
        "Forecast Covariance Computation")
    parser.add_argument("--prototype", required=True,
        help="Prototype Name")
    parser.add_argument("--num_missing_samples", default=0,
        type=int, help="Number of additional missing samples")
    parser.add_argument("--norm", action="store_true",
        help="If set, compute correlation instead of covariance")
    parser.add_argument("--scheduler", default=None,
        help="Dask scheduler address, e.g. tcp://nodeA:8786")
    parser.add_argument("--lat-chunk", type=int, default=256,
        help="Lat chunk size")
    parser.add_argument("--lon-chunk", type=int, default=256,
        help="Lon chunk size")
    args = parser.parse_args()
    
    # Dask client
    cluster = LocalCluster(n_workers=128, 
                           #threads_per_worker=2, 
                           memory_limit='8GB'
    )
    #client = Client(args.scheduler) if args.scheduler else Client()
    client = Client(cluster)
    print(client)
    print_cluster_overview(client)
    
    # Logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(f"prototype: {args.prototype}")
    logging.info(f"Num of missing samples: {args.num_missing_samples}")
    logging.info(f"Computing {'correlation' if args.norm else 'covariance'} matrix")

    # Emulator 
    config_trainer_path = f"./{args.prototype}/config.yaml"
    trainer_config = OmegaConf.load(config_trainer_path)
    emulator = OcnTrainer(config=trainer_config)
    logging.info("Emulator object created")
 
    # Data
    fname = "targets.zarr"
    chunks = {"lat":args.lat_chunk, "lon":args.lon_chunk, "sample":-1, "channels":1}
    ds_targets = xr.open_zarr(os.path.join(emulator.local_store_path, 
        "training", fname), chunks="auto").astype("float32")
    ds_targets = ds_targets.chunk(chunks).persist()
    logging.info("dataset rechunked!!!")

    # Build datetime coordinate for sample
    target_lead_time = int(emulator.target_lead_time[:-1])
    delta_t_data = int(emulator.delta_t_data[:-1])
    start = pd.Timestamp(emulator.training_dates[0])
    end = pd.Timestamp(emulator.training_dates[-1])
    targets_time = pd.date_range(
        start = start + pd.Timedelta(target_lead_time, "h"),
        end = end - int(args.num_missing_samples)*pd.Timedelta(delta_t_data, "h"),
        freq=emulator.delta_t_data,
        inclusive="both"
    ) 
    ds_targets = ds_targets.assign_coords(datetime=("sample", targets_time.values))
    logging.info(f"Datetime coordinate assigned") 
   
    # subsample along time
    step = int(target_lead_time/delta_t_data)
    ds_targets = ds_targets.isel(sample=slice(None, 1000, step)).astype("float32")
    logging.info("Targets subsampled")

    # Compute tendency
    ds_targets = ds_targets.diff(dim="sample")
    logging.info("Tendency computed")
    
    # Remove seasonality
    #month = tendency_targets.coords["datetime"].dt.month
    #seasonality = flox.xarray.groupby_reduce(
    #    tendency_targets.targets,
    #    month,
    #    func="mean",
    #    axis="sample",
    #    method="map-reduce"
    #)
    varname = "targets" if "targets" in ds_targets.variables else list(ds_targets.data_vars)[0]
    seasonality = ds_targets[varname].groupby("datetime.month").mean("sample")
    da_targets = ds_targets[varname].groupby('datetime.month') - seasonality
    # The above changes the chunking. So we must rechunk to make the
    # xr.apply_ufunc() work
    da_targets = da_targets.chunk(chunks).unify_chunks().persist()
    logging.info("Deseasonalized tendency (by month)")

    # Cross-channel tendency covariance
    # Upper triangular channel pairs 
    channels = ds_targets.sizes["channels"]
    pairs = [(i, j) for i in range(channels) for j in range(i, channels)]
    
    # For covariance, using a custom function like below may perform better than
    # using the builtin xr.cov as that does not provide enough granularity to
    # provide unbiased estimate (i.e., dividing by N-1 and not N) and may
    # increase memory usage.
    def cov_over_time_then_spatial_avg(a, b):
        # covariance along time per (lat,lon) then spatial mean
        a0 = a - a.mean("sample")
        b0 = b - b.mean("sample")
        num = (a0 * b0).sum("sample")
        den = a.sizes["sample"] - 1
        field_cov = num / den
        return field_cov.mean(("lat", "lon"), skipna=True)

    # Use xr.corr directly for correlation computation as it is more optimized
    # and reduced the memory usage by reducing intermediate arrays and
    # simultaneous computation of normalizing factors.
    def corr_over_time_then_spatial_avg(a, b):
        field_corr = xr.corr(a, b, dim="sample")
        return field_corr.mean(("lat", "lon"), skipna=True)
     
    comp_func = corr_over_time_then_spatial_avg if args.norm else cov_over_time_then_spatial_avg
    
    # Build scalar DataArray tasks (each pair returns a scalar DataArray)
    tasks = []
    for (i, j) in pairs:
        print("i=", i, "j=", j)
        xi = da_targets.isel(channels=i)
        xj = da_targets.isel(channels=j)  
        tasks.append(comp_func(xi, xj))
    logging.info("task list populated. Next step is the execution.")    
 
    # Compute in parallel
    scalars = dask.compute(*tasks) 
    logging.info("covariance computed parallely!!!")
 
    # Assemble the matrix
    stat_matrix = np.empty((channels, channels), dtype=np.float32)
    
    for idx, (i,j) in enumerate(pairs):
        val = float(scalars[idx].values)
        stat_matrix[i, j] = val
        stat_matrix[j, i] = val
    
    # Wrap in a dataset with coords and write to disk
    name = "correlation" if args.norm else "covariance"
    channels_coords = ds_targets["channels"].values if "channels" in ds_targets.coords else range(channels)
    ds_out = xr.DataArray(
        stat_matrix,
        dims=("channels_i", "channels_j"),
        coords={"channels_i": channels_coords, "channels_j": channels_coords},
        name=name,
    ).to_dataset()
    
    # Save
    output_file = f"./{args.prototype}/tendency_{name}_ocn_only_{emulator.target_lead_time}.nc"
    ds_out.to_netcdf(output_file)
    logging.info(f"Saved {name} matrix to {output_file}")

    # Optional: print live worker memory after compute
    print_worker_runtime_state(client)

if __name__ == "__main__":
    main()
