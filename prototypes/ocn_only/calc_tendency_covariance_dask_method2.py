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
from dask.distributed import Client
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
    cluster = LocalCluster(n_workers=64, 
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
    chunks = {"lat":args.lat_chunk, "lon":args.lon_chunk, "sample":-1, "channels":-1}
    ds_targets = xr.open_zarr(os.path.join(emulator.local_store_path, 
        "training", fname), chunks="auto").astype("float32")
    ds_targets = ds_targets.chunk(chunks)

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
    da_targets = da_targets.chunk(chunks).unify_chunks()
    logging.info("Deseasonalized tendency (by month)")

    # Cross-channel tendency covariance
    channels = da_targets.sizes["channels"]
    
    def crosscorr_mat(a: np.ndarray, b:np.ndarray) -> np.ndarray:
        """
        Compute cross-correlation matrix along 'sample' for all channel pairs at once.
        a: (..., sample, channels)
        b: (..., sample, channels)
        returns: (..., channels, channels) correlation matrix
        """
        # center
        a0 = a - a.mean(axis=-2, keepdims=True)
        b0 = b - b.mean(axis=-2, keepdims=True)

        # covariance matrix (..., channels, channels)
        num = np.einsum("...tc,...td->...cd", a0, b0)

        # std devs
        sa = np.sqrt(np.einsum("...tc,...tc->...c", a0, a0))
        sb = np.sqrt(np.einsum("...td,...td->...d", b0, b0))
        den = sa[..., :, None] * sb[..., None, :]

        return num / den

    def crosscov_mat(a: np.ndarray, b:np.ndarray) -> np.ndarray:
        """
        Compute cross-correlation matrix along 'sample' for all channel pairs at once.
        a: (..., sample, channels)
        b: (..., sample, channels)
        returns: (..., channels, channels) correlation matrix
        """
        # center
        a0 = a - a.mean(axis=-2, keepdims=True)
        b0 = b - b.mean(axis=-2, keepdims=True)

        # covariance matrix (..., channels, channels)
        num = np.einsum("...tc,...td->...cd", a0, b0)

        return num
   
    comp_func = crosscorr_mat if args.norm else crosscov_mat
    
    # Create a one big task graph using xarray.apply_ufunc and braodcast over
    # (lat, lon)
    cross_channel_stat = xr.apply_ufunc(
        comp_func,
        da_targets,   # (lat, lon, sample, channels)
        da_targets,   # (lat, lon, sample, channels)
        input_core_dims=[["sample","channels"], ["sample","channels"]],
        output_core_dims=[["channels_i","channels_j"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        output_sizes={"channels_i":channels, "channels_j":channels}
    )
    cross_channel_stat = cross_channel_stat.mean(("lat","lon")).compute()
    logging.info("corr vectorization using xr.apply_ufunc is done")
 
    # Wrap in a dataset with coords and write to disk
    name = "correlation" if args.norm else "covariance"
    channels_coords = ds_targets["channels"].values if "channels" in ds_targets.coords else range(channels)
    ds_out = xr.DataArray(
        cross_channel_stat_mean,
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
