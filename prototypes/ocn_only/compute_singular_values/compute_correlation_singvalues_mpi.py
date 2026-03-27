#!/usr/bin/env python
"""
Simple parallel correlation matrix computation using xarray + zarr + MPI.
Computes cross-channel correlation for Zarr datasets.

Dataset: 192 x 384 x 40880 x 29 (~1.5 TB)
Output: 29x29 correlation matrix + full eigenvalue spectrum
"""

from mpi4py import MPI
import numpy as np
import xarray as xr
import time

def compute_correlation_mpi(zarr_path, output_path='correlation_results.npz'):
    """
    Compute correlation matrix using MPI parallelization.
    Each rank processes a chunk of time steps.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Starting with {size} MPI ranks")
        print(f"Loading data from: {zarr_path}")
        t_start = time.time()
    
    # Open Zarr with xarray (all ranks)
    ds = xr.open_zarr(zarr_path)
    
    # Assuming the dataset contains a single variable
    data_var = list(ds.data_vars)[0]
    data = ds[data_var]
    
    n_time, n_lat, n_lon, n_channels = data.shape
    n_samples = n_time * n_lat * n_lon
    
    if rank == 0:
        print(f"Data variable: {data_var}")
        print(f"Shape: {n_time} x {n_lat} x {n_lon} x {n_channels}")
        print(f"Total samples: {n_samples:,}")
    
    # Divide time dimension among ranks
    time_per_rank = n_time // size
    time_start = rank * time_per_rank
    time_end = (rank + 1) * time_per_rank if rank < size - 1 else n_time
    
    if rank == 0:
        print(f"Each rank processes ~{time_per_rank} time steps")
        print(f"\nStep 1/4: Loading local data chunk (rank {rank})...")
    
    # Each rank reads its time chunk
    local_data = data.isel(sample=slice(time_start, time_end)).values.astype(np.float32)
    n_local_samples = local_data.shape[0] * local_data.shape[1] * local_data.shape[2]
    
    # Reshape to (n_local_samples, n_channels)
    local_data = local_data.reshape(n_local_samples, n_channels)
    
    if rank == 0:
        print(f"Local chunk shape: {local_data.shape}")
        print(f"\nStep 2/4: Computing statistics...")
    
    # Compute local sums and sums of squares
    local_sum = np.sum(local_data, axis=0, dtype=np.float64)
    local_sum_sq = np.sum(local_data**2, axis=0, dtype=np.float64)
    local_count = n_local_samples
    
    # Gather all statistics to rank 0
    global_sum = np.zeros(n_channels, dtype=np.float64)
    global_sum_sq = np.zeros(n_channels, dtype=np.float64)
    
    comm.Reduce(local_sum, global_sum, op=MPI.SUM, root=0)
    comm.Reduce(local_sum_sq, global_sum_sq, op=MPI.SUM, root=0)
    
    # Rank 0 computes global mean and std
    if rank == 0:
        global_mean = global_sum / n_samples
        global_std = np.sqrt(global_sum_sq / n_samples - global_mean**2)
        print(f"Global means computed: min={global_mean.min():.4f}, max={global_mean.max():.4f}")
        print(f"Global stds computed: min={global_std.min():.4f}, max={global_std.max():.4f}")
    else:
        global_mean = np.zeros(n_channels, dtype=np.float64)
        global_std = np.zeros(n_channels, dtype=np.float64)
    
    # Broadcast mean and std to all ranks
    comm.Bcast(global_mean, root=0)
    comm.Bcast(global_std, root=0)
   
    # Convert to float32 for normalization
    global_mean_f32 = global_mean.astype(np.float32)
    global_std_f32 = global_std.astype(np.float32)
 
    if rank == 0:
        print(f"\nStep 3/4: Computing correlation matrix...")
    
    # Normalize local data
    local_normalized = (local_data - global_mean_f32) / global_std_f32
    
    # Compute local contribution to correlation matrix
    # Correlation = (1/n) * X^T @ X where X is normalized
    local_corr = np.dot(local_normalized.T, local_normalized).astype(np.float64)
   
    # Free memory
    del local_data, local_normalized
 
    # Sum all local correlations
    global_corr = np.zeros((n_channels, n_channels), dtype=np.float64)
    comm.Reduce(local_corr, global_corr, op=MPI.SUM, root=0)
    
    # Only rank 0 computes final results
    if rank == 0:
        # Divide by total samples to get correlation
        corr_matrix = global_corr / n_samples
        
        t_corr = time.time() - t_start
        print(f"Correlation matrix computed in {t_corr/60:.2f} minutes")
        print(f"Correlation range: [{corr_matrix.min():.4f}, {corr_matrix.max():.4f}]")
        print(f"Diagonal mean: {np.diag(corr_matrix).mean():.6f} (should be ~1.0)")
        
        print(f"\nStep 4/4: Computing eigendecomposition...")
        t0 = time.time()
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        
        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute singular values from eigenvalues: σ = √(λ * (n-1))
        singular_values = np.sqrt(eigenvalues * (n_samples - 1))
        
        print(f"Eigendecomposition completed in {time.time()-t0:.2f} seconds")
        
        # Print eigenvalue summary
        print(f"\nEigenvalue spectrum:")
        print(f"  Max: {eigenvalues[0]:.6f}")
        print(f"  Min: {eigenvalues[-1]:.6e}")
        print(f"\nSingular value spectrum:")
        print(f"  Max: {singular_values[0]:.6e}")
        print(f"  Min: {singular_values[-1]:.6e}")
        print(f"  Condition number: {singular_values[0]/singular_values[-1]:.6e}")
        
        # Save results
        print(f"\nSaving results to {output_path}...")
        np.savez_compressed(
            output_path,
            correlation_matrix=corr_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            singular_values=singular_values,
            channel_means=global_mean,
            channel_stds=global_std
        )
        
        print(f"\nTotal time: {(time.time()-t_start)/60:.2f} minutes")
        print(f"Done! Results saved to {output_path}")
        
        return corr_matrix, eigenvalues, eigenvectors
    else:
        return None, None, None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: mpirun -n <nprocs> python correlation_xarray.py <zarr_path> [output_path]")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'correlation_results.npz'
    
    compute_correlation_mpi(zarr_path, output_path)
