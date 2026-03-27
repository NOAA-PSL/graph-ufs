# Cross-Channel Correlation Analysis

MPI-parallel tools for computing correlation matrices and eigendecomposition for 
large Zarr datasets. Note that this code is optimized for the Perlmutter
supercomputer, but it should be straightforward to adapt it to other machines.

## What It Computes

- **Correlation matrix**: Cross-channel correlations (n_channels × n_channels)
- **Eigenvalues & eigenvectors**: Full spectrum for scree plots and PCA
- **Singular values**: For condition number and ill-conditioning analysis
- **Channel statistics**: Mean and standard deviation per channel

## Files

- `compute_correlation_mpi.py` - MPI-parallel computation script
- `run_correlation_mpi.sh` - SLURM submission script
- `README.md` 

## Setup
If you do not have a conda environment containing mpi4py, xarray, zarr, then
create a conda environment as below:
```bash
conda create -n corr_analysis python=3.11 xarray zarr mpi4py numpy -c conda-forge
```
If you have one already, feel free to use that.

Edit `submit_compute_correlation.sh`:
- Update `#SBATCH --account your_account_here` with your NERSC allocation
- Update `conda activate your_env` with your conda environment name

## Usage

Submit job with run directory name:
```bash
sbatch run_correlation_mpi.sh <RUN_NAME>
```
**Example:**
```bash
sbatch run_correlation_mpi.sh R10
```

**Paths used:**
- Input: `/pscratch/sd/n/nagarwal/ocn-only/<RUN_NAME>/training/inputs.zarr`
- Output: `/pscratch/sd/n/nagarwal/ocn-only/<RUN_NAME>/input_correlation_singvalues.npz`

## Input Data Format

Zarr dataset with dimensions: `(lat, lon, time, channels)`

The code automatically detects the number of channels and processes accordingly.

## Output

Results saved as compressed `.npz` file:

```python
import numpy as np

data = np.load('input_correlation_singvalues.npz')
correlation_matrix = data['correlation_matrix']  # (n_channels, n_channels)
eigenvalues = data['eigenvalues']                # (n_channels,) sorted descending
eigenvectors = data['eigenvectors']              # (n_channels, n_channels)
singular_values = data['singular_values']        # (n_channels,) sorted descending
channel_means = data['channel_means']            # (n_channels,)
channel_stds = data['channel_stds']              # (n_channels,)
```

## Scree Plot Example

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.load('input_correlation_singvalues.npz')
plt.plot(data['eigenvalues'], 'o-')
plt.yscale('log')
plt.xlabel('Component')
plt.ylabel('Eigenvalue')
plt.savefig('scree_plot.png')
```

## Configuration

Default SLURM settings (edit in `run_correlation_mpi.sh`):
- **Nodes**: 8
- **MPI ranks**: 32 (4 per node)
- **Time**: 1.5 hours

Adjust for your dataset size:
```bash
# Smaller datasets
#SBATCH -N 4
#SBATCH -t 1:00:00

# Larger datasets
#SBATCH -N 16
#SBATCH -t 2:00:00
```

## Performance

Typical runtimes for ~1.5 TB datasets:
- 8 nodes (32 ranks): 10 minutes (single precision in/ double precision out)

Memory usage: ~5-6 GB per rank (uses float32 for processing, float64 for final results)

## Troubleshooting

**Dataset not found:**
```bash
ls /pscratch/sd/n/nagarwal/ocn-only/<RUN_NAME>/training/inputs.zarr
```

**Out of memory:**
- Reduce tasks per node: `--ntasks-per-node=2`
- Or use more nodes

**Wrong variable name:**
Check your Zarr variable names and edit line 33 in `compute_correlation_mpi.py`:
```python
data_var = 'your_variable_name'  # Instead of auto-detect
```

## Theory

**Correlation matrix:** For normalized data **X**, computes **C = (1/n) X^T X**

**Singular values:** Related to eigenvalues by **σᵢ = √(λᵢ × (n-1))**
