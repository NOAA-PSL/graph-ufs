#!/bin/bash

#SBATCH -J compute_singvalues
#SBATCH -o /pscratch/sd/n/nagarwal/ocn-only/slurm/correlation-%j.out
#SBATCH -e /pscratch/sd/n/nagarwal/ocn-only/slurm/correlation-%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

# Simple MPI job for correlation computation
# Dataset: 192 x 384 x 40880 x 29 (~1.5 TB)
# Expected runtime: 30-60 minutes

conda activate graphufs-mpi
export OMP_NUM_THREADS=1  # Avoid thread oversubscription

if [ -z "$1" ]; then
    echo "ERROR: No run name provided"
    echo "Usage: sbatch run_correlation_mpi.sh R*"
    exit 1
fi

RUN_DIR="$1"
ZARR_PATH="/pscratch/sd/n/nagarwal/ocn-only/${RUN_DIR}/training/inputs.zarr"
OUTPUT_PATH="/pscratch/sd/n/nagarwal/ocn-only/${RUN_DIR}/input_correlation_singvalues.npz"

# Debug: print the paths
echo "RUN_DIR: ${RUN_DIR}"
echo "ZARR_PATH: ${ZARR_PATH}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"

# Verify dataset exists
if [ ! -d "${ZARR_PATH}" ]; then
    echo "ERROR: Dataset not found at ${ZARR_PATH}"
    exit 1
fi

echo "Job started at $(date)"
echo "Using $SLURM_NTASKS MPI ranks"
echo "Dataset: $ZARR_PATH"

srun -n $SLURM_NTASKS python -u compute_correlation_singvalues_mpi.py $ZARR_PATH $OUTPUT_PATH

echo "Job completed at $(date)"
