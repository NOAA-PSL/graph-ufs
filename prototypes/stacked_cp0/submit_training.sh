#!/bin/bash

#SBATCH -J training-stacked-cp0
#SBATCH -o /global/homes/n/nagarwal/graph-ufs/prototypes/stacked_cp0/slurm/training.%j.out
#SBATCH -e /global/homes/n/nagarwal/graph-ufs/prototypes/stacked_cp0/slurm/training.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 06:00:00

conda activate graphufs-mpi
export MPI4JAX_USE_CUDA_MPI=1

cd /global/homes/n/nagarwal/graph-ufs/prototypes/stacked_cp0
python train.py
