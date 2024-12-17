#!/bin/bash

#SBATCH -J training-p3-uvwc
#SBATCH -o /pscratch/sd/t/timothys/p3/uvwc/slurm/training.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p3/uvwc/slurm/training.%j.err
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 36:00:00

conda activate /global/common/software/m4718/timothys/graphufs
export MPI4JAX_USE_CUDA_MPI=1

cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p3/uvwc
srun $COMMON/select_gpu_device python train.py
srun $COMMON/select_gpu_device python inference.py
