#!/bin/bash

#SBATCH -J training-p2p-dlwsltghSP
#SBATCH -o /pscratch/sd/t/timothys/p2p/dlwsltghSP/slurm/training.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p2p/dlwsltghSP/slurm/training.%j.err
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 36:00:00

conda activate /global/common/software/m4718/timothys/graphufs
export MPI4JAX_USE_CUDA_MPI=1

cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p2p/dlwsltghSP
srun $COMMON/select_gpu_device python train.py
srun $COMMON/select_gpu_device python inference.py
