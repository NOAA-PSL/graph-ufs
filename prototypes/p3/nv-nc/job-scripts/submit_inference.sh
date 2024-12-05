#!/bin/bash

#SBATCH -J inference-p3-nv-nc
#SBATCH -o /pscratch/sd/t/timothys/p3/nv-nc/slurm/inference.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p3/nv-nc/slurm/inference.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 00:30:00

conda activate /global/common/software/m4718/timothys/graphufs

cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p3/nv-nc
srun ./select_gpu_device python inference.py
