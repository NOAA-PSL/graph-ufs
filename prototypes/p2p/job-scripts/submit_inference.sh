#!/bin/bash

#SBATCH -J p2pinference
#SBATCH -o /pscratch/sd/t/timothys/p2p/slurm/inference.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p2p/slurm/inference.%j.err
#SBATCH --nodes=8
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=gpu&hbm80g
#SBATCH -t 00:30:00

conda activate /global/common/software/m4718/timothys/graphufs

cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p2p
srun ./select_gpu_device python inference.py
