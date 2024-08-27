#!/bin/bash

#SBATCH -J t256
#SBATCH -o slurm/training.256.%j.out
#SBATCH -e slurm/training.256.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=gpu
#SBATCH -t 120:00:00

source /contrib2/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate graphufs

python train.py
