#!/bin/bash

#SBATCH -J training
#SBATCH -o slurm/stacked_training.%j.out
#SBATCH -e slurm/stacked_training.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=gpu4
#SBATCH -t 120:00:00

source /contrib2/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate graphufs

python stacked_train.py
