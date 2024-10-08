#!/bin/bash

#SBATCH -J t064
#SBATCH -o slurm/training.064.%j.out
#SBATCH -e slurm/training.064.%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=gpu
#SBATCH -t 120:00:00

source /contrib2/Tim.Smith/miniconda3/etc/profile.d/conda.sh
conda activate graphufs

python train.py --latent-size=64 --local-store-path=/testlfs/latent-size-test-064
