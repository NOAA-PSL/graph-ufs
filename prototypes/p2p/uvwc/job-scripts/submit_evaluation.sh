#!/bin/bash

#SBATCH -J eval-p2p-uvwc
#SBATCH -o /pscratch/sd/t/timothys/p2p/uvwc/slurm/evaluation.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p2p/uvwc/slurm/evaluation.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 03:00:00

conda activate graphufs-cpu
cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p2p/uvwc

python postprocess_inference.py
. ./evaluate_with_wb2.sh
