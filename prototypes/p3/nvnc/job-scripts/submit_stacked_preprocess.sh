#!/bin/bash

#SBATCH -J preprocess
#SBATCH -o /pscratch/sd/t/timothys/p2p/nvnc/slurm/preprocess.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p2p/nvnc/slurm/preprocess.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 24:00:00

conda activate /global/common/software/m4718/timothys/graphufs
python -c "from preprocess import store_batch_of_samples
store_batch_of_samples('training')
store_batch_of_samples('validation')
"
