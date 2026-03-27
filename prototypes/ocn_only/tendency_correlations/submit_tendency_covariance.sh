#!/bin/bash

#SBATCH -J calc_tendency_covariance_dask
#SBATCH -o /pscratch/sd/n/nagarwal/ocn-only/slurm/calc_tend_cov.%j.out
#SBATCH -e /pscratch/sd/n/nagarwal/ocn-only/slurm/calc_tend_cov.%j.err
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

conda activate graphufs-mpi
cd /global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/tendency_correlations

# To compute correlation with seasonality removed, use below
# Note, that for the boolean args, only use them if you want to set the to True,
# otherwise they are false by default. Use --no_spatial_avg if you don't want
# spatially averaged statistics.
python calc_tendency_covariance_dask_rechunk_in_memory.py --prototype R5 --num_missing_samples 3 --norm --remove_seasonality --no_spatial_avg
