#!/bin/bash

#SBATCH -J append_bathymetry
#SBATCH -o /pscratch/sd/n/nagarwal/ocn-only/slurm/append_bathymetry.%j.out
#SBATCH -e /pscratch/sd/n/nagarwal/ocn-only/slurm/append_bathymetry.%j.err
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 00:30:00

conda activate graphufs-mpi

python append_bathymetry.py \
    --input_path="/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/bathymetry/regrid-bathymetry/bathymetry.gaussian.0.25-degree-subsampled.zarr" \
    --output_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.zarr" \
    --num_workers=4

python append_bathymetry.py \
    --input_path="/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/bathymetry/regrid-bathymetry/bathymetry.gaussian.0.25-degree.zarr" \
    --output_path="gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/06h-freq/zarr/mom6.zarr" \
    --num_workers=4
