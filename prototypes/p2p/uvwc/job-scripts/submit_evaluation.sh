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

# cleanup, copy to community
mywork=$WORK/p2p/uvwc
mycommunity=$COMMUNITY/p2p/uvwc
mkdir -p $mycommunity/inference/validation
mkdir -p $mycommunity/logs/training
mkdir -p $mycommunity/logs/inference

cp $mywork/loss.nc $mycommunity
cp -r $mywork/models $mycommunity
cp $mywork/logs/training/*.00.*.* $mycommunity/logs/training
cp $mywork/logs/inference/*.00.*.* $mycommunity/logs/inference
cp $mywork/inference/validation/*.nc $mycommunity/inference/validation
cp -r $mywork/inference/validation/graphufs*.zarr $mycommunity/inference/validation

cd $mycommunity/inference/validation
mkdir to-psl
cp *.nc to-psl/
cp -r graphufs.240h.spectra.zarr to-psl/
tar -zcvf to-psl.tar.gz to-psl/
