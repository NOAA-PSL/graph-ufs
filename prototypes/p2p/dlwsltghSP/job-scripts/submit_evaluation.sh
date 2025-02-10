#!/bin/bash

#SBATCH -J eval-p2p-dlwsltghSP
#SBATCH -o /pscratch/sd/t/timothys/p2p/dlwsltghSP/slurm/evaluation.%j.out
#SBATCH -e /pscratch/sd/t/timothys/p2p/dlwsltghSP/slurm/evaluation.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=regular
#SBATCH --account=m4718
#SBATCH --constraint=cpu
#SBATCH -t 03:00:00

conda activate graphufs-cpu
cd /global/common/software/m4718/timothys/graph-ufs/prototypes/p2p/dlwsltghSP

python postprocess_inference.py
. ./evaluate_with_wb2.sh

# cleanup, copy to community
mywork=$WORK/p2p/dlwsltghSP
mycommunity=$COMMUNITY/p2p/dlwsltghSP
mkdir -p $mycommunity/inference/validation
mkdir -p $mycommunity/logs/training
mkdir -p $mycommunity/logs/inference

mv $mywork/loss.nc $mycommunity
mv $mywork/models $mycommunity
cp $mywork/logs/training/*.00.*.* $mycommunity/logs/training
cp $mywork/logs/inference/*.00.*.* $mycommunity/logs/inference
mv $mywork/inference/validation/*.nc $mycommunity/inference/validation
mv $mywork/inference/validation/graphufs*.zarr $mycommunity/inference/validation

cd $mycommunity/inference/validation
mkdir to-psl
cp *.nc to-psl/
cp -r graphufs.240h.spectra.zarr to-psl/
tar -zcvf to-psl.tar.gz to-psl/
