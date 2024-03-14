#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

GRAPHUFS_ZARR="$PWD/prototypes/p0/zarr-stores"

python weatherbench2/scripts/evaluate.py \
  --forecast_path=$GRAPHUFS_ZARR/graphufs_predictions.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --by_init=True \
  --output_dir=./ \
  --output_file_prefix=hres_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic \
  --variables="2m_temperature" \
  --levels=100,500,100
#  --time_start=2020-01-01 \
#  --time_stop=2020-12-31 \
