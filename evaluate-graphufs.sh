#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

GRAPHUFS_ZARR_DIR="$PWD/prototypes/p0/zarr-stores"

python weatherbench2/scripts/evaluate.py \
  --forecast_path=$GRAPHUFS_ZARR_DIR/graphufs_predictions.zarr \
  --obs_path=$GRAPHUFS_ZARR_DIR/graphufs_targets.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --by_init=False \
  --output_dir=./ \
  --output_file_prefix=graphufs_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic \
  --time_start=1995-01-01 \
  --time_stop=1995-12-31 \
  --variables="surface_pressure,temperature,10m_u_component_of_wind,10m_v_component_of_wind" \
  --levels=100,500,1000

# We need to match dates before comparing against this obs, currently gives NaNs for metrics
#   --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \

