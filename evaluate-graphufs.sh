#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

GRAPHUFS_ZARR_DIR="$PWD/prototypes/p1/zarr-stores"

OBS_PATH="gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

python weatherbench2/scripts/evaluate.py \
  --forecast_path=$GRAPHUFS_ZARR_DIR/graphufs_predictions.zarr \
  --obs_path=$OBS_PATH \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --by_init=False \
  --output_dir=./ \
  --output_file_prefix=graphufs_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic \
  --time_start=1994-01-01 \
  --time_stop=1994-01-16 \
  --variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,temperature,u_component_of_wind,v_component_of_wind,vertical_velocity,specific_humidity,geopotential" \
  --levels=50,100,150,200,250,300,400,500,600,700,850,925,1000

