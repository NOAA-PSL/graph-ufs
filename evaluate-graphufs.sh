#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts


python weatherbench2/scripts/evaluate.py \
  --forecast_path=prototypes/p1/results/v1/validation/predictions.fakeplevel.24h.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --by_init=False \
  --output_dir=./prototypes/p1/results/v1/validation \
  --output_file_prefix=predictions_vs_era5_fakeplevel_24h_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic,deterministic_temporal \
  --time_start=2022-01-01T06 \
  --time_stop=2022-01-02T18 \
  --variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,temperature,u_component_of_wind,v_component_of_wind,vertical_velocity,specific_humidity" \
  --levels=100,500,850

python weatherbench2/scripts/evaluate.py \
  --forecast_path=prototypes/p1/results/v1/validation/replay.fakeplevel.24h.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
  --by_init=False \
  --output_dir=./prototypes/p1/results/v1/validation \
  --output_file_prefix=replay_vs_era5_fakeplevel_24h_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic,deterministic_temporal \
  --time_start=2022-01-01T06 \
  --time_stop=2022-01-02T18 \
  --variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature,temperature,u_component_of_wind,v_component_of_wind,vertical_velocity,specific_humidity" \
  --levels=100,500,850
