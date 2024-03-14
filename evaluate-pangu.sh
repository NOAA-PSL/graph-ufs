#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

python weatherbench2/scripts/evaluate.py \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr \
  --output_dir=./ \
  --output_file_prefix=hres_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic,deterministic_spatial,deterministic_temporal \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed

