#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

year=2018

python weatherbench2/scripts/evaluate.py \
 --forecast_path=gs://weatherbench2/datasets/graphcast/${year}/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr \
 --obs_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
 --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
 --output_dir=./prototypes/p1/results/v1/validation \
 --output_file_prefix=graphcast_vs_era_${year}_ \
 --by_init=True \
 --input_chunks=init_time=1 \
 --eval_configs=deterministic \
 --evaluate_climatology=False \
 --evaluate_persistence=False \
 --time_start=${year}-01-01T06 \
 --time_stop=${year}-01-03T18 \
 --variables="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure" \
 --levels=100,500,850
