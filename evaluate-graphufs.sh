#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

forecast_duration="24h"

for dataset in "predictions" "replay"
do
    echo "Evaluating ${dataset} ..."
    python weatherbench2/scripts/evaluate.py \
      --forecast_path=prototypes/p1/results/v1/validation/${dataset}.fakeplevel.${forecast_duration}.zarr \
      --obs_path=gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr \
      --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
      --by_init=True \
      --output_dir=./prototypes/p1/results/v1/validation \
      --output_file_prefix=${dataset}_vs_era5_fakeplevel_${forecast_duration}_ \
      --eval_configs=deterministic,deterministic_temporal \
      --time_start=2022-01-01T06 \
      --time_stop=2022-01-03T18 \
      --evaluate_climatology=False \
      --evaluate_persistence=False \
      --variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature" \
      --levels=100,500,850
    echo " ... done with ${dataset}"
done


