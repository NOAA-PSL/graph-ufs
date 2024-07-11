#!/bin/bash

output_dir=/p1-evaluation/v1/validation
forecast_duration="240h"
time_start="2022-01-01T00"
time_stop="2023-10-13T03"
variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature"
levels=100,500,850

truth_names=("era5" "hres_analysis")
truth_paths=( \
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \
    "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr" \
)

for dataset in "graphufs" "replay"
do

    forecast_path=${output_dir}/${dataset}.${forecast_duration}.postprocessed.zarr

    for i in "${!truth_names[@]}"
    do
        truth_name=${truth_names[i]}
        truth_path=${truth_datasets[i]}

        echo "Evaluating ${dataset} against ${truth_name} ..."
        python weatherbench2/scripts/evaluate.py \
          --forecast_path=${forecast_path} \
          --obs_path=${truth_path} \
          --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
          --by_init=True \
          --output_dir=${output_dir} \
          --output_file_prefix=${dataset}_vs_${truth_name}_${forecast_duration}_ \
          --eval_configs=deterministic,deterministic_temporal,deterministic_spatial \
          --time_start=${time_start} \
          --time_stop=${time_stop} \
          --evaluate_climatology=False \
          --evaluate_persistence=False \
          --variables=${variables} \
          --levels=${levels}
    done

    echo "Computing spectra for ${dataset} ..."
    python weatherbench2/scripts/compute_zonal_energy_spectrum.py \
      --input_path=${forecast_path} \
      --output_path=${output_dir}/${dataset}.${forecast_duration}.spectra.zarr \
      --base_variables=${variables} \
      --time_dim="time" \
      --time_start=${time_start} \
      --time_stop=${time_stop} \
      --levels=${levels} \
      --averaging_dims="time"
done
