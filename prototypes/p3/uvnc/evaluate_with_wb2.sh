#!/bin/bash

# Note that we have to have pip install --no-deps weatherbench2 as in the README

output_dir=/pscratch/sd/t/timothys/p2p/uvnc/inference/validation
wb2_dir=$COMMON/graph-ufs/weatherbench2
forecast_duration="240h"
time_start="2022-01-01T00"
time_stop="2023-10-13T03"
surface_variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature"
replay_variables="2m_specific_humidity"
level_variables="temperature,specific_humidity,u_component_of_wind,v_component_of_wind,vertical_velocity"
diagnosed_variables="geopotential"

levels=250,500,850
native_levels='226.08772546708585,522.5402821445465,825.8226804542542,874.7199656200409,974.296663646698'

truth_names=("era5" "hres_analysis")
truth_paths=( \
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \
    "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr" \
)
rename_variables='{"pressfc":"surface_pressure","ugrd10m":"10m_u_component_of_wind","vgrd10m":"10m_v_component_of_wind","tmp2m":"2m_temperature","tmp":"temperature","ugrd":"u_component_of_wind","vgrd":"v_component_of_wind","dzdt":"vertical_velocity","spfh":"specific_humidity","spfh2m":"2m_specific_humidity","lat":"latitude","lon":"longitude"}'

# Standard WB2 deterministic evaluation
for dataset in "graphufs" "replay"
do

    forecast_path=${output_dir}/${dataset}.${forecast_duration}.postprocessed.zarr
    native_forecast_path=${output_dir}/${dataset}.${forecast_duration}.zarr

    if [[ ${dataset} == "replay" ]] ; then
        by_init="False"
    else
        by_init="True"
    fi

    for i in "${!truth_names[@]}"
    do
        truth_name=${truth_names[i]}
        truth_path=${truth_paths[i]}

        echo "Evaluating ${dataset} against ${truth_name} ..."
        python ${wb2_dir}/scripts/evaluate.py \
          --forecast_path=${forecast_path} \
          --obs_path=${truth_path} \
          --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
          --by_init=${by_init} \
          --output_dir=${output_dir} \
          --output_file_prefix=${dataset}_vs_${truth_name}_${forecast_duration}_ \
          --eval_configs=deterministic,deterministic_spatial \
          --time_start=${time_start} \
          --time_stop=${time_stop} \
          --evaluate_climatology=False \
          --evaluate_persistence=False \
          --variables="${surface_variables},${level_variables},${diagnosed_variables}" \
          --levels=${levels} \
          --skipna=True

    done

    if [[ ${dataset} == "graphufs" ]]; then
        echo "Computing spectra for ${dataset} ..."
        python ${wb2_dir}/scripts/compute_zonal_energy_spectrum.py \
          --input_path=${native_forecast_path} \
          --output_path=${output_dir}/${dataset}.${forecast_duration}.spectra.zarr \
          --base_variables="${surface_variables},${level_variables}" \
          --time_dim="time" \
          --time_start=${time_start} \
          --time_stop=${time_stop} \
          --levels=${native_levels} \
          --averaging_dims="time" \
          --rename_variables=${rename_variables}
    fi
done

# evaluate native against replay
echo "Comparing GraphUFS vs Replay"
python ${wb2_dir}/scripts/evaluate.py \
  --forecast_path=${output_dir}/graphufs.${forecast_duration}.zarr \
  --obs_path=${output_dir}/replay.vertical_regrid.zarr \
  --by_init=True \
  --output_dir=${output_dir} \
  --output_file_prefix=graphufs_vs_replay_${forecast_duration}_ \
  --eval_configs=deterministic,deterministic_spatial \
  --time_start=${time_start} \
  --time_stop=${time_stop} \
  --evaluate_climatology=False \
  --evaluate_persistence=False \
  --variables="${surface_variables},${level_variables},${replay_variables}" \
  --rename_variables=${rename_variables} \
  --levels=${native_levels}
