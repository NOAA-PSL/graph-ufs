#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

output_dir=/p1-evaluation/v1/long-forecasts
forecast_duration="8754h"
t0="2019-01-01T00"
time_start="2019-01-01T06"
time_stop="2019-02-28T18"
surface_variables="surface_pressure,10m_u_component_of_wind,10m_v_component_of_wind,2m_temperature"
level_variables="temperature,specific_humidity,u_component_of_wind,v_component_of_wind,vertical_velocity"
all_variables="${surface_variables},total_precipitation_3hr,layer_thickness,${level_variables}"

levels=100,500,850

truth_names=("era5")
truth_paths=( \
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \
)
rename_variables='{"pressfc":"surface_pressure","ugrd10m":"10m_u_component_of_wind","vgrd10m":"10m_v_component_of_wind","tmp2m":"2m_temperature","tmp":"temperature","ugrd":"u_component_of_wind","vgrd":"v_component_of_wind","dzdt":"vertical_velocity","spfh":"specific_humidity","delz":"layer_thickness","prateb_ave":"total_precipitation_3hr","lat":"latitude","lon":"longitude"}'

# Standard WB2 deterministic evaluation
for dataset in "graphufs"
do

    forecast_path=${output_dir}/${dataset}.${t0}.${forecast_duration}.postprocessed.zarr
    native_forecast_path=${output_dir}/${dataset}.${t0}.${forecast_duration}.zarr

    for i in "${!truth_names[@]}"
    do
        truth_name=${truth_names[i]}
        truth_path=${truth_paths[i]}

        echo "Evaluating ${dataset} against ${truth_name} ..."
        python weatherbench2/scripts/evaluate.py \
          --forecast_path=${forecast_path} \
          --obs_path=${truth_path} \
          --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
          --by_init=False \
          --output_dir=${output_dir} \
          --output_file_prefix=${dataset}_vs_${truth_name}_${forecast_duration}_ \
          --eval_configs=deterministic_temporal \
          --time_start=${time_start} \
          --time_stop=${time_stop} \
          --evaluate_climatology=False \
          --evaluate_persistence=False \
          --variables="${surface_variables},${level_variables}" \
          --levels=${levels} \
          --skipna=True

    done

    #echo "Computing spectra for ${dataset} ..."
    #python weatherbench2/scripts/compute_zonal_energy_spectrum.py \
    #  --input_path=${native_forecast_path} \
    #  --output_path=${output_dir}/${dataset}.${t0}.${forecast_duration}.spectra.zarr \
    #  --base_variables=${surface_variables} \
    #  --time_dim="time" \
    #  --time_start=${time_start} \
    #  --time_stop=${time_stop} \
    #  --levels=${native_levels} \
    #  --averaging_dims="" \
    #  --rename_variables=${rename_variables}
done
