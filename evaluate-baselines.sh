#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

year=2018
output_dir=/p1-evaluation/v1/validation
time_start="${year}-01-01T00"
time_stop="${year}-12-31T23"
time_stride=3
variables="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,temperature,specific_humidity,u_component_of_wind,v_component_of_wind"

truth_names=("hres_analysis" "era5")
truth_paths=( \
    "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr" \
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr" \

)

model_names=("ifs_ens_mean" "graphcast" "pangu")
regridded_model_paths=( \
    "gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr" \
    "gs://weatherbench2/datasets/graphcast/${year}/date_range_2017-11-16_2019-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr" \
    "gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr" \
)
native_model_paths=( \
    "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr" \
    "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr" \
    "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
)

for i in "${!model_names[@]}"
do

    model_name=${model_names[i]}
    regridded_model_path=${regridded_model_paths[i]}
    native_model_path=${native_model_paths[i]}

    if [[ ${model_name} == "ifs_ens_mean" ]] ; then
        levels=500,850
    else
        levels=100,500,850
    fi

    rename_variables="None"
    configs="deterministic"
    if [[ ${model_name} == "graphcast" ]]  ; then
        rename_variables="{'lat':'latitude','lon':'longitude'}"
        if [[ ${truth_name} == "hres_analysis" ]] ; then
            configs="deterministic,deterministic_spatial,deterministic_temporal"
        fi
    fi

    for j in "${!truth_names[@]}"
    do
        truth_name=${truth_names[j]}
        truth_path=${truth_paths[j]}

        echo "Evaluating ${model_name} against ${truth_name} ..."
        python weatherbench2/scripts/evaluate.py \
         --forecast_path=${regridded_model_path} \
         --obs_path=${truth_path} \
         --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr \
         --output_dir=${output_dir} \
         --output_file_prefix=${model_name}_vs_${truth_name}_${year}_ \
         --by_init=True \
         --input_chunks=init_time=1 \
         --eval_configs=${configs} \
         --evaluate_climatology=False \
         --evaluate_persistence=False \
         --time_start=${time_start} \
         --time_stop=${time_stop} \
         --time_stride=${time_stride} \
         --variables=${variables} \
         --levels=${levels}
    done

    echo "Computing spectra for ${model_name} ..."
    python weatherbench2/scripts/compute_zonal_energy_spectrum.py \
      --input_path=${native_model_path} \
      --output_path=${output_dir}/${model_name}.${year}.spectra.zarr \
      --base_variables=${variables} \
      --time_dim="time" \
      --time_start=${time_start} \
      --time_stop=${time_stop} \
      --time_stride=${time_stride} \
      --levels=${levels} \
      --averaging_dims="time" \
      --rename_variables=${rename_variables}
done
