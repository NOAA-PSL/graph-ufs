#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/weatherbench2:$PWD/weatherbench2/weatherbench2:$PWD/weatherbench2/scripts

year=2018
output_dir=/p2-lustre/baselines
time_start="${year}-01-01T00"
time_stop="${year}-12-31T23"
time_stride=3
variables="2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,geopotential,temperature,specific_humidity,u_component_of_wind,v_component_of_wind"

model_names=("ifs_ens_mean" "graphcast" "pangu")
native_model_paths=( \
    "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr" \
    "gs://weatherbench2/datasets/graphcast/2018/date_range_2017-11-16_2019-02-01_12_hours.zarr" \
    "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
)

for i in "${!model_names[@]}"
do

    model_name=${model_names[i]}
    native_model_path=${native_model_paths[i]}

    if [[ ${model_name} == "ifs_ens_mean" ]] ; then
        levels=500,850
    else
        levels=100,250,500,850
    fi

    rename_variables="None"
    configs="deterministic"
    if [[ ${model_name} == "graphcast" ]]  ; then
        rename_variables="{'lat':'latitude','lon':'longitude'}"
        configs="deterministic,deterministic_spatial"
    fi

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
