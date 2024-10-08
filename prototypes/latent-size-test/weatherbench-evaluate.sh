#!/bin/bash

forecast_duration="240h"
time_start="2022-01-01T00"
time_stop="2022-12-13T03"
surface_variables="surface_pressure"
level_variables="geopotential"
all_variables="${surface_variables},${level_variables}"

levels=100,500,850
native_levels=998.780701,847.783203,505.652069,231.394791,112.316956,1.124391

rename_variables='{"pressfc":"surface_pressure","lat":"latitude","lon":"longitude"}'


# Standard WB2 deterministic evaluation
for suffix in "-016" "-064" "-128" ""
do

    output_dir=/testlfs/latent-size-test${suffix}/evaluation/validation
    native_forecast_path=${output_dir}/graphufs.${forecast_duration}.zarr

#    # evaluate native against replay
#    echo "Evaluating standard metrics for suffix=${suffix} ..."
#    python ../../weatherbench2/scripts/evaluate.py \
#      --forecast_path=${output_dir}/graphufs.${forecast_duration}.zarr \
#      --obs_path=${output_dir}/replay.${forecast_duration}.zarr \
#      --by_init=True \
#      --output_dir=${output_dir} \
#      --output_file_prefix=graphufs_vs_replay_${forecast_duration}_ \
#      --eval_configs=deterministic,deterministic_spatial,deterministic_temporal \
#      --time_start=${time_start} \
#      --time_stop=${time_stop} \
#      --evaluate_climatology=False \
#      --evaluate_persistence=False \
#      --variables=${all_variables} \
#      --rename_variables=${rename_variables} \
#      --levels=${native_levels}

    echo "Computing spectra for suffix=${suffix} ..."
    python ../../weatherbench2/scripts/compute_zonal_energy_spectrum.py \
      --input_path=${native_forecast_path} \
      --output_path=${output_dir}/graphufs.${forecast_duration}.spectra.zarr \
      --base_variables=${all_variables} \
      --time_dim="time" \
      --time_start=${time_start} \
      --time_stop=${time_stop} \
      --levels=${native_levels} \
      --averaging_dims="time" \
      --rename_variables=${rename_variables}
done
