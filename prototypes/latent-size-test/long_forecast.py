from functools import partial
import xarray as xr
import pandas as pd
import jax
import haiku as hk

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization as graphcastnorm,
    rollout,
)

from evaluate import swap_batch_time_dims

_norm_urls = {
    "mean": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/mean_by_level.zarr",
    "std": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/stddev_by_level.zarr",
    "stddiff": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/diffs_stddev_by_level.zarr",
}

_data_url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"

def load_checkpoint(path):

    with open(path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    params = ckpt.params
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    return params, model_config, task_config


def construct_wrapped_graphcast(model_config, task_config, normalization):

    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = graphcastnorm.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=normalization["stddiff"],
        mean_by_level=normalization["mean"],
        stddev_by_level=normalization["std"],
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(inputs, targets_template, forcings, model_config, task_config, normalization):
    predictor = construct_wrapped_graphcast(model_config, task_config, normalization)
    return predictor(inputs, targets_template, forcings)

def get_all_variables(task_config):
    return tuple(set(
        task_config.input_variables + task_config.target_variables + task_config.forcing_variables
    ))

def load_normalization(task_config):
    """Assumes we want normalization from the 1/4 degree subsampled to ~1 degree data"""

    normalization = dict()
    for key in ["mean", "std", "stddiff"]:
        this_norm = xr.open_zarr(_norm_urls[key], storage_options={"token":"anon"})
        this_norm = this_norm[[x for x in get_all_variables(task_config) if x in this_norm]]
        this_norm = this_norm.sel(pfull=list(task_config.pressure_levels), method="nearest")
        this_norm = this_norm.rename({"pfull": "level"})
        normalization[key] = this_norm.load()

    return normalization


def load_sample_initial_conditions(t0, lead_times, task_config, sample_idx=0):
    """This pulls two sample timesteps from the Replay dataset on GCS
    and puts it in the form needed to make a forecast with GraphCast
    """
    xds = xr.open_zarr(
        _data_url,
        storage_options={"token": "anon"},
    )
    # select t0 and t-1
    time_slice = slice(
        pd.Timestamp(t0) - pd.Timedelta(task_config.input_duration) + pd.Timedelta("1s"),
        pd.Timestamp(t0) + pd.Timedelta(lead_times[-1]),
    )
    xds = xds.sel(time=time_slice)

    # select vertical levels
    xds = xds.sel(pfull=list(task_config.pressure_levels), method="nearest")

    # select variables
    xds = xds[[x for x in get_all_variables(task_config) if x in xds]]

    xds["land_static"] = xds["land_static"].astype(float)

    # drop these
    xds = xds.drop_vars(["cftime", "ftime"])

    # rename
    xds = xds.rename({
        "time": "datetime",
        "pfull": "level",
        "grid_yt": "lat",
        "grid_xt": "lon",
    })

    # re-create time dimension as relative to t-1
    xds["time"] = xds["datetime"] - xds["datetime"][1]
    xds = xds.swap_dims({"datetime":"time"}).reset_coords()
    xds = xds.set_coords(["datetime"])
    xds.attrs["t0"] = t0

    # transform into inputs, targets_template, forcings
    levels = list(
        xds["level"].sel(
            level=list(task_config.pressure_levels),
            method="nearest",
        ).values
    )
    inputs, targets_template, forcings = data_utils.extract_inputs_targets_forcings(
        xds,
        input_variables=task_config.input_variables,
        target_variables=task_config.target_variables,
        forcing_variables=task_config.forcing_variables,
        pressure_levels=levels,
        input_duration=task_config.input_duration,
        target_lead_times=lead_times,
    )

    inputs = inputs.expand_dims({"batch": [sample_idx]})
    targets_template = targets_template.expand_dims({"batch": [sample_idx]})
    forcings = forcings.expand_dims({"batch": [sample_idx]})
    return inputs, targets_template, forcings


def predict(
    params: dict,
    inputs: xr.DataArray,
    targets_template: xr.DataArray,
    forcings: xr.DataArray,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    normalization: dict,
) -> xr.Dataset:

    state = dict()

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    def with_configs(fn):
        return partial(
            fn,
            model_config=model_config,
            task_config=task_config,
            normalization=normalization,
        )

    gc = drop_state( with_params( jax.jit( with_configs( run_forward.apply ) ) ) )

    predictions = rollout.chunked_prediction(
        gc,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )
    return predictions


if __name__ == "__main__":

    t0 = "2019-01-01T00"
    tf = "2019-03-31T18"
    time = pd.date_range(t0, tf, freq="3h")
    delta_t = time - pd.Timestamp(t0)
    delta_t = delta_t.days*24 + delta_t.seconds /3600
    lead_times = [f"{int(dt)}h" for dt in delta_t[1:]]

    for latent_size in [16, 64, 128, 256]:
        params, model_config, task_config = load_checkpoint(f"/testlfs/latent-size-test-{latent_size:03d}/models/model_64.npz")

        normalization = load_normalization(task_config)

        inputs, targets_template, forcings = load_sample_initial_conditions(
            t0=t0,
            lead_times=lead_times,
            task_config=task_config,
        )

        inputs.load()
        forcings.load()

        prediction = predict(
            params=params,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
            model_config=model_config,
            task_config=task_config,
            normalization=normalization,
        )

        prediction = swap_batch_time_dims(prediction, [pd.Timestamp(t0)])
        prediction = prediction.rename({"time": "t0"})
        prediction["time"] = pd.Timestamp(t0) + prediction["lead_time"]
        prediction = prediction.swap_dims({"lead_time": "time"})

        duration = lead_times[-1]
#        prediction = prediction.chunk({
#            "lat": -1,
#            "lon": -1,
#            "level": -1,
#            "time": 1,
#        })
        prediction.to_zarr(
            f"/testlfs/latent-size-test-{latent_size:03d}/evaluation/long-forecast/graphufs.{t0}.{duration}.zarr",
            mode="w",
        )
        print(f"Done with latent_size={latent_size}")
