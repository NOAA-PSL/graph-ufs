import logging
from functools import partial
import numpy as np
import xarray as xr
import pandas as pd
import jax
import haiku as hk
import flox

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization as graphcastnorm,
    rollout,
)

from graphufs.inference import swap_batch_time_dims
from graphufs.utils import load_checkpoint
from graphufs.fvemulator import fv_vertical_regrid
from graphufs.log import setup_simple_log
from config import P2EvaluationEmulator as Emulator

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
    for norm_component in ["mean", "std", "stddiff"]:
        this_norm = xr.open_zarr(Emulator.norm_urls[norm_component], storage_options={"token":"anon"})
        myvars = list(x for x in get_all_variables(task_config) if x in this_norm)
        # keep attributes in order to distinguish static from time varying components
        with xr.set_options(keep_attrs=True):

            if Emulator.input_transforms is not None:
                for key, transform_function in Emulator.input_transforms.items():

                    # make sure e.g. log_spfh is in the dataset
                    transformed_key = f"{transform_function.__name__}_{key}" # e.g. log_spfh
                    assert transformed_key in this_norm, \
                        f"Emulator.set_normalization: couldn't find {transformed_key} in {component} normalization dataset"
                    # there's a chance the original, e.g. spfh, is not in the dataset
                    # if it is, replace it with e.g. log_spfh
                    if key in myvars:
                        idx = myvars.index(key)
                        myvars[idx] = transformed_key
            this_norm = this_norm[myvars]
            if Emulator.input_transforms is not None:
                for key, transform_function in Emulator.input_transforms.items():
                    transformed_key = f"{transform_function.__name__}_{key}" # e.g. log_spfh
                    idx = myvars.index(transformed_key)
                    myvars[idx] = key

                    # necessary for graphcast.dataset to stacked operations
                    this_norm = this_norm.rename({transformed_key: key})

        this_norm = this_norm.sel(pfull=list(task_config.pressure_levels), method="nearest")
        this_norm = this_norm.rename({"pfull": "level"})
        normalization[norm_component] = this_norm.load()

    return normalization


def load_sample_initial_conditions(t0, lead_times, task_config, sample_idx=0):
    """This pulls two sample timesteps from the Replay dataset on GCS
    and puts it in the form needed to make a forecast with GraphCast
    """
    xds = xr.open_zarr(
        Emulator.data_url,
        storage_options={"token": "anon"},
    )

    # select variables
    varlist = list(get_all_variables(task_config)) + ["delz"]
    xds = xds[[x for x in varlist if x in xds]]

    # drop these
    xds = xds.drop_vars(["cftime", "ftime"])

    # select t0 and t-1
    time_slice = slice(
        pd.Timestamp(t0) - pd.Timedelta(task_config.input_duration) + pd.Timedelta("1s"),
        pd.Timestamp(t0) + pd.Timedelta(lead_times[-1]),
    )
    logging.info("Selecting time slice from replay data based on slice")
    logging.info(f"\tstart = {time_slice.start}")
    logging.info(f"\tstop = {time_slice.stop}")
    xds = xds.sel(time=time_slice)
    logging.info("Replay data has the following date range")
    logging.info(f"\tstart = {xds.time.isel(time=0).values}")
    logging.info(f"\tstop = {xds.time.isel(time=-1).values}")

    # check for ints
    for key in xds.data_vars:
        if "int" in str(xds[key].dtype):
            logging.info(f"Converting {key} from {xds[key].dtype} to float32")
            xds[key] = xds[key].astype(np.float32)

    # regrid vertical levels
    logging.info("Regridding vertical coordinate")
    xds = fv_vertical_regrid(xds, interfaces=list(Emulator.interfaces))

    # rename
    xds = xds.rename({
        "time": "datetime",
        "pfull": "level",
        "grid_yt": "lat",
        "grid_xt": "lon",
    })

    # re-create time dimension as relative to t0
    xds["time"] = xds["datetime"] - xds["datetime"][1]
    xds = xds.swap_dims({"datetime":"time"}).reset_coords()
    xds = xds.set_coords(["datetime"])
    xds.attrs["t0"] = t0

    # transform into inputs, targets_template, forcings
    levels = list(xds["level"].values)
    inputs, targets_template, forcings = data_utils.extract_inputs_targets_forcings(
        xds,
        input_variables=task_config.input_variables,
        target_variables=task_config.target_variables,
        forcing_variables=task_config.forcing_variables,
        pressure_levels=levels,
        input_duration=task_config.input_duration,
        target_lead_times=lead_times,
    )

    # map the inputs to logspace (or whatever)
    for key, mapping in Emulator.input_transforms.items():
        with xr.set_options(keep_attrs=True):
            inputs[key] = mapping(inputs[key])

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

    logging.info("Compiling run_forward.apply")
    gc = drop_state( with_params( jax.jit( with_configs( run_forward.apply ) ) ) )

    logging.info("Starting forecast")
    predictions = rollout.chunked_prediction(
        gc,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )
    return predictions


if __name__ == "__main__":

    setup_simple_log()
    params, model_config, task_config = load_checkpoint("/p1-evaluation/p2/models/model_64.npz")

    normalization = load_normalization(task_config)

    t0 = "2019-01-01T03"
    tf = "2019-12-31T21"
    time = pd.date_range(t0, tf, freq="3h")
    delta_t = time - pd.Timestamp(t0)
    delta_t = delta_t.days*24 + delta_t.seconds /3600
    lead_times = [f"{int(dt)}h" for dt in delta_t[1:]]
    inputs, targets_template, forcings = load_sample_initial_conditions(
        t0=t0,
        lead_times=lead_times,
        task_config=task_config,
    )

    logging.info("Loading inputs and forcings...")
    inputs.load()
    forcings.load()
    logging.info("Done")

    prediction = predict(
        params=params,
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
        model_config=model_config,
        task_config=task_config,
        normalization=normalization,
    )

    # map back from log space
    for key, mapping in Emulator.output_transforms.items():
        with xr.set_options(keep_attrs=True):
            prediction[key] = mapping(prediction[key])

    prediction = swap_batch_time_dims(prediction, [pd.Timestamp(t0)])
    prediction = prediction.rename({"time": "t0"})
    prediction["time"] = pd.Timestamp(t0) + prediction["lead_time"]
    prediction = prediction.swap_dims({"lead_time": "time"})

    path = f"/p1-evaluation/p2/long-forecasts/graphufs.{t0}.{tf}.zarr"
    logging.info(f"Storing prediction at {path}")
    prediction.to_zarr(path, mode="w")
    logging.info(f"Done.")
