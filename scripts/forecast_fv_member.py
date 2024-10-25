"""Standalone script to run GraphUIFS forecast

Example usage:

    $ python forecast_fv_member.py \
            --ensemble_member=10 \
            --previous_cycle=2021120106 \
            --experiment_directory=/work2/noaa/gsienkf/whitaker/C96L127_graphufs_psonlyiau \
            --fhr_ics=6,9

For help with the inputs:

    $ python forecast.py --help

Note:
    For the first cycle, we want to read fhr06 and fhr09 (--fhr_ics=6,9)
    After that we will just make 6 hour predictions with the ML models, so we just want fhr03 and fhr06 (--fhr_ics=3,6)
"""
import os
import sys
from typing import Optional
from functools import partial
import logging
import numpy as np
import xarray as xr
import pandas as pd
import jax
import haiku as hk
import flox

from absl import app, flags

from cftime import DatetimeJulian
from datetime import datetime

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization as graphcastnorm,
    rollout,
    solar_radiation,
)

from ufs2arco import Layers2Pressure


ENSEMBLE_MEMBER = flags.DEFINE_integer(
    "ensemble_member",
    None,
    help="Ensemble member ID to propagate",
)
PREVIOUS_CYCLE = flags.DEFINE_string(
    "previous_cycle",
    None,
    help="Previous DA cycle to use for forecast Initial Conditions, e.g. '2021120106'",
)
EXPERIMENT_DIRECTORY = flags.DEFINE_string(
    "experiment_directory",
    None,
    help="Outermost directory, containing all the DA cycles.",
)
FHR_ICS = flags.DEFINE_list(
    "fhr_ics",
    None,
    help="Forecast hours to grab for initial conditions. It will be --fhr_ics=6,9 when initializing from a single-IC model (e.g. UFS), and --fhr_ics=3,6 when initializing from a 2-IC ML model",
)
MODEL_WEIGHTS = flags.DEFINE_string(
    "model_weights",
    "/work2/noaa/gsienkf/timsmith/model-weights/graphufs_p2_ckpt64.npz",
    help="Path to model weights to be used with GraphCast code base",
)

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

_input_transforms = {
    "spfh": log,
    "spfh2m": log,
}
_output_transforms = {
    "spfh": exp,
    "spfh2m": exp,
}
_interfaces = tuple(x for x in range(200, 1001, 50))

_outerpath = "/work2/noaa/gsienkf/timsmith/replay-normalization-statistics/0.25-degree-subsampled/03h-freq/fv3.fvstatistics.trop16.1993-2019/"
_norm_paths = {
    "mean": f"{_outerpath}/mean_by_level.nc",
    "std": f"{_outerpath}/stddev_by_level.nc",
    "stddiff": f"{_outerpath}/diffs_stddev_by_level.nc",
}



class SimpleFormatter(logging.Formatter):
    def format(self, record):
        record.relativeCreated = record.relativeCreated // 1000
        return super().format(record)

def setup_log(level=logging.INFO):

    logging.basicConfig(
        stream=sys.stdout,
        level=level,
    )
    logger = logging.getLogger()
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        handler.setFormatter(formatter)



def _cycle2timestamp(cycle):
    return pd.Timestamp(f"{cycle[:4]}-{cycle[4:6]}-{cycle[6:8]}T{cycle[8:10]}")

def _timestamp2cycle(timestamp):
    cycle = str(timestamp)
    return cycle.replace("-","").replace("T","").replace(" ","").replace(":","")[:10]

def _cftime2datetime(cf_time):

    time = np.array([
        pd.Timestamp(
            datetime(
                int(t.dt.year),
                int(t.dt.month),
                int(t.dt.day),
                int(t.dt.hour),
                int(t.dt.minute),
                int(t.dt.second),
            )
        )
        for t in cf_time
    ])
    return xr.DataArray(
        time,
        coords=cf_time.coords,
        attrs={
            "description": "valid time",
        },
    )

def _datetime2cftime(datetime):
    cf_time = np.array([
        DatetimeJulian(
            int(t.dt.year),
            int(t.dt.month),
            int(t.dt.day),
            int(t.dt.hour),
            int(t.dt.minute),
            int(t.dt.second),
            has_year_zero=False,
        )
        for t in datetime
    ])
    return xr.DataArray(
        cf_time,
        coords=datetime.coords,
        attrs={
            "long_name": "time",
            "axis": "T",
            "calendar_type": "JULIAN",
        },
    )


def postprocess(xds, last_ic_datetime):
    """Take prediction from GraphCast and prepare it for FV3/DA"""

    rename = {
        "level": "pfull",
        "lat": "grid_yt",
        "lon": "grid_xt",
    }
    for key, val in rename.items():
        if key in xds:
            xds = xds.rename({key: val})

    xds = xds.squeeze().drop_vars("batch")
    if "time" in xds:
        xds = xds.rename({"time": "ftime"})

        datetime = last_ic_datetime + xds["ftime"]
        xds["time"] = _datetime2cftime(datetime)

        # keep ftime as the main dimension to make storing the result easier
        xds = xds.set_coords("time")

    for key, mapping in _output_transforms.items():
        if key in xds:
            logging.info(f"Mapping {key} -> {mapping.__name__}({key})")
            with xr.set_options(keep_attrs=True):
                xds[key] = mapping(xds[key])

    return xds

def store_result(xds, prefix_mapper, cycle, member, experiment_directory):

    outer_dir = f"{experiment_directory}/{cycle}"
    if not os.path.isdir(outer_dir):
        os.makedirs(outer_dir)

    for ftime in xds.ftime.values:
        fhr = int(ftime / 3600 / 1e9)
        result = xds.sel(ftime=[ftime])
        result = result.swap_dims({"ftime": "time"}).drop_vars("ftime")
        for prefix in ["sfg", "bfg"]:
            fname = f"{outer_dir}/{prefix}_{cycle}_fhr{fhr:02d}_mem{member:03d}"
            tmpds = xr.Dataset({
                key: result[key]
                for key in result.data_vars if prefix_mapper[key] == prefix or key == "pressfc"
            },
            attrs=xds.attrs["file_attrs"][prefix.replace("sfg","sanl")]
            )

            # rename _static variables here...
            for key in tmpds.data_vars:
                if "_static" in key:
                    with xr.set_options(keep_attrs=True):
                        newkey = key.replace("_static","")
                        logging.info(f"Before storing to {prefix}, renaming {key} -> {newkey}")
                        tmpds = tmpds.rename({key: newkey})
            if "ak" in xds.attrs and prefix == "sfg" and "ak" not in tmpds.attrs:
                tmpds.attrs["ak"] = xds.ak
                tmpds.attrs["bk"] = xds.bk
            tmpds.to_netcdf(fname)
            logging.info(f"Stored {fname}")


def get_all_variables(task_config):
    return tuple(set(
        task_config.input_variables + task_config.target_variables + task_config.forcing_variables
    ))


def get_empty_targets(lead_times, template):
    """Creates array of zeros at target lead times, basd on template dataset"""

    lead_times, duration = data_utils._process_target_lead_times_and_get_duration(lead_times)
    container = xr.concat(
        list(
            xr.zeros_like(
                template.isel(time=0, drop=True).expand_dims(
                    {"time": [t]}
                )
            )
            for t in lead_times
        ),
        dim="time",
    )
    container["datetime"] = xr.DataArray(
        [template.datetime.values[-1] + t for t in container.time.values],
        coords=container["time"].coords,
    )
    container = container.set_coords("datetime")
    return container

def get_forcings(lead_times, ids):
    """This is a hack for now, since the forcings component is a bit wonky"""

    # setup container for the forcings

    fds = xr.Dataset()
    for key in ["lat", "lon"]:
        fds[key] = ids[key]

    time = [pd.Timedelta(t) for t in lead_times]
    datetime = [pd.Timestamp(ids.datetime.values[-1]) + pd.Timedelta(t) for t in lead_times]
    fds["time"] = xr.DataArray(
        time,
        coords={"time": time},
        dims="time",
    )
    fds["datetime"] = xr.DataArray(
        datetime,
        coords=fds.time.coords,
    )
    fds = fds.set_coords("datetime")

    # First, use code in GraphCast repo to compute downward shortwave radiation
    # note that this is numerically identical to Replay, especially given that
    # this will be converted to half precision
    fds["dswrf_avetoa"] = solar_radiation.get_toa_incident_solar_radiation_for_xarray(
        fds,
        integration_period="3h",
    )
    # GraphCast computes temporal accumulation of solar radiation
    # whereas our model expects the instantaneous flux, averaged over the same time period
    fds["dswrf_avetoa"] = fds["dswrf_avetoa"] / (3600 * 3)
    return fds



def get_model_inputs(xds, member, lead_times, task_config):
    """Transform initial conditions from sanl/bfg files to inputs, targets_template, forcings
    so they're ready to be passed to the model
    """

    # append template and forcings for the target lead times
    targets_template = get_empty_targets(lead_times, xds[[x for x in task_config.target_variables if x in xds]])
    forcings = get_forcings(lead_times, xds[[x for x in task_config.forcing_variables if x in xds]])
    future = xr.merge([targets_template, forcings])
    static = {key: xds[key] for key in xds.data_vars if "time" not in xds[key].dims}
    for key in static.keys():
        xds = xds.drop_vars(key)
    xds = xr.concat([xds, future], dim="time")
    for key, xda in static.items():
        xds[key] = xda

    # check for NaNs and ints
    for key in xds.data_vars:
        isnans = np.isnan(xds[key]).any(["lat", "lon"])
        if isnans.any().values:
            logging.warning(f"{key} found NaNs: \n{isnans}\n")
        if "int" in str(xds[key].dtype):
            logging.info(f"Converting {key} from {xds[key].dtype} to float32")
            xds[key] = xds[key].astype(np.float32)

    # Note that by here, dataset should have the correct vertical coordinate
    levels = list(xds["level"].values)

    # transform into inputs, targets_template, forcings
    inputs, targets_template, forcings = data_utils.extract_inputs_targets_forcings(
        xds,
        input_variables=task_config.input_variables,
        target_variables=task_config.target_variables,
        forcing_variables=task_config.forcing_variables,
        pressure_levels=levels,
        input_duration=task_config.input_duration,
        target_lead_times=lead_times,
    )
    inputs = inputs.expand_dims({"batch": [member]})
    targets_template = targets_template.expand_dims({"batch": [member]})
    forcings = forcings.expand_dims({"batch": [member]})
    return inputs, targets_template, forcings

def get_new_vertical_grid(interfaces):


    # Create the parent vertical grid via layers2pressure object
    replay_layers = Layers2Pressure()
    phalf = replay_layers.phalf.sel(phalf=interfaces, method="nearest")

    # Make a new Layers2Pressure object, which has the subsampled vertical grid
    # note that pfull gets defined internally
    child_layers = Layers2Pressure(
        ak=replay_layers.xds["ak"].sel(phalf=phalf),
        bk=replay_layers.xds["bk"].sel(phalf=phalf),
    )
    nds = child_layers.xds.copy(deep=True)
    return nds


def fv_vertical_regrid(xds, interfaces, keep_delz=False):
    """Vertically regrid a dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in interfaces

    Args:
        xds (xr.Dataset)
        interfaces (array_like)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """
    # create a new dataset with the new vertical grid
    nds = get_new_vertical_grid(interfaces)

    # if the dataset has somehow already renamed pfull -> level, rename to pfull for Layers2Pressure computations
    has_level_not_pfull = False
    if "level" in xds.dims and "pfull" not in xds.dims:
        with xr.set_options(keep_attrs=True):
            xds = xds.rename({"level": "pfull"})

    # Regrid vertical distance, and get weighting
    delz = xds["delz"].groupby_bins(
        "pfull",
        bins=nds["phalf"],
    ).sum()
    new_delz_inverse = 1/delz

    # Do the regridding
    vars2d = [x for x in xds.data_vars if "pfull" not in xds[x].dims]
    vars3d = [x for x in xds.data_vars if "pfull" in xds[x].dims and x != "delz"]
    for key in vars3d:
        with xr.set_options(keep_attrs=True):
            nds[key] = new_delz_inverse * (
                (
                    xds[key]*xds["delz"]
                ).groupby_bins(
                    "pfull",
                    bins=nds["phalf"],
                ).sum()
            )
        nds[key].attrs = xds[key].attrs.copy()

    nds = nds.set_coords("pfull")
    nds["pfull_bins"] = nds["pfull_bins"].swap_dims({"pfull_bins": "pfull"})
    for key in vars3d:
        with xr.set_options(keep_attrs=True):
            nds[key] = nds[key].swap_dims({"pfull_bins": "pfull"})
        nds[key].attrs["regridding"] = "delz weighted average in vertical, new coordinate bounds represented by 'phalf'"
    for v in vars2d:
        nds[v] = xds[v]

    if keep_delz:
        delz = delz.swap_dims({"pfull_bins": "pfull"})
        nds["delz"] = delz

    # unfortunately, cannot store the pfull_bins due to this issue: https://github.com/pydata/xarray/issues/2847
    nds = nds.drop_vars("pfull_bins")
    return nds

def preprocess(xds, cycle, relative_time, task_config, prefix):
    """

    Args:
        xds (xr.Dataset): dataset from a single file, could be sanl, sfg, bfg
        cycle (str): DA cycle
        relative_hours (float, int): time relative to t0, t0 should be 0
        task_config (TaskConfig): for GraphCast
    """

    # select variables
    is_from_ufs = "o3mr" in xds
    xds = xds[[x for x in get_all_variables(task_config)+("delz", "land", "hgtsfc") if x in xds]]

    if is_from_ufs and "pfull" in xds.dims:
        xds = fv_vertical_regrid(xds, interfaces=list(_interfaces))

    if "land" in xds.data_vars:
        xds["land_static"] = xr.where(xds["land"].isel(time=0, drop=True) == 1, 1, 0).astype(np.float32)
        xds = xds.drop_vars("land")

    if "hgtsfc" in xds.data_vars:
        xds["hgtsfc_static"] = xds["hgtsfc"].isel(time=0, drop=True)
        xds = xds.drop_vars("hgtsfc")

    # ak/bk are coordinates, but I think it's expected that they're attributes
    for key in ["ak", "bk"]:
        if key in xds.coords or key in xds.data_vars:
            val = xds[key].values
            xds = xds.drop_vars(key)
            xds.attrs[key] = val

    # convert datetime format from cftime to np.datetime64
    xds = xds.rename({"time": "cftime"})
    xds["datetime"] = _cftime2datetime(xds["cftime"])
    xds = xds.swap_dims({"cftime": "datetime"})
    xds = xds.drop_vars("cftime")

    # set time as time relative to initial conditions
    xds["time"] = xr.DataArray(
        [pd.Timedelta(hours=relative_time)],
        coords=xds["datetime"].coords,
        attrs={"description": "time relative to last initial condition"},
    )
    xds = xds.swap_dims({"datetime": "time"})
    xds = xds.set_coords("datetime")

    # rename these before passing to GraphCast code
    rename = {
        "pfull": "level",
        "grid_yt": "lat",
        "grid_xt": "lon",
    }
    for k, v in rename.items():
        if k in xds:
            xds = xds.rename({k: v})

    # add file prefix to these variables in order to store result in separate netcdf files
    for key in xds.data_vars:
        xds[key].attrs["file_prefix"] = prefix.replace("sanl", "sfg")

    # finally, load this into memory
    xds = xds.load();

    # if any transform variables are here, map to transformed
    for key, mapping in _input_transforms.items():
        if key in xds:
            logging.info(f"Mapping {key} -> {mapping.__name__}({key})")
            with xr.set_options(keep_attrs=True):
                xds[key] = mapping(xds[key])
    return xds


def open_dataset(
    cycle: str,
    member: int,
    task_config: graphcast.TaskConfig,
    fhrs: Optional[tuple[int]] = (3, 6),
    experiment_directory: Optional[str] = "./",
) -> xr.Dataset:
    """
    Args:
        cycle (str): DA cycle in YYYYMMDDHH
        member (int): ensemble member number
        task_config (graphcast.TaskConfig): for GraphCast, defines variables, pressure levels etc used
        fhrs (tuple[int], optional): forecast hours from the previous forecast to use
        experiment_directory (str, optional): where to look for the data

    Returns:
        xds (xr.Dataset): with sanl/bfg combined, taking surface pressure from sanl since that's updated by the DA
    """

    attrs = {}
    dslist1 = []
    static_vars = {}
    for fhr in fhrs:
        dslist2 = []
        for prefix in ["sanl", "bfg"]:
            tmp = xr.open_dataset(
                f"{experiment_directory}/{cycle}/{prefix}_{cycle}_fhr{fhr:02d}_mem{member:03d}",
            )

            if prefix == "bfg" and "pressfc" in tmp:
                tmp = tmp.drop_vars("pressfc")
            if prefix not in attrs:
                attrs[prefix] = tmp.attrs.copy()


            tmp = preprocess(
                tmp,
                cycle=cycle,
                relative_time=fhr-fhrs[-1],
                task_config=task_config,
                prefix=prefix,
            )

            # modify any of the original attrs if they still exist after preprocess
            # namely, ak/bk
            for key, val in tmp.attrs.items():
                attrs[prefix][key] = val

            for key in tmp.data_vars:
                if "time" not in tmp[key].dims:
                    if key not in static_vars:
                        static_vars[key] = tmp[key]
                    tmp = tmp.drop_vars(key)

            dslist2.append(tmp)
        dslist1.append(xr.merge(dslist2))

    xds = xr.concat(dslist1, dim="time")
    for key, xda in static_vars.items():
        xds[key] = xda

    xds.attrs["file_attrs"] = attrs.copy()
    return xds


def load_normalization(task_config):
    """Assumes we want normalization from the 1/4 degree subsampled to ~1 degree data"""

    normalization = dict()
    for norm_component in ["mean", "std", "stddiff"]:
        this_norm = xr.open_dataset(_norm_paths[norm_component])
        myvars = list(x for x in get_all_variables(task_config) if x in this_norm)
        # keep attributes in order to distinguish static from time varying components
        with xr.set_options(keep_attrs=True):

            if _input_transforms is not None:
                for key, transform_function in _input_transforms.items():

                    # make sure e.g. log_spfh is in the dataset
                    transformed_key = f"{transform_function.__name__}_{key}" # e.g. log_spfh
                    assert transformed_key in this_norm, \
                        f"load_normalization: couldn't find {transformed_key} in {component} normalization dataset"
                    # there's a chance the original, e.g. spfh, is not in the dataset
                    # if it is, replace it with e.g. log_spfh
                    if key in myvars:
                        idx = myvars.index(key)
                        myvars[idx] = transformed_key
            this_norm = this_norm[myvars]
            if _input_transforms is not None:
                for key, transform_function in _input_transforms.items():
                    transformed_key = f"{transform_function.__name__}_{key}" # e.g. log_spfh
                    idx = myvars.index(transformed_key)
                    myvars[idx] = key

                    # necessary for graphcast.dataset to stacked operations
                    this_norm = this_norm.rename({transformed_key: key})

        this_norm = this_norm.sel(pfull=list(task_config.pressure_levels), method="nearest")
        this_norm = this_norm.rename({"pfull": "level"})
        normalization[norm_component] = this_norm.load()

    return normalization


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

    if not hasattr(predict, "gc_forecast"):
        logging.info("JIT compiling the predictor")
        predict.gc_forecast = drop_state( with_params( jax.jit( with_configs( run_forward.apply ) ) ) )

    predictions = rollout.chunked_prediction(
        predict.gc_forecast,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )
    return predictions

def main(argv):

    # Inputs and some checking
    previous_cycle = PREVIOUS_CYCLE.value
    member = ENSEMBLE_MEMBER.value
    experiment_directory = EXPERIMENT_DIRECTORY.value
    fhr_ics = tuple(int(x) for x in FHR_ICS.value)
    model_weights = MODEL_WEIGHTS.value

    assert len(previous_cycle) == 10, "previous_cycle must be a 10 character long string in format YYYYMMDDHH"
    expected = _timestamp2cycle(_cycle2timestamp(previous_cycle))
    assert previous_cycle == expected, "previous_cycle must be a 10 character long string in format YYYYMMDDHH"

    next_cycle = _timestamp2cycle( _cycle2timestamp(previous_cycle) + pd.Timedelta(hours=6) )

    setup_log()
    logging.info(f"Reading weights from: {model_weights}")
    params, model_config, task_config = load_checkpoint(model_weights)

    logging.info("Reading normalization")
    normalization = load_normalization(task_config)

    # Note that all of the code below, including postprocess, assumes
    # we are working one ensemble member at a time

    logging.info(f"Reading initial conditions from cycle: {previous_cycle}, fhrs: {fhr_ics}, member: {member}")
    xds = open_dataset(
        cycle=previous_cycle,
        member=member,
        task_config=task_config,
        fhrs=fhr_ics,
        experiment_directory=experiment_directory,
    )
    prefix_mapper = {key: xds[key].attrs["file_prefix"] for key in xds.data_vars}

    logging.info("Preparing initial conditions for GraphCast code")
    inputs, targets_template, forcings = get_model_inputs(
        xds,
        member=member,
        lead_times=["3h", "6h"],
        task_config=task_config,
    )

    logging.info(f"Starting GraphUFS Prediction for cycle: {next_cycle}, member: {member}")
    prediction = predict(
        params=params,
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
        model_config=model_config,
        task_config=task_config,
        normalization=normalization,
    )

    logging.info("Postprocessing Prediction for DA")
    prediction = postprocess(prediction, last_ic_datetime=xds["datetime"].values[1])
    prediction.attrs = xds.attrs.copy()

    # take the forcing and stick it in the prediction dataset as well
    postforcing = postprocess(forcings, last_ic_datetime=xds["datetime"].values[1])
    for key in postforcing.data_vars:
        if key in xds.data_vars:
            logging.info(f"Storing forcing[{key}] -> prediction dataset")
            prediction[key] = postforcing[key]
            prediction[key].attrs = xds[key].attrs.copy()

    # stick in the static variables
    poststatic = postprocess(inputs[[x for x in inputs.data_vars if "_static" in x]], last_ic_datetime=xds["datetime"].values[1])
    for key in poststatic.data_vars:
        if key in xds.data_vars:
            logging.info(f"Storing inputs[{key}] -> prediction dataset")
            prediction[key] = poststatic[key].expand_dims("time")
            prediction[key].attrs = xds[key].attrs.copy()


    store_result(
        prediction,
        prefix_mapper=prefix_mapper,
        cycle=next_cycle,
        member=member,
        experiment_directory=experiment_directory,
    )

if __name__ == "__main__":
    app.run(main)
