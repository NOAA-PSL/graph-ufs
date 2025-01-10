import os
import xarray as xr
from jax import tree_util, numpy as jnp
import numpy as np

from graphufs import FVEmulator


tp0_path = os.path.dirname(os.path.realpath(__file__))

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def jlog(array):
    cond = array > 0
    masked = jnp.where(cond, array, 1.)
    return jnp.log(masked)

def exp(xda):
    return np.exp(xda)

def jexp(array):
    cond = array != 0
    return jnp.where(
        cond,
        jnp.exp(array),
        0.
    )

class BaseTP0Emulator(FVEmulator):

    data_url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": f"{tp0_path}/fv3.fvstatistics.1994/mean_by_level.zarr",
        "std": f"{tp0_path}/fv3.fvstatistics.1994/stddev_by_level.zarr",
        "stddiff": f"{tp0_path}/fv3.fvstatistics.1994/diffs_stddev_by_level.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

    local_store_path = None

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        "pressfc",
        "tmp2m",
        "spfh2m",
        "ugrd10m",
        "vgrd10m",
        "tmp",
        "spfh",
        "ugrd",
        "vgrd",
        "land_static",
        "hgtsfc_static",
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "pressfc",
        "tmp2m",
        "spfh2m",
        "ugrd10m",
        "vgrd10m",
        "tmp",
        "spfh",
        "ugrd",
        "vgrd",
    )
    forcing_variables = (
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    interfaces = (400, 600, 800, 1000)

    # time related
    delta_t = "3h"              # the model time step
    input_duration = "6h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "3h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "1994-01-01T00",        # start
        "1994-12-31T21"         # stop
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "1995-01-01T00",        # start
        "1995-01-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "1996-01-01T00",        # start
        "1996-01-31T21"         # stop
    )

    # training protocol
    batch_size = 16
    num_epochs = 50

    # model config options
    resolution = 1.0
    mesh_size = 2
    latent_size = 256
    gnn_msg_steps = 4
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_latitude = True
    weight_loss_per_level = False
    loss_weights_per_variable = dict()
    input_transforms = {
        "spfh": log,
        "spfh2m": log,
    }
    output_transforms = {
        "spfh": exp,
        "spfh2m": exp,
    }
    compilable_input_transforms = {
        "spfh": jlog,
        "spfh2m": jlog,
    }
    compilable_output_transforms = {
        "spfh": jexp,
        "spfh2m": jexp,
    }

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data chunking options
    max_queue_size = 1
    num_workers = 1
    dask_threads = 8
    num_gpus = 1

tree_util.register_pytree_node(
    BaseTP0Emulator,
    BaseTP0Emulator._tree_flatten,
    BaseTP0Emulator._tree_unflatten
)
