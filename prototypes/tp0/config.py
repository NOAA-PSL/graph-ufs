import xarray as xr
from jax import tree_util
import numpy as np

from graphufs import FVEmulator

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

class TP0Emulator(FVEmulator):

    data_url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "./fv3.fvstatistics.1993-01/mean_by_level.zarr",
        "std": "./fv3.fvstatistics.1993-01/stddev_by_level.zarr",
        "stddiff": "./fv3.fvstatistics.1993-01/diffs_stddev_by_level.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

    local_store_path = "./local-output"
    cache_data = True

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        "pressfc",
        "tmp2m",
        "spfh2m",
        "tmp",
        "spfh",
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
        "tmp",
        "spfh",
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
        "1994-03-31T18"         # stop
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "1995-01-01T00",        # start
        "1995-01-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "1996-01-01T00",        # start
        "1996-01-31T18"         # stop
    )

    # training protocol
    batch_size = 16
    num_batch_splits = 1
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
    loss_weights_per_variable = {
        "pressfc"   : 1.0,
        "tmp2m"     : 1.0,
        "spfh2m"    : 1.0,
        "tmp"       : 1.0,
        "spfh"      : 1.0,
    }
    input_transforms = {
        "spfh": log,
        "spfh2m": log,
    }
    output_transforms = {
        "spfh": exp,
        "spfh2m": exp,
    }

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data chunking options
    chunks_per_epoch = 1
    steps_per_chunk = None
    checkpoint_chunks = 1
    max_queue_size = 1
    num_workers = 1
    load_chunk = True
    store_loss = True
    use_preprocessed = True

    # others
    num_gpus = 1
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = False
    dask_threads = 8

tree_util.register_pytree_node(
    TP0Emulator,
    TP0Emulator._tree_flatten,
    TP0Emulator._tree_unflatten
)
