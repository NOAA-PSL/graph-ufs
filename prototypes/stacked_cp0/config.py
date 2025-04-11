import os
import xarray as xr
from jax import tree_util
import numpy as np

from graphufs import FVCoupledEmulator


stackedcp0_path = os.path.dirname(os.path.realpath(__file__))

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

class StackedCP0Emulator(FVCoupledEmulator):
    data_url = {}
    norm_urls = {}

    norm_url = {
        "mean": f"{stackedcp0_path}/fvstatistics.1994/mean_by_level.zarr",
        "std": f"{stackedcp0_path}/fvstatistics.1994/stddev_by_level.zarr",
        "stddiff": f"{stackedcp0_path}/fvstatistics.1994/diffs_stddev_by_level.zarr",
    }
    data_url["atm"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls["atm"] = norm_url

    data_url["ocn"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.zarr"
    norm_urls["ocn"] = norm_url

    data_url["ice"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls["ice"] = norm_url
    
    data_url["land"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls["land"] = norm_url

    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

    local_store_path = f"{stackedcp0_path}/local-output"
    cache_data = True

    # these could be moved to a yaml file later
    # task config options
    # inputs
    atm_input_variables = (
        # Surface Variables
        #"pressfc",
        "ugrd10m",
        "vgrd10m",
        #"tmp2m",
        #"spfh2m",
        # 3D Variables
        #"tmp",
        #"spfh",
        #"ugrd",
        #"vgrd",
        #"dzdt",
        "land_static",
        # Forcing Variables at input time
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    ocn_input_variables = (
        # Surface Variables
        "SSH",
        "LW",
        "SW",
        # 3D Variables
        "so",
        "temp",
        "uo",
        "vo",
        "landsea_mask",
    )
    ice_input_variables = (
        #"icec",
        #"icetk",
    )
    land_input_variables = (
        #"soilm",
        #"soilt1",
        #"tmpsfc",
        #"land_static",
    )

    # targets
    atm_target_variables = (
        # Surface Variables
        #"pressfc",
        #"ugrd10m",
        #"vgrd10m",
        #"tmp2m",
        #"spfh2m",
        # 3D Variables
        #"tmp",
        #"spfh",
        #"ugrd",
        #"vgrd",
        #"dzdt",
    )
    ocn_target_variables = (
        # Surface Variables
        "SSH",
        "LW",
        "SW",
        # 3D Variables
        "so",
        "temp",
        "uo",
        "vo",
    )
    ice_target_variables = (
        #"icec",
        #"icetk",
    )
    land_target_variables = (
        #"soilm",
        #"soilt1",
        #"tmpsfc",
    )

    # forcing
    atm_forcing_variables = (
        "ugrd10m",
        "vgrd10m",
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    ocn_forcing_variables = ()
    ice_forcing_variables = ()
    land_forcing_variables = ()

    all_variables = tuple() # this is created in __init__
    interfaces = {}
    interfaces["atm"] = () #tuple(x for x in range(200, 1001, 50)) #(400, 600, 800, 1000)
    interfaces["ocn"] = (#(0, 1) 
        0,
        1,
        5,
        10,
        20,
        40,
        70,
        120,
        200,
        350,
        500,
    )
    interfaces["ice"] = ()
    interfaces["land"] = ()
    
    # transforms related
    input_transforms = {
    #    "spfh": log,
    #    "spfh2m": log,
    }
    output_transforms = {
    #    "spfh": exp,
    #    "spfh2m": exp,
    }

    # time related
    delta_t_model = "6h"        # the model time step
    delta_t_data = "6h"         # time steps in the data
    input_duration = "12h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "1993-12-31T18",        # start
        "1994-03-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "2021-12-31T18",        # start
        "2022-01-31T18",        # stop
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "2020-01-01T18",        # start
        "2020-01-31T18"         # stop
    )

    # training protocol
    batch_size = 16
    num_batch_splits = 1
    num_epochs = 5

    # model config options
    resolution = 1.0
    mesh_size = 2
    latent_size = 256
    gnn_msg_steps = 4
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_channel = True
    weight_loss_per_latitude = True
    weight_loss_per_level = False
    loss_weights_per_variable = dict()
    
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
    use_half_precision = False # this must be set to false for coupled stacked graphcast with masked loss to work

class StackedCP0Tester(StackedCP0Emulator):
    #target_lead_time = ["6h", "12h", "18h", "24h", "30h", "36h", "42h", "48h"]
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]

tree_util.register_pytree_node(
    StackedCP0Emulator,
    StackedCP0Emulator._tree_flatten,
    StackedCP0Emulator._tree_unflatten
)

tree_util.register_pytree_node(
    StackedCP0Tester,
    StackedCP0Tester._tree_flatten,
    StackedCP0Tester._tree_unflatten
)
