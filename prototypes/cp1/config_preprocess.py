from jax import tree_util
import numpy as np
import xarray as xr
from graphufs import ReplayCoupledEmulator

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

class CP1TrainingEmulator(ReplayCoupledEmulator):
    
    # paths
    data_url = {}
    norm_urls = {}

    data_url["atm"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls["atm"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/diffs_stddev_by_level.zarr",
    }
    data_url["ocn"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.zarr"
    norm_urls["ocn"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/diffs_stddev_by_level.zarr",
    }
    data_url["ice"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls["ice"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/diffs_stddev_by_level.zarr",
    }
    data_url["land"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls["land"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/diffs_stddev_by_level.zarr",
    }
    
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

    local_store_path = "./zarr-stores"
    no_cache_data = False

    # these could be moved to a yaml file later
    # task config options
    # make sure that atm_inputs and ocn_inputs are mutually exclusive sets
    atm_input_variables = (
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "tmp2m",
        "spfh2m",
        "tmp",
        "spfh",
        "ugrd",
        "vgrd",
        "dzdt",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    ocn_input_variables = (
        "SSH",
        "LW",
        "SW",
        "so",
        "temp",
        "uo",
        "vo",
        #"landsea_mask",
    )
    ice_input_variables = (
        "icec",
        "icetk",
    )
    land_input_variables = (
        "soilm",
        "soilt1",
        "tmpsfc",
    )
    atm_target_variables = (
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "tmp2m",
        "spfh2m",
        "tmp",
        "spfh",
        "ugrd",
        "vgrd",
        "dzdt",
    )
    ocn_target_variables = (
        "SSH",
        "LW",
        "SW",
        "so",
        "temp",
        "uo",
        "vo",
    )
    ice_target_variables = (
        "icec",
        "icetk",
    )
    land_target_variables = (
        "soilm",
        "soilt1",
        "tmpsfc",
    )
    atm_forcing_variables = (
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
    interfaces["atm"] = tuple(x for x in range(200, 1001, 100))
    interfaces["ocn"] = (
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

    # time related
    delta_t = "6h"              # the model time step, assumed to be the same for both atm and ocn for now.
                                # A more complicated case of diffential time steps and grid size will be 
                                # developed in the future
    input_duration = "12h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    #target_lead_time = [f"{n}h" for n in range(6, 6*4*1+1, 6)]
    training_dates = (          # bounds of training data (inclusive)
        "1993-12-31T18",        # start
        "2019-12-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "2022-01-01T00",        # start
        "2023-10-13T18"         # stop
    )
    testing_dates = (
        "2020-01-01T00",
        "2021-12-31T18",
    )

    # training protocol
    batch_size = 16
    num_epochs = 64

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_latitude = True
    weight_loss_per_level = True
    atm_loss_weights_per_variable = {
        "tmp"           : 1.0,
        "ugrd10m"       : 0.1,
        "vgrd10m"       : 0.1,
        "pressfc"       : 0.1,
        "prateb_ave"    : 0.1,
    }
    ocn_loss_weights_per_variable = {
        "SSH"           : 0.1,
        "so"            : 1.0,
        "temp"          : 1.0,
    }
    ice_loss_weights_per_variable = {
        "icec"          : 1.0,
        "icetk"         : 0.1,
    }
    land_loss_weights_per_variable = {
        "soilm"         : 0.1,
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
    max_queue_size = 1
    num_workers = 1

    # hardware
    num_gpus = 1
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = True
    dask_threads = 32

class CP1PreprocessedEmulator(CP1TrainingEmulator):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class CP1EvaluationEmulator(CP1TrainingEmulator):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    sample_stride = 12
    evaluation_checkpoint_id = 64
    num_gpus = 1



tree_util.register_pytree_node(
    CP1EvaluationEmulator,
    CP1EvaluationEmulator._tree_flatten,
    CP1EvaluationEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    CP1PreprocessedEmulator,
    CP1PreprocessedEmulator._tree_flatten,
    CP1PreprocessedEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    CP1TrainingEmulator,
    CP1TrainingEmulator._tree_flatten,
    CP1TrainingEmulator._tree_unflatten
)

