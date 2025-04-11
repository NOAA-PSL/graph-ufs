from jax import tree_util
import numpy as np
import xarray as xr
from graphufs import FVCoupledEmulator

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

_scratch = "/pscratch/sd/n/nagarwal"

class BaseOcnTrainer(FVCoupledEmulator):

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
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.ice.statistics.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.ice.statistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.ice.statistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    data_url["land"] = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls["land"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.land.statistics.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.land.statistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.land.statistics.1993-2019/diffs_stddev_by_level.zarr",
    }

    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
    no_cache_data = False

    # these could be moved to a yaml file later
    # task config options
    # make sure that atm_inputs and ocn_inputs are mutually exclusive sets
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
        "landsea_mask", # this is diagnosed inside
    )
    ice_input_variables = (
        #"icec",
        #"icetk",
    )
    land_input_variables = (
        #"soilm",
        #"soilt1",
        #"tmpsfc",
    )
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
    interfaces["atm"] = ()
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

    # transforms related
    input_transforms = {}
    output_transforms = {}

    # time related
    delta_t_model = "6h"        # the model time step
    delta_t_data = "6h"         # time steps in the data
    input_duration = "12h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    #target_lead_time = [f"{n}h" for n in range(6, 6*4*1+1, 6)]
    training_dates = (          # bounds of training data (inclusive)
        "1993-12-31T18",        # start
        "2019-12-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)  
        "2022-01-01T00",        # start
        "2023-10-13T00",        # stop 
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "2020-01-01T00",        # start
        "2021-12-31T18"         # stop
    )

    # training protocol
    batch_size = 16
    num_batch_splits = 1
    num_epochs = 80
    use_half_precision = False

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    #mesh2grid_edge_normalization_factor = 0.6180338738074472

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_channel = True
    weight_loss_per_latitude = True
    weight_loss_per_level = False # weight both ocean and atm vertical levels
    loss_weights_per_variable = dict() # weight all of them equally
    input_transforms = {}
    output_transforms = {}

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data loading options
    max_queue_size = 1
    num_workers = 1


tree_util.register_pytree_node(
    BaseOcnTrainer,
    BaseOcnTrainer._tree_flatten,
    BaseOcnTrainer._tree_unflatten
)

