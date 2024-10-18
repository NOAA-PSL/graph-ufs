from jax import tree_util

from graphufs import ReplayCoupledEmulator

class CP0Emulator(ReplayCoupledEmulator):

    atm_data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr"
    atm_norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/diffs_stddev_by_level.zarr",
    }

    ocn_data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/mom6.zarr"
    ocn_norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/mom6.statistics.1993-1997/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/mom6.statistics.1993-1997/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/mom6.statistics.1993-1997/diffs_stddev_by_level.zarr",
    }
    
    ice_data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr"
    ice_norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/diffs_stddev_by_level.zarr",
    }
    land_data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr"
    land_norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/fv3.statistics.1993-1997/diffs_stddev_by_level.zarr",
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
        "landsea_mask",
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
    atm_pressure_levels = (
        200,
        250,
        300,
        350,
        400,
        450,
        500,
        550,
        600,
        650,
        700,
        750,
        800,
        850,
        900,
        950,
        1000,
    )
    ocn_vert_levels = (
        0.5, 
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

    # time related
    delta_t = "6h"              # the model time step, assumed to be the same for both atm and ocn for now.
                                # A more complicated case of diffential time steps and grid size will be 
                                # developed in the future
    input_duration = "12h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    #target_lead_time = [f"{n}h" for n in range(6, 6*4*1+1, 6)]
    training_dates = (          # bounds of training data (inclusive)
        "1994-01-01T00",        # start
        "2019-12-31T18"         # stop
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "2020-01-01T00",        # start
        "2020-02-01T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "2022-01-01T00",        # start
        "2023-10-13T18"         # stop
    )

    # training protocol
    batch_size = 16
    num_batch_splits = 1
    num_epochs = 64

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    #mesh2grid_edge_normalization_factor = 0.6180338738074472

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
    #no_load_chunks = False
    store_loss = True
    use_preprocessed = True

    # others
    num_gpus = 1
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = False
    dask_threads = None

tree_util.register_pytree_node(
    CP0Emulator,
    CP0Emulator._tree_flatten,
    CP0Emulator._tree_unflatten
)
