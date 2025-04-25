from jax import tree_util

from graphufs import FVEmulator

class P0Emulator(FVEmulator):

    data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/diffs_stddev_by_level.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

    local_store_path = "./zarr-stores"
    no_cache_data = False

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        "pressfc",
        #"ugrd10m",
        #"vgrd10m",
        "tmp",
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "pressfc",
        #"ugrd10m",
        #"vgrd10m",
        "tmp",
    )
    forcing_variables = (
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    all_variables = tuple() # this is created in __init__
    interfaces = tuple(x for x in range(200, 1001, 50))

    # time related
    delta_t = "3h"
    input_duration = "6h"
    target_lead_time = "3h"
    #delta_t_data = "3h"         # time step of the data
    #delta_t_model = "6h"        # the model time step
    #input_duration = "12h"      # time covered by initial condition(s), note the 1s is necessary for GraphCast code
    #target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "1994-01-01T00",        # start
        "1994-01-31T18"         # stop
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "1995-01-01T00",        # start
        "1995-01-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "2020-01-01T00",        # start
        "2020-01-31T18"         # stop
    )

    # training protocol
    batch_size = 16
    num_epochs = 1

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_channel = True
    weight_loss_per_latitude = True
    weight_loss_per_level = False
    loss_weights_per_variable = {
    #    "tmp2m"         : 1.0,
    #    "ugrd10m"       : 0.1,
    #    "vgrd10m"       : 0.1,
    #    "pressfc"       : 0.1,
    #    "prateb_ave"    : 0.1,
    }

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    #mesh2grid_edge_normalization_factor = 0.6180338738074472

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data chunking options
    chunks_per_epoch = 1
    steps_per_chunk = None
    checkpoint_chunks = 1

    # others
    num_gpus = 0
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = False

tree_util.register_pytree_node(
    P0Emulator,
    P0Emulator._tree_flatten,
    P0Emulator._tree_unflatten
)
