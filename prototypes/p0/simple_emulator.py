from jax import tree_util

from graphufs import ReplayEmulator

class P0Emulator(ReplayEmulator):

    data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/mean_by_level.p0.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/stddev_by_level.p0.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/diffs_stddev_by_level.p0.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

    local_store_path = "./zarr-stores"

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "tmp",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "tmp",
    )
    forcing_variables = (
        "land",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    all_variables = tuple() # this is created in __init__
    pressure_levels = (
        100,
        500,
        1000,
    )

    # time related
    delta_t = "6h"              # the model time step
    input_duration = "12h"      # time covered by initial condition(s) + 6h (necessary for GraphCast code)
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "1994-01-01T00",        # start
        "1994-12-31T18"         # stop
    )
    testing_dates = (           # bounds of testing data (inclusive)
        "1995-01-01T00",        # start
        "1995-12-31T18"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "1996-01-01T00",        # start
        "1996-12-31T18"         # stop
    )

    # training protocol
    batch_size = 16
    num_epochs = 1

    # model config options
    resolution = 1.0
    mesh_size = 2
    latent_size = 256
    gnn_msg_steps = 4
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    mesh2grid_edge_normalization_factor = 0.6180338738074472

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data chunking options
    chunks_per_epoch = 1
    steps_per_chunk = None
    checkpoint_chunks = 1

tree_util.register_pytree_node(
    P0Emulator,
    P0Emulator._tree_flatten,
    P0Emulator._tree_unflatten
)
