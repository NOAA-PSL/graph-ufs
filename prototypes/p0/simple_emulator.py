from jax import tree_util

from graphufs import ReplayEmulator

class P0Emulator(ReplayEmulator):

    data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/mean_by_level.p0.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/stddev_by_level.p0.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/normalization/diffs_stddev_by_level.p0.zarr",
    }
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
    input_duration = "12h"

    # model config options
    resolution = 1.0
    mesh_size = 2
    latent_size = 32
    gnn_msg_steps = 4
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    mesh2grid_edge_normalization_factor = 0.6180338738074472

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0

tree_util.register_pytree_node(
    P0Emulator,
    P0Emulator._tree_flatten,
    P0Emulator._tree_unflatten
)
