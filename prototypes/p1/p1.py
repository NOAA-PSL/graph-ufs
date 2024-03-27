from jax import tree_util

from graphufs import ReplayEmulator

class P1Emulator(ReplayEmulator):

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
        "delz", # height thickness (as a replacement for geopotential)
        "hgtsfc", # geopotential at surface
        "land", # land-sea mask
        "tmp2m", # 2m temperature
        "pressfc", # mean sea level pressure
        "ugrd10m", # 10m u component of wind
        "vgrd10m", # 10m v component of wind
        "tprecp",  # total precipitation 3hr, we need 6 hrs though
        "dswrf_avetoa", # toa incident solar radiation
        "tmp", # temperature
        "ugrd", # u component of wind
        "vgrd", # v component of wind
        "dzdt", # vertical velocity
        "spfh", # specific humidity
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "delz"
        "tmp2m",
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "tprecp",
        "tmp",
        "ugrd",
        "vgrd",
        "dzdt",
        "spfh",
    )
    forcing_variables = (
        "dswrf_avetoa", # toa incident solar radiation
        "land", # land-sea-ice mask
        "hgtsfc", # geopotential at the surface
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    
    all_variables = tuple() # this is created in __init__

    pressure_levels = (
        50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

    # time related
    delta_t = "6h"              # the model time step
    input_duration = "6h 1s"    # time covered by initial condition(s), note the 1s is necessary for GraphCast code
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets
    training_dates = (          # bounds of training data (inclusive)
        "1993-12-31T18",        # start
        "1994-12-31T18"         # stop, includes all of 1994
    )

    # training protocol
    batch_size = 16

    # model config options
    resolution = 1.0
    mesh_size = 4
    latent_size = 32
    gnn_msg_steps = 4
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6
    mesh2grid_edge_normalization_factor = None # 0.6180338738074472

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

tree_util.register_pytree_node(
    P0Emulator,
    P0Emulator._tree_flatten,
    P0Emulator._tree_unflatten
)
