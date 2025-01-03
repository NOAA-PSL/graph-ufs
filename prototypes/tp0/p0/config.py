from jax import tree_util

from graphufs import ReplayEmulator

from prototypes.tp0.config import tp0_path

class P0Emulator(ReplayEmulator):

    norm_urls = {
        "mean": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/mean_by_level.zarr",
        "std": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"

    local_store_path = "{tp0_path}/p0"

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
    pressure_levels = (498, 698, 899)

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

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

    # data chunking options
    max_queue_size = 1
    num_workers = 1
    dask_threads = 8


class P0Tester(P0Emulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

tree_util.register_pytree_node(
    P0Emulator,
    P0Emulator._tree_flatten,
    P0Emulator._tree_unflatten
)

tree_util.register_pytree_node(
    P0Tester,
    P0Tester._tree_flatten,
    P0Tester._tree_unflatten
)
