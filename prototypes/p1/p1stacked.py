from jax import tree_util

from graphufs import ReplayEmulator

class P1Emulator(ReplayEmulator):

    data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    local_store_path = "/lustre/stacked-p1-data"
    cache_data = True

    # task config options
    input_variables = (
        "delz",
        "hgtsfc",
        "land",
        "tmp2m",
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "prateb_ave",
        "dswrf_avetoa",
        "tmp",
        "ugrd",
        "vgrd",
        "dzdt",
        "spfh",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "delz",
        "tmp2m",
        "pressfc",
        "ugrd10m",
        "vgrd10m",
        "prateb_ave",
        "tmp",
        "ugrd",
        "vgrd",
        "dzdt",
        "spfh",
    )
    forcing_variables = (
        "dswrf_avetoa",
        "land",
        "hgtsfc",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )

    all_variables = tuple() # this is created in __init__

    pressure_levels = (
        50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000,
    )

    # time related
    delta_t = "3h"
    input_duration = "6h"
    #target_lead_time = "3h"
    target_lead_time = [f"{n}h" for n in range(3, 3*8*10+1, 3)]
    training_dates = (
        "1993-12-31T18",
        "2019-12-31T21"
    )
    validation_dates = (
        "2022-01-01T00",
        "2023-10-13T03"
    )
    testing_dates = (
        "2020-01-01T00",
        "2020-02-01T03",
    )

    # training protocol
    batch_size = 16
    num_epochs = 50
    chunks_per_epoch = 48
    steps_per_chunk = None
    checkpoint_chunks = 1
    max_queue_size = 1
    num_workers = 1
    load_chunk = True
    store_loss = True
    use_preprocessed = False

    # evaluation
    sample_stride = 9 # sample every 27h, results in 569 ICs, ~1.6 TiB of data
    evaluation_checkpoint_id = 50

    # multi GPU and xla options
    num_gpus = 4
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = False
    dask_threads = 16

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 512
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

tree_util.register_pytree_node(
    P1Emulator,
    P1Emulator._tree_flatten,
    P1Emulator._tree_unflatten
)
