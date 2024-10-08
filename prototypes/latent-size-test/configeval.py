from jax import tree_util

from graphufs import ReplayEmulator

class LatentTestEmulator(ReplayEmulator):

    data_url = "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.statistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    local_store_path = "/testlfs/latent-size-test"
    cache_data = True

    # task config options
    input_variables = (
        "geopotential",
        "pressfc",
        "hgtsfc_static",
        "land_static",
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    target_variables = (
        "geopotential",
        "pressfc",
    )
    forcing_variables = (
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )

    all_variables = tuple() # this is created in __init__

    pressure_levels = (
        0.012781458906829357,
        0.048782818019390106,
        0.15953919291496277,
        0.45232152938842773,
        1.124390959739685,
        2.478975772857666,
        4.903191566467285,
        8.800692558288574,
        14.499824523925781,
        22.181779861450195,
        31.871253967285156,
        43.50650405883789,
        57.074893951416016,
        72.7860336303711,
        91.08661651611328,
        112.31695556640625,
        136.75987243652344,
        164.65609741210938,
        196.17715454101562,
        231.39479064941406,
        270.25048828125,
        312.5292053222656,
        357.8437805175781,
        405.63275146484375,
        455.1800537109375,
        505.6520690917969,
        556.1478271484375,
        605.7676391601562,
        653.6714477539062,
        699.1375122070312,
        741.5904541015625,
        780.6387329101562,
        816.0591430664062,
        847.783203125,
        875.869873046875,
        900.4920043945312,
        921.8844604492188,
        940.3500366210938,
        956.1390380859375,
        969.6144409179688,
        981.0247192382812,
        990.6548461914062,
        998.7807006835938,
    )

    # time related
    delta_t = "3h"
    input_duration = "6h"
    target_lead_time = [f"{n}h" for n in range(3, 3*8*10+1, 3)]
    training_dates = (
        "1993-12-31T18",
        "2019-12-31T21"
    )
    validation_dates = (
        "2022-01-01T00",
        "2022-12-31T21"
    )
    testing_dates = (
        "2020-01-01T00",
        "2020-02-01T03",
    )

    # training protocol
    batch_size = 16 # 32
    num_epochs = 64 # 127
    chunks_per_epoch = 48
    steps_per_chunk = None
    checkpoint_chunks = 1
    max_queue_size = 1
    num_workers = 1
    load_chunk = True
    store_loss = True
    use_preprocessed = False

    weight_loss_per_latitude = True
    weight_loss_per_level = True
    loss_weights_per_variable = {
        "pressfc"       : 1.0,
    }

    # evaluation
    sample_stride = 9 # sample every 27h, results in 569 ICs, ~1.6 TiB of data
    evaluation_checkpoint_id = 64

    # multi GPU and xla options
    num_gpus = 1
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = True
    dask_threads = 32

    # model config options
    resolution = 1.0
    mesh_size = 5
    latent_size = 256
    gnn_msg_steps = 16
    hidden_layers = 1
    radius_query_fraction_edge_length = 0.6

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = 0
    init_rng_seed = 0
    training_batch_rng_seed = 100

tree_util.register_pytree_node(
    LatentTestEmulator,
    LatentTestEmulator._tree_flatten,
    LatentTestEmulator._tree_unflatten
)
