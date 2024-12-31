import xarray as xr
from jax import tree_util
import numpy as np

from graphufs import FVEmulator

def log(xda):
    cond = xda > 0
    return xr.where(
        cond,
        np.log(xda.where(cond)),
        0.,
    )

def exp(xda):
    return np.exp(xda)

class P2TrainingEmulator(FVEmulator):

    # paths
    data_url = "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.zarr"
    norm_urls = {
        "mean": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/mean_by_level.zarr",
        "std": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/stddev_by_level.zarr",
        "stddiff": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/diffs_stddev_by_level.zarr",
    }
    local_store_path = "/p2-lustre/p2"

    # these could be moved to a yaml file later
    # task config options
    input_variables = (
        # 3D Variables
        "ugrd",
        "vgrd",
        "dzdt",
        "tmp",
        "spfh",
        # Surface Variables
        "ugrd10m",
        "vgrd10m",
        "tmp2m",
        "spfh2m",
        "pressfc",
        # Forcing Variables at Input Time
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
        # Static Variables
        "land_static",
        "hgtsfc_static",
    )
    target_variables = (
        # 3D Variables
        "ugrd",
        "vgrd",
        "dzdt",
        "tmp",
        "spfh",
        # Surface Variables
        "ugrd10m",
        "vgrd10m",
        "tmp2m",
        "spfh2m",
        "pressfc",
    )
    forcing_variables = (
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )

    # vertical grid
    interfaces = tuple(x for x in range(200, 1001, 50))

    # time related
    delta_t = "3h"
    input_duration = "6h"
    target_lead_time = "3h"
    training_dates = (
        "1993-12-31T18",
        "2019-12-31T18",
    )
    validation_dates = (
        "2022-01-01T00",
        "2023-10-13T03",
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
    weight_loss_per_level = False
    loss_weights_per_variable = dict() # weight them all equally
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

    # data loading options
    max_queue_size = 1
    num_workers = 1
    dask_threads = 32

    # hardware
    num_gpus = 1
    log_only_rank0 = False
    use_jax_distributed = False
    use_xla_flags = True

class P2PreprocessedEmulator(P2TrainingEmulator):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None


class P2EvaluationEmulator(P2TrainingEmulator):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(3, 3*8*10+1, 3)]
    sample_stride = 9
    evaluation_checkpoint_id = 64
    num_gpus = 1



tree_util.register_pytree_node(
    P2TrainingEmulator,
    P2TrainingEmulator._tree_flatten,
    P2TrainingEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    P2EvaluationEmulator,
    P2EvaluationEmulator._tree_flatten,
    P2EvaluationEmulator._tree_unflatten
)
