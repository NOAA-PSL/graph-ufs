from jax import tree_util
from prototypes.p2p.config import BaseP2PTrainer, _scratch

class P2PTrainer(BaseP2PTrainer):

    norm_urls = {
        "mean": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.fvstatistics.p2p/mean_by_level.zarr",
        "std": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.fvstatistics.p2p/stddev_by_level.zarr",
        "stddiff": "gs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/03h-freq/zarr/fv3.fvstatistics.p2p/diffs_stddev_by_level.zarr",
    }
    local_store_path = f"{_scratch}/p2p/nvnc"

    # Same as base, just remove clock variables from inputs and forcings
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
        # Static Variables
        "land_static",
        "hgtsfc_static",
    )
    forcing_variables = (
        "dswrf_avetoa",
    )

    # vertical grid
    interfaces = (
        200, 240, 280, 320, 360,
        470, 580, 690, 800,
        825, 850, 875, 900, 925, 950, 975, 1000
    )


class P2PPreprocessed(P2PTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None


class P2PEvaluator(P2PTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(3, 3*8*10+1, 3)]
    sample_stride = 9
    evaluation_checkpoint_id = 64
    batch_size = 32

tree_util.register_pytree_node(
    P2PTrainer,
    P2PTrainer._tree_flatten,
    P2PTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    P2PPreprocessed,
    P2PPreprocessed._tree_flatten,
    P2PPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    P2PEvaluator,
    P2PEvaluator._tree_flatten,
    P2PEvaluator._tree_unflatten
)
