import os
from jax import tree_util
from prototypes.ocn_only.config import BaseOcnTrainer, _scratch

class OcnTrainer(BaseOcnTrainer):
    case = "R1" 
    local_store_path = f"{_scratch}/ocn-only/{case}"
    use_half_precision = False

    # point ocean statistics to those for 24hr time step 
    norm_urls = {}
    norm_urls["atm"] = {
        "mean": "/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/24hr/atm.fvstatistics.1993-2019/mean_by_level.zarr",
        "std": "/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/24hr/atm.fvstatistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/24hr/atm.fvstatistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    norm_urls["ocn"] = {
        "mean": "/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/24hr/ocn.fvstatistics.l10.1993-2019/mean_by_level.zarr",
        "std": "/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/24hr/ocn.fvstatistics.l10.1993-2019/stddev_by_level.zarr",
        "stddiff": "/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/24hr/ocn.fvstatistics.l10.1993-2019/diffs_stddev_by_level.zarr",
    }
    norm_urls["ice"] = {
        "mean": "",
        "std": "",
        "stddiff": "",
    }
    norm_urls["land"] = {
        "mean": "",
        "std": "",
        "stddiff": "",
    }

    # time related
    delta_t_data = "6h"
    delta_t_model = "24h"
    input_duration = "48h"
    target_lead_time = "24h"

class OcnPreprocessor(OcnTrainer):
    batch_size = 64

class OcnPreprocessed(OcnTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class OcnEvaluator(OcnTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(24, 24*180+1, 24)]
    sample_stride = 5
    #evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    OcnTrainer,
    OcnTrainer._tree_flatten,
    OcnTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    OcnPreprocessor,
    OcnPreprocessor._tree_flatten,
    OcnPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    OcnPreprocessed,
    OcnPreprocessed._tree_flatten,
    OcnPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    OcnEvaluator,
    OcnEvaluator._tree_flatten,
    OcnEvaluator._tree_unflatten
)
