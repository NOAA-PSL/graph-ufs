import os
from jax import tree_util
from prototypes.ocn_only.config import BaseOcnTrainer, _scratch

class OcnTrainer(BaseOcnTrainer):
    case = "R1" 
    local_store_path = f"{_scratch}/ocn-only/{case}"
    use_half_precision = False
    norm_urls = {}
    norm_urls["atm"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.fvstatistics.trop16.1993-2019/diffs_stddev_by_level.zarr",
    }
    
    norm_urls["ocn"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/mom6.fvstatistics.l10.1993-2019/sixtimes/diffs_stddev_by_level.zarr",
    }
    
    norm_urls["ice"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.ice.statistics.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.ice.statistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.ice.statistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    
    norm_urls["land"] = {
        "mean": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.land.statistics.1993-2019/mean_by_level.zarr",
        "std": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.land.statistics.1993-2019/stddev_by_level.zarr",
        "stddiff": "gcs://noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree-subsampled/06h-freq/zarr/fv3.land.statistics.1993-2019/diffs_stddev_by_level.zarr",
    }
   
    weight_loss_per_level_ocn = True

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
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
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
