import os
from jax import tree_util
from prototypes.atm_wocnsfc_land_ice.config import BaseAtmOcnSfcLandIceTrainer, _scratch

class AtmOcnSfcLandIceTrainer(BaseAtmOcnSfcLandIceTrainer):
    case = "R0"
    local_store_path = f"{_scratch}/atm-wocnsfc-land-ice/{case}"
    use_half_precision = False
    
class AtmOcnSfcLandIcePreprocessor(AtmOcnSfcLandIceTrainer):
    batch_size = 64

class AtmOcnSfcLandIcePreprocessed(AtmOcnSfcLandIceTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class AtmOcnSfcLandIceEvaluator(AtmOcnSfcLandIceTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    sample_stride = 5
    #evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    AtmOcnSfcLandIceTrainer,
    AtmOcnSfcLandIceTrainer._tree_flatten,
    AtmOcnSfcLandIceTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnSfcLandIcePreprocessor,
    AtmOcnSfcLandIcePreprocessor._tree_flatten,
    AtmOcnSfcLandIcePreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnSfcLandIcePreprocessed,
    AtmOcnSfcLandIcePreprocessed._tree_flatten,
    AtmOcnSfcLandIcePreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnSfcLandIceEvaluator,
    AtmOcnSfcLandIceEvaluator._tree_flatten,
    AtmOcnSfcLandIceEvaluator._tree_unflatten
)
