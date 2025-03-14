import os
from jax import tree_util
from prototypes.ocn_only.config import BaseOcnTrainer, _scratch

class OcnTrainer(BaseOcnTrainer):
    case = "R0" 
    local_store_path = f"{_scratch}/ocn-only/{case}"
    use_half_precision = False
    
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
