import os
from jax import tree_util
from prototypes.atm_only.config import BaseAtmTrainer, _scratch

class AtmTrainer(BaseAtmTrainer):
    case = "R0" 
    local_store_path = f"{_scratch}/atm-only/{case}"
    use_half_precision = False
    
class AtmPreprocessor(AtmTrainer):
    batch_size = 64

class AtmPreprocessed(AtmTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class AtmEvaluator(AtmTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    sample_stride = 5
    #evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    AtmTrainer,
    AtmTrainer._tree_flatten,
    AtmTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    AtmPreprocessor,
    AtmPreprocessor._tree_flatten,
    AtmPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    AtmPreprocessed,
    AtmPreprocessed._tree_flatten,
    AtmPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    AtmEvaluator,
    AtmEvaluator._tree_flatten,
    AtmEvaluator._tree_unflatten
)
