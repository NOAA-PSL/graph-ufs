import os
from jax import tree_util
from prototypes.atm_wocn.config import BaseAtmOcnTrainer, _scratch

class AtmOcnTrainer(BaseAtmOcnTrainer):
    case = "R0"
    local_store_path = f"{_scratch}/atm-wocn/{case}"
    use_half_precision = False
    
class AtmOcnPreprocessor(AtmOcnTrainer):
    batch_size = 64

class AtmOcnPreprocessed(AtmOcnTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class AtmOcnEvaluator(AtmOcnTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    sample_stride = 5
    #evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    AtmOcnTrainer,
    AtmOcnTrainer._tree_flatten,
    AtmOcnTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnPreprocessor,
    AtmOcnPreprocessor._tree_flatten,
    AtmOcnPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnPreprocessed,
    AtmOcnPreprocessed._tree_flatten,
    AtmOcnPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnEvaluator,
    AtmOcnEvaluator._tree_flatten,
    AtmOcnEvaluator._tree_unflatten
)
