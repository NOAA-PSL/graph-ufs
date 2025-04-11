import os
from jax import tree_util
from prototypes.atm_only.config import BaseAtmTrainer, _scratch

class Trainer(BaseAtmTrainer):
    case = "R0" 
    local_store_path = f"{_scratch}/atm-only/{case}"
    use_half_precision = False
    
class Preprocessor(Trainer):
    batch_size = 64

class Preprocessed(Trainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class Evaluator(Trainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    sample_stride = 5
    #evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    Trainer,
    Trainer._tree_flatten,
    Trainer._tree_unflatten
)

tree_util.register_pytree_node(
    Preprocessor,
    Preprocessor._tree_flatten,
    Preprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    Preprocessed,
    Preprocessed._tree_flatten,
    Preprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    Evaluator,
    Evaluator._tree_flatten,
    Evaluator._tree_unflatten
)
