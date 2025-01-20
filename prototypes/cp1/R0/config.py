import os
from jax import tree_util
from prototypes.cp1.config import BaseCP1Trainer, _scratch

class CP1Trainer(BaseCP1Trainer):
    case = os.getcwd().split('/')[-1] 
    local_store_path = f"{_scratch}/cp1/{case}"
    use_half_precision = False
    
class CP1Preprocessor(CP1Trainer):
    batch_size = 64

class CP1Preprocessed(CP1Trainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class CP1Evaluator(CP1Trainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    sample_stride = 5
    evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    CP1Trainer,
    CP1Trainer._tree_flatten,
    CP1Trainer._tree_unflatten
)

tree_util.register_pytree_node(
    CP1Preprocessor,
    CP1Preprocessor._tree_flatten,
    CP1Preprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    CP1Preprocessed,
    CP1Preprocessed._tree_flatten,
    CP1Preprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    CP1Evaluator,
    CP1Evaluator._tree_flatten,
    CP1Evaluator._tree_unflatten
)
