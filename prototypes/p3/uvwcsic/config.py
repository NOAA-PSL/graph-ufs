from jax import tree_util
from prototypes.p3.config import BaseP3Trainer, _scratch

class P3Trainer(BaseP3Trainer):

    local_store_path = f"{_scratch}/p3/uvwcsic"
    input_duration = "3h"

class P3Preprocessor(P3Trainer):

    batch_size = 64

class P3Preprocessed(P3Trainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None


class P3Evaluator(P3Trainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(3, 3*8*10+1, 3)]
    sample_stride = 9
    evaluation_checkpoint_id = 64


tree_util.register_pytree_node(
    P3Trainer,
    P3Trainer._tree_flatten,
    P3Trainer._tree_unflatten
)

tree_util.register_pytree_node(
    P3Preprocessor,
    P3Preprocessor._tree_flatten,
    P3Preprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    P3Preprocessed,
    P3Preprocessed._tree_flatten,
    P3Preprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    P3Evaluator,
    P3Evaluator._tree_flatten,
    P3Evaluator._tree_unflatten
)
