from jax import tree_util
from prototypes.p2p.config import BaseP2PTrainer, _scratch

class P2PTrainer(BaseP2PTrainer):

    local_store_path = f"{_scratch}/p2p/uvwc"

class P2PPreprocessor(P2PTrainer):

    batch_size = 64

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


tree_util.register_pytree_node(
    P2PTrainer,
    P2PTrainer._tree_flatten,
    P2PTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    P2PPreprocessor,
    P2PPreprocessor._tree_flatten,
    P2PPreprocessor._tree_unflatten
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
