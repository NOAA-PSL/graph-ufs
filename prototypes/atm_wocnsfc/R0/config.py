import os
from jax import tree_util
from prototypes.atm_wocnsfc.config import BaseAtmOcnSfcTrainer, _scratch

class AtmOcnSfcTrainer(BaseAtmOcnSfcTrainer):
    case = "R0"
    local_store_path = f"{_scratch}/atm-wocnsfc/{case}"
    use_half_precision = False
    
class AtmOcnSfcPreprocessor(AtmOcnSfcTrainer):
    batch_size = 64

class AtmOcnSfcPreprocessed(AtmOcnSfcTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class AtmOcnSfcEvaluator(AtmOcnSfcTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*100+1, 6)]
    sample_stride = 5
    #evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    AtmOcnSfcTrainer,
    AtmOcnSfcTrainer._tree_flatten,
    AtmOcnSfcTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnSfcPreprocessor,
    AtmOcnSfcPreprocessor._tree_flatten,
    AtmOcnSfcPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnSfcPreprocessed,
    AtmOcnSfcPreprocessed._tree_flatten,
    AtmOcnSfcPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    AtmOcnSfcEvaluator,
    AtmOcnSfcEvaluator._tree_flatten,
    AtmOcnSfcEvaluator._tree_unflatten
)
