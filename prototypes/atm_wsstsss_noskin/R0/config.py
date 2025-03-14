import os
from jax import tree_util
from prototypes.atm_wsstsss_noskin.config import BaseAtmSSTSSSTrainer, _scratch

class AtmSSTSSSTrainer(BaseAtmSSTSSSTrainer):
    case = os.getcwd().split('/')[-1] 
    local_store_path = f"{_scratch}/atm-wsstsss-noskin/{case}"
    use_half_precision = False
    
class AtmSSTSSSPreprocessor(AtmSSTSSSTrainer):
    batch_size = 64

class AtmSSTSSSPreprocessed(AtmSSTSSSTrainer):
    """The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no transforms.
    """
    input_transforms = None
    output_transforms = None

class AtmSSTSSSEvaluator(AtmSSTSSSTrainer):
    wb2_obs_url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    target_lead_time = [f"{n}h" for n in range(6, 6*4*10+1, 6)]
    sample_stride = 5
    #evaluation_checkpoint_id = 64

tree_util.register_pytree_node(
    AtmSSTSSSTrainer,
    AtmSSTSSSTrainer._tree_flatten,
    AtmSSTSSSTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    AtmSSTSSSPreprocessor,
    AtmSSTSSSPreprocessor._tree_flatten,
    AtmSSTSSSPreprocessor._tree_unflatten
)

tree_util.register_pytree_node(
    AtmSSTSSSPreprocessed,
    AtmSSTSSSPreprocessed._tree_flatten,
    AtmSSTSSSPreprocessed._tree_unflatten
)

tree_util.register_pytree_node(
    AtmSSTSSSEvaluator,
    AtmSSTSSSEvaluator._tree_flatten,
    AtmSSTSSSEvaluator._tree_unflatten
)
