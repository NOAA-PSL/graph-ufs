from prototypes.p3.uvnc.config import P3Trainer as UVNCP3Trainer, _scratch

class P3Trainer(UVNCP3Trainer):

    # paths
    local_store_path = f"{_scratch}/p3/uvncbs32"

    # training protocol
    batch_size = 32
    num_epochs = 128

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
    evaluation_checkpoint_id = 128


tree_util.register_pytree_node(
    P3Trainer,
    P3Trainer._tree_flatten,
    P3Trainer._tree_unflatten
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
