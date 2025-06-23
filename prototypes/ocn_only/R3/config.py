import os
from jax import tree_util
from prototypes.ocn_only.config import BaseOcnTrainer, _scratch

class OcnTrainer(BaseOcnTrainer):
    case = "R3" 
    local_store_path = f"{_scratch}/ocn-only/{case}"
    use_half_precision = False
    
    norm_urls = {}
    norm_urls["atm"] = {
        "mean": f"/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/6h/atm.fvstatistics.1993-2019/mean_by_level.zarr",
        "std": f"/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/6h/atm.fvstatistics.1993-2019/stddev_by_level.zarr",
        "stddiff": f"/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/6h/atm.fvstatistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    norm_urls["ocn"] = {
        "mean": f"/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/6h/ocn.fvstatistics.1993-2019/mean_by_level.zarr",
        "std": f"/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/6h/ocn.fvstatistics.1993-2019/stddev_by_level.zarr",
        "stddiff": f"/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/6h/ocn.fvstatistics.1993-2019/diffs_stddev_by_level.zarr",
    }
    norm_urls["ice"] = {
        "mean": "",
        "std": "",
        "stddiff": "",
    }
    norm_urls["land"] = {
        "mean": "",
        "std": "",
        "stddiff": "",
    }

    # vertical interfaces
    interfaces = {}
    interfaces["atm"] = (950, 1000)
    interfaces["ocn"] = (
        0,
        1,
        21,
        75,
        120,
        200,
        350,
        500,
    )
    interfaces["ice"] = tuple()
    interfaces["land"] = tuple()

    # time related
    delta_t_model = "6h"        # the model time step
    delta_t_data = "6h"         # time steps in the data
    input_duration = "6h"      # time covered by initial condition(s) + delta_t (necessary for GraphCast code)
    target_lead_time = "6h"     # how long is the forecast ... at what point do we compare model to targets

    # hyperparams
    #latent_size = 192
    #lr_peak_value = 1e-3
    #weight_decay = 0.1
    
    # training related
    num_epochs = 64

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
    delta_t = OcnTrainer.delta_t_model # string
    dt = int(delta_t[0]) if len(delta_t) == 2 else int(delta_t[:2]) # assuming delta_t_model to be bounded above by "99h"

    fcast_days = 10
    n_autoreg_steps = int(fcast_days*24)
    target_lead_time = [f"{n}h" for n in range(dt, n_autoreg_steps+1, dt)]
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
