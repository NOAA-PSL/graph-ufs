import os
from jax import tree_util
from prototypes.ocn_only.config import BaseOcnTrainer, _scratch

class OcnTrainer(BaseOcnTrainer):
    # === Configuration ===
    case = "R2"
    local_store_path = f"{_scratch}/ocn-only/{case}"
    use_half_precision = False

    # === Normalization statistics paths ===
    _base_path = (
        "/global/homes/n/nagarwal/graph-ufs/prototypes/ocn_only/statistics/"
        f"ocn_only_7L_coarseTop/{delta_t_model}"
    )

    norm_urls = {
        "atm": {
            "mean": f"{_base_path}/atm.fvstatistics.1993-2019/mean_by_level.zarr",
            "std": f"{_base_path}/atm.fvstatistics.1993-2019/stddev_by_level.zarr",
            "stddiff": (
                f"{_base_path}/atm.fvstatistics.1993-2019/"
                "diffs_stddev_by_level.zarr"
            ),
        },
        "ocn": {
            "mean": f"{_base_path}/ocn.fvstatistics.1993-2019/mean_by_level.zarr",
            "std": f"{_base_path}/ocn.fvstatistics.1993-2019/stddev_by_level.zarr",
            "stddiff": (
                f"{_base_path}/ocn.fvstatistics.1993-2019/"
                "diffs_stddev_by_level.zarr"
            ),
        },
        "ice": {
            "mean": "",
            "std": "",
            "stddiff": ""
        },
        "land": {
            "mean": "",
            "std": "",
            "stddiff": ""
        },
    }

    # === Vertical layer interfaces ===
    interfaces = {
        "atm": (950, 1000),
        "ocn": (0, 1, 21, 75, 120, 200, 350, 500),
        "ice": tuple(),
        "land": tuple(),
    }

    # === Hyperparameters ===
    #latent_size = 256
    #lr_peak_value = 1e-4
    #weight_decay = 0.01

class OcnPreprocessor(OcnTrainer):
    batch_size = 64

class OcnPreprocessed(OcnTrainer):
    """
    The log transform has already been taken care of during preprocessing.
    This version operates on transformed (preprocessed) data, so needs no 
    transforms.
    """
    input_transforms = None
    output_transforms = None

class OcnEvaluator(OcnTrainer):
    # === Reference truth dataset (e.g., ERA5 from WeatherBench2) ===
    wb2_obs_url = (
        "gs://weatherbench2/datasets/era5/"
        "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
    )

    # === Forecast time settings ===
    delta_t = OcnTrainer.delta_t_model  # string, like "6h"

    # Parse the number of hours from delta_t (assumes format like "6h" or "12h")
    dt = int(delta_t[:-1])  # strips trailing "h"

    fcast_days = 10
    n_autoreg_steps = fcast_days * 24  # total hours of forecasting

    # List of target lead times: ['6h', '12h', ..., '240h']
    target_lead_time = [f"{n}h" for n in range(dt, n_autoreg_steps + 1, dt)]

    # Sampling stride (controls data subset during evaluation)
    sample_stride = 5

    # Uncomment to specify evaluation checkpoint explicitly
    # evaluation_checkpoint_id = 64

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
