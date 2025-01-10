from jax import tree_util
from prototypes.tp0.config import BaseTP0Emulator, tp0_path

class TP0Emulator(BaseTP0Emulator):

    local_store_path = f"{tp0_path}/diagnostics"

    norm_urls = {
        "mean": f"{tp0_path}/diagnostics/fvstatistics/mean_by_level.zarr",
        "std": f"{tp0_path}/diagnostics/fvstatistics/stddev_by_level.zarr",
        "stddiff": f"{tp0_path}/diagnostics/fvstatistics/diffs_stddev_by_level.zarr",
    }
    diagnostics = ("horizontal_wind_speed", "layer_thickness")
    training_dates = (          # bounds of training data (inclusive)
        "1994-01-01T00",        # start
        "1994-01-05T21"         # stop
    )
    validation_dates = (        # bounds of validation data (inclusive)
        "1996-01-01T00",        # start
        "1996-01-05T21"         # stop
    )
    input_transforms = dict()
    output_transforms = dict()

class TP0Tester(TP0Emulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

tree_util.register_pytree_node(
    TP0Emulator,
    TP0Emulator._tree_flatten,
    TP0Emulator._tree_unflatten
)

tree_util.register_pytree_node(
    TP0Tester,
    TP0Tester._tree_flatten,
    TP0Tester._tree_unflatten
)
