from jax import tree_util
from prototypes.tp0.config import BaseTP0Emulator, tp0_path

class TP0Emulator(BaseTP0Emulator):

    local_store_path = f"{tp0_path}/diagnostics"

    norm_urls = {
        "mean": f"{tp0_path}/diagnostics/fvstatistics/mean_by_level.zarr",
        "std": f"{tp0_path}/diagnostics/fvstatistics/stddev_by_level.zarr",
        "stddiff": f"{tp0_path}/diagnostics/fvstatistics/diffs_stddev_by_level.zarr",
    }
    diagnostics = (
        "10m_horizontal_wind_speed",
        "horizontal_wind_speed",
        "hydrostatic_layer_thickness",
        "hydrostatic_geopotential",
    )
    num_epochs = 10

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
