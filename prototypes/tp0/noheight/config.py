from jax import tree_util
from prototypes.tp0.config import BaseTP0Emulator, tp0_path

class TP0Emulator(BaseTP0Emulator):
    local_store_path = f"{tp0_path}/noheight"

    input_variables = (
        "pressfc",
        "tmp2m",
        "spfh2m",
        "ugrd10m",
        "vgrd10m",
        "tmp",
        "spfh",
        "ugrd",
        "vgrd",
        "land_static",
        #"hgtsfc_static",
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )

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
