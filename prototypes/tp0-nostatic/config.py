import os
from jax import tree_util
from prototypes.tp0.config import TP0Emulator

local_path = os.path.dirname(os.path.realpath(__file__))

class NoLandEmulator(TP0Emulator):
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
        #"land_static",
        "hgtsfc_static",
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        "day_progress_sin",
        "day_progress_cos",
    )
    local_store_path = f"{local_path}/output-no-land"

class NoLandTester(NoLandEmulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

class NoHeightEmulator(TP0Emulator):
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
    local_store_path = f"{local_path}/output-no-height"

class NoHeightTester(NoHeightEmulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

class NoDayClockEmulator(TP0Emulator):
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
        "hgtsfc_static",
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        #"day_progress_sin",
        #"day_progress_cos",
    )
    forcing_variables = (
        "dswrf_avetoa",
        "year_progress_sin",
        "year_progress_cos",
        #"day_progress_sin",
        #"day_progress_cos",
    )
    local_store_path = f"{local_path}/output-no-dayclock"

class NoDayClockTester(NoDayClockEmulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

class NoAllEmulator(TP0Emulator):
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
        #"land_static",
        #"hgtsfc_static",
        "dswrf_avetoa",
#        "year_progress_sin",
#        "year_progress_cos",
        #"day_progress_sin",
        #"day_progress_cos",
    )
    forcing_variables = (
        "dswrf_avetoa",
#        "year_progress_sin",
#        "year_progress_cos",
        #"day_progress_sin",
        #"day_progress_cos",
    )
    local_store_path = f"{local_path}/output-no-all"

class NoAllTester(NoAllEmulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]


class NoClockEmulator(TP0Emulator):
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
        "hgtsfc_static",
        "dswrf_avetoa",
        #"year_progress_sin",
        #"year_progress_cos",
        #"day_progress_sin",
        #"day_progress_cos",
    )
    forcing_variables = (
        "dswrf_avetoa",
#        "year_progress_sin",
#        "year_progress_cos",
        #"day_progress_sin",
        #"day_progress_cos",
    )
    local_store_path = f"{local_path}/output-no-clock"

class NoClockTester(NoClockEmulator):
    target_lead_time = [f"{x}h" for x in range(3, 49, 3)]

tree_util.register_pytree_node(
    NoLandEmulator,
    NoLandEmulator._tree_flatten,
    NoLandEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    NoLandTester,
    NoLandTester._tree_flatten,
    NoLandTester._tree_unflatten
)

tree_util.register_pytree_node(
    NoHeightEmulator,
    NoHeightEmulator._tree_flatten,
    NoHeightEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    NoHeightTester,
    NoHeightTester._tree_flatten,
    NoHeightTester._tree_unflatten
)

tree_util.register_pytree_node(
    NoDayClockEmulator,
    NoDayClockEmulator._tree_flatten,
    NoDayClockEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    NoDayClockTester,
    NoDayClockTester._tree_flatten,
    NoDayClockTester._tree_unflatten
)

tree_util.register_pytree_node(
    NoAllEmulator,
    NoAllEmulator._tree_flatten,
    NoAllEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    NoAllTester,
    NoAllTester._tree_flatten,
    NoAllTester._tree_unflatten
)

tree_util.register_pytree_node(
    NoClockEmulator,
    NoClockEmulator._tree_flatten,
    NoClockEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    NoClockTester,
    NoClockTester._tree_flatten,
    NoClockTester._tree_unflatten
)
