import os
from jax import tree_util
from prototypes.tp0.config import TP0Emulator, TP0Tester

local_path = os.path.dirname(os.path.realpath(__file__))

class TP0LightEmulator(TP0Emulator):
    local_store_path = f"{local_path}/output-light"
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
        "dswrf_avetoa",
    )
    forcing_variables = (
        "dswrf_avetoa",
    )

    input_duration = "3h"

    weight_loss_per_channel = True


class TP0LightTester(TP0LightEmulator):
    #target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]
    target_lead_time = [f"{x}h" for x in range(3, 49, 3)]

class TP0Light2Emulator(TP0LightEmulator):
    local_store_path = f"{local_path}/output-light2"
    input_duration = "6h"

class TP0Light2Tester(TP0Light2Emulator):
   # target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]
    target_lead_time = [f"{x}h" for x in range(3, 49, 3)]

class TP0Light2NCEmulator(TP0Emulator):
    local_store_path = f"{local_path}/output-light2-nc"
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
        "dswrf_avetoa",
        "land_static",
        "hgtsfc_static",
    )
    forcing_variables = (
        "dswrf_avetoa",
    )

    weight_loss_per_channel = True

class TP0Light2NCTester(TP0Light2NCEmulator):
    #target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]
    target_lead_time = [f"{x}h" for x in range(3, 49, 3)]


class TP0Light2WSEmulator(TP0Emulator):
    local_store_path = f"{local_path}/output-light2-ws"
    weight_loss_per_channel = True

class TP0Light2WSTester(TP0Light2WSEmulator):
    #target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]
    target_lead_time = [f"{x}h" for x in range(3, 49, 3)]

class TP02DayTester(TP0Tester):
    target_lead_time = [f"{x}h" for x in range(3, 49, 3)]

tree_util.register_pytree_node(
    TP0LightEmulator,
    TP0LightEmulator._tree_flatten,
    TP0LightEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    TP0LightTester,
    TP0LightTester._tree_flatten,
    TP0LightTester._tree_unflatten
)

tree_util.register_pytree_node(
    TP0Light2Emulator,
    TP0Light2Emulator._tree_flatten,
    TP0Light2Emulator._tree_unflatten
)

tree_util.register_pytree_node(
    TP0Light2Tester,
    TP0Light2Tester._tree_flatten,
    TP0Light2Tester._tree_unflatten
)

tree_util.register_pytree_node(
    TP0Light2NCEmulator,
    TP0Light2NCEmulator._tree_flatten,
    TP0Light2NCEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    TP0Light2NCTester,
    TP0Light2NCTester._tree_flatten,
    TP0Light2NCTester._tree_unflatten
)

tree_util.register_pytree_node(
    TP0Light2WSEmulator,
    TP0Light2WSEmulator._tree_flatten,
    TP0Light2WSEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    TP0Light2WSTester,
    TP0Light2WSTester._tree_flatten,
    TP0Light2WSTester._tree_unflatten
)

tree_util.register_pytree_node(
    TP02DayTester,
    TP02DayTester._tree_flatten,
    TP02DayTester._tree_unflatten
)
