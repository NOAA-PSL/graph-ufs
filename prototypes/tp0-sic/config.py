import os
from jax import tree_util
from prototypes.tp0.config import TP0Emulator

sic_path = os.path.dirname(os.path.realpath(__file__))


class SICEmulator(TP0Emulator):
    local_store_path = f"{sic_path}/local-output"
    input_duration = "3h"

class SICTester(SICEmulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

class TICEmulator(TP0Emulator):
    local_store_path = f"{sic_path}/local-output-3ic"
    input_duration = "9h"

class TICTester(TICEmulator):
    target_lead_time = ("3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h")

tree_util.register_pytree_node(
    SICEmulator,
    SICEmulator._tree_flatten,
    SICEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    SICTester,
    SICTester._tree_flatten,
    SICTester._tree_unflatten
)

tree_util.register_pytree_node(
    TICEmulator,
    TICEmulator._tree_flatten,
    TICEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    TICTester,
    TICTester._tree_flatten,
    TICTester._tree_unflatten
)
