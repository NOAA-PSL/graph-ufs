import os
from jax import tree_util
from prototypes.tp0.config import TP0Emulator

local_path = os.path.dirname(os.path.realpath(__file__))

class ChannelEmulator(TP0Emulator):
    weight_loss_per_channel = True
    local_store_path = f"{local_path}/channel-loss-output"

class ChannelTester(ChannelEmulator):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

tree_util.register_pytree_node(
    ChannelEmulator,
    ChannelEmulator._tree_flatten,
    ChannelEmulator._tree_unflatten
)

tree_util.register_pytree_node(
    ChannelTester,
    ChannelTester._tree_flatten,
    ChannelTester._tree_unflatten
)
