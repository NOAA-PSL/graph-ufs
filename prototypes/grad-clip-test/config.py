import os
import xarray as xr
from jax import tree_util
import numpy as np

from graphufs import FVEmulator
from prototypes.tp0.config import TP0Emulator

local_path = os.path.dirname(os.path.realpath(__file__))

class GradClipTrainer(TP0Emulator):
    local_store_path = f"{local_path}/clipby-64"
    input_duration = "3h"
    grad_clip_value = 64.
    num_epochs = 50

class GradClipTester(GradClipTrainer):
    target_lead_time = ["3h", "6h", "9h", "12h", "15h", "18h", "21h", "24h"]

tree_util.register_pytree_node(
    GradClipTrainer,
    GradClipTrainer._tree_flatten,
    GradClipTrainer._tree_unflatten
)

tree_util.register_pytree_node(
    GradClipTester,
    GradClipTester._tree_flatten,
    GradClipTester._tree_unflatten
)
