import xarray as xr
from jax import tree_util
import numpy as np

from graphufs import FVEmulator
from prototypes.tp0.config import TP0Emulator

class BatchTester(TP0Emulator):
    local_store_path = "clipby-batch-16"
    batch_size = 16
    num_epochs = 4

tree_util.register_pytree_node(
    BatchTester,
    BatchTester._tree_flatten,
    BatchTester._tree_unflatten
)
