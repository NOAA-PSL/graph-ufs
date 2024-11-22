import os
import logging
from mpi4py import MPI
import jax
import jax.numpy as jnp

from graphufs.mpi import MPITopology
from graphufs.utils import load_checkpoint

scratch = os.getenv("SCRATCH")
topo = MPITopology(log_dir=f"{SCRATCH}/test-mpi4jax")
params, _, _ = load_checkpoint("/pscratch/sd/t/timothys/p2p/models/model_0.npz")
logging.info("loaded parameters")


@jax.jit
def foo(arr, local_params):
   rng = jax.random.PRNGKey(8)
   arr = arr + topo.rank
   myparams = jax.tree_util.tree_map(lambda x: x+topo.rank, local_params)
   arr_sum = topo.device_mean(arr)
   params_avg = topo.device_mean(myparams)

   return arr_sum, params_avg

a = jnp.zeros((3, 3))
rng = jax.random.PRNGKey(8)
x, y = foo(a, params)



if topo.is_root:
   print(x)
   print(list(y.keys())[0])
