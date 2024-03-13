
import os
import io
from functools import partial
import xarray as xr
from jax import jit
from jax.random import PRNGKey
import optax

from ufs2arco.timer import Timer

from graphcast import graphcast
from simple_emulator import P0Emulator
from graphufs import optimize, run_forward, loss_fn

# Note that adding these slows things down on PSL GPU
#os.environ['XLA_FLAGS'] = (
#    '--xla_gpu_enable_triton_softmax_fusion=true '
#    '--xla_gpu_triton_gemm_any=True '
#    '--xla_gpu_enable_async_collectives=true '
#    '--xla_gpu_enable_latency_hiding_scheduler=true '
#    '--xla_gpu_enable_highest_priority_async_stream=true '
#)


if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Starting Training")

    localtime.start("Extracting Training Batches from Replay on GCS")

    gufs = P0Emulator()

    inputs, targets, forcings = gufs.get_training_batches()
    localtime.stop()

    localtime.start("Loading Training Batches into Memory")

    inputs.load()
    targets.load()
    forcings.load()

    localtime.stop()


    localtime.start("Initializing Optimizer and Parameters")

    init_jitted = jit( run_forward.init )
    params, state = init_jitted(
        rng=PRNGKey(gufs.init_rng_seed),
        emulator=gufs,
        inputs=inputs.sel(optim_step=0),
        targets_template=targets.sel(optim_step=0),
        forcings=forcings.sel(optim_step=0),
    )

    # linearly increase learning rate
    n_linear = len(inputs.optim_step)
    schedule_1 = optax.linear_schedule(
        init_value=0.0,
        end_value=1e-3,
        transition_steps=n_linear,
    )
    # curriculum and parameters as in GraphCast
    optimizer = optax.chain(
        optax.clip_by_global_norm(32),
        optax.adamw(
            learning_rate=schedule_1,
            b1=0.9,
            b2=0.95,
            weight_decay=0.1,
        ),
    )
    localtime.stop()

    localtime.start("Starting Optimization")

    params, loss = optimize(
        params=params,
        state=state,
        optimizer=optimizer,
        emulator=gufs,
        input_batches=inputs,
        target_batches=targets,
        forcing_batches=forcings,
        store_results=True,
        description="P0 Optimized Parameters",
    )

    localtime.stop()

    walltime.stop("Total Walltime")
