
from functools import partial
import xarray as xr
from jax import jit
from jax.random import PRNGKey
import optax

from ufs2arco.timer import Timer

from simple_emulator import P0Emulator
from graphufs import optimize, run_forward, loss_fn


if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Starting Training")

    localtime.start("Extracting Training Batches from Replay on GCS")

    gufs = P0Emulator()

    ds = xr.open_zarr(gufs.data_url, storage_options={"token": "anon"})
    inputs, targets, forcings = gufs.get_training_batches(
        xds=ds,
        n_optim_steps=4,
        batch_size=2,
        delta_t="6h",
        target_lead_time="6h",
    )
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
    optimizer = optax.adam(learning_rate=1e-4)
    localtime.stop()

    loss_jitted = jit( loss_fn.apply )
    loss_val = loss_jitted(
        rng=PRNGKey(gufs.init_rng_seed),
        params=params,
        state=state,
        emulator=gufs,
        inputs=inputs.sel(optim_step=0),
        targets=targets.sel(optim_step=0),
        forcings=forcings.sel(optim_step=0),
    )
    print(loss_val)


    localtime.start("Starting Optimization")

    params, loss, diagnostics, opt_state, grads = optimize(
        params=params,
        state=state,
        optimizer=optimizer,
        emulator=gufs,
        input_batches=inputs,
        target_batches=targets,
        forcing_batches=forcings,
    )

    localtime.stop()

    walltime.stop("Total Walltime")
