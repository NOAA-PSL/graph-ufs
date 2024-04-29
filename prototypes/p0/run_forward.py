
import numpy as np
from functools import partial
import xarray as xr
from jax import jit
from jax.random import PRNGKey
import optax

from ufs2arco.timer import Timer

from simple_emulator import P0Emulator
from graphufs import run_forward, DataGenerator, init_devices


if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Testing run_forward...")

    localtime.start("Extracting Batches from Replay on GCS")

    gufs = P0Emulator()

    # for multi-gpu training
    init_devices(gufs)

    # data generator
    generator = DataGenerator(
        emulator = gufs,
        n_optim_steps=2,
        mode="testing",
    )

    data = generator.get_data()
    inputs = data["inputs"]
    targets = data["targets"]
    forcings = data["forcings"]

    localtime.stop()

    localtime.start("Loading Training Batches into Memory")

    inputs.load()
    targets.load()
    forcings.load()

    localtime.stop()


    localtime.start("Initializing Parameters and State")

    init_jitted = jit( run_forward.init )
    params, state = init_jitted(
        rng=PRNGKey(gufs.init_rng_seed),
        emulator=gufs,
        inputs=inputs.sel(optim_step=0),
        targets_template=targets.sel(optim_step=0),
        forcings=forcings.sel(optim_step=0),
    )
    localtime.stop()

    localtime.start("Making Predictions")

    fwd_jitted = jit( run_forward.apply )
    predictions, state = fwd_jitted(
        emulator=gufs,
        inputs=inputs.sel(optim_step=0),
        targets_template=targets.sel(optim_step=0) * np.nan,
        forcings=forcings.sel(optim_step=0),
        rng=PRNGKey(0),
        params=params,
        state=state,
    )
    localtime.stop()

    walltime.stop("Total Walltime")
