
import numpy as np
from functools import partial
import xarray as xr
from jax import jit
from jax.random import PRNGKey
import optax

from ufs2arco.timer import Timer

from simple_emulator import P0Emulator
from graphufs import run_forward


if __name__ == "__main__":

    walltime = Timer()
    localtime = Timer()

    walltime.start("Testing run_forward...")

    localtime.start("Extracting Batches from Replay on GCS")

    gufs = P0Emulator()

    ds = xr.open_zarr(gufs.data_url, storage_options={"token": "anon"})
    inputs, targets, forcings = gufs.get_training_batches(
        xds=ds,
        n_batches=2,
        batch_size=1,
        delta_t="6h",
        target_lead_time="12h",
    )
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
        inputs=inputs.sel(batch=[0]),
        targets_template=targets.sel(batch=[0]),
        forcings=forcings.sel(batch=[0]),
    )
    localtime.stop()

    localtime.start("Making Predictions")

    fwd_jitted = jit( run_forward.apply )
    predictions, state = fwd_jitted(
        emulator=gufs,
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
        rng=PRNGKey(0),
        params=params,
        state=state,
    )
    localtime.stop()

    walltime.stop("Total Walltime")
