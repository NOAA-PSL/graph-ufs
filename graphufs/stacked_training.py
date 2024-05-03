"""
Same as training.py, but for StackedGraphCast
"""

import os
import logging
import warnings
from functools import partial
import numpy as np
import xarray as xr
from jax import (
    jit,
    value_and_grad,
    tree_util,
    local_devices,
    devices,
    local_device_count,
    device_count,
    print_environment_info,
    distributed,
    block_until_ready,
)
from graphcast.xarray_jax import pmap
from jax.lax import pmean
from jax.random import PRNGKey
import jax.numpy as jnp
import optax
import haiku as hk
import xarray as xr
from math import ceil

from graphcast.checkpoint import dump
from graphcast.stacked_graphcast import StackedGraphCast
from graphcast.stacked_casting import StackedBfloat16Cast
from graphcast.xarray_tree import map_structure
from graphcast.stacked_normalization import StackedInputsAndResiduals
from graphcast.xarray_jax import unwrap_data
from graphcast import rollout

from tqdm import tqdm

try:
    from mpi4py import MPI
    import mpi4jax
except:
    warnings.warn("Import failed for either mpi4py or mpi4jax.")


def construct_wrapped_graphcast(emulator, last_input_channel_mapping):
    """Constructs and wraps the GraphCast Predictor object"""

    predictor = StackedGraphCast(emulator.model_config, emulator.task_config)

    # handle inputs/outputs float32 <-> BFloat16
    # ... and so that this happens after applying
    # normalization to inputs & targets
    predictor = StackedBfloat16Cast(predictor)
    predictor = StackedInputsAndResiduals(
        predictor,
        diffs_stddev_by_level=emulator.stacked_norm["stddiff"],
        mean_by_level=emulator.stacked_norm["mean"],
        stddev_by_level=emulator.stacked_norm["std"],
        last_input_channel_mapping=last_input_channel_mapping,
    )
    # multi step rollout is not implemented yet
    return predictor


def init_model(emulator, gds, last_input_channel_mapping):
    """Initialize model with random weights.
    """

    @hk.transform_with_state
    def run_forward(emulator, last_input_channel_mapping, inputs):
        predictor = construct_wrapped_graphcast(emulator, last_input_channel_mapping)
        return predictor(inputs)

    inputs, _ = gds[0]
    init_jitted = jit(run_forward.init)
    params, state = init_jitted(
        rng=PRNGKey(emulator.init_rng_seed),
        emulator=emulator,
        last_input_channel_mapping=last_input_channel_mapping,
        inputs=inputs,
    )
    return params, state


def optimize(
    params, state, optimizer, emulator, generator, weights, last_input_channel_mapping
):
    """Optimize the model parameters by running through all optim_steps in data

    Args:
        params (dict): with the initialized model parameters
        state (dict): this is empty, but for now has to be here
        optimizer (Callable, optax.optimizer): see `here <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_
        emulator (ReplayEmulator): the emulator object
        input_batches, training_batches (chex.Array): with data needed for training

    Returns:
        params (dict): optimized model parameters
        loss_ds (xarray.Dataset): with the total loss function and loss per variable for each optim_step
            this doesn't have gradient info, but we could add that
    """

    opt_state = optimizer.init(params)
    num_gpus = emulator.num_gpus
    mpi_size = emulator.mpi_size
    use_jax_distributed = emulator.use_jax_distributed
    if use_jax_distributed:
        raise NotImplementedError
    if num_gpus > 1:
        raise NotImplementedError

    @hk.transform_with_state
    def loss_fn(inputs, targets):
        predictor = construct_wrapped_graphcast(emulator, last_input_channel_mapping)
        loss, diagnostics = predictor.loss(inputs, targets, weights=weights)
        return loss.mean(), diagnostics.mean(axis=0)

    def optim_step(
        params,
        state,
        opt_state,
        input_batch,
        target_batch,
    ):
        """Note that this function has to be definied within optimize so that we do not
        pass optimizer as an argument. Otherwise we get some craazy jax errors"""

        # NOTE I think this can be deleted and we can just use loss_fn.apply directly
        def _aux(params, state, i, t):
            (loss, diagnostics), next_state = loss_fn.apply(
                inputs=i,
                targets=t,
                params=params,
                state=state,
                rng=PRNGKey(0),
            )
            return loss, (diagnostics, next_state)

        # process one batch per GPU
        def process_batch(inputs, targets):
            (loss, (diagnostics, next_state)), grads = value_and_grad(
                _aux, has_aux=True
            )(
                params,
                state,
                inputs,
                targets,
            )
            return (loss, (diagnostics, next_state)), grads

        (loss, (diagnostics, next_state)), grads = process_batch(
            input_batch,
            target_batch,
        )

        # update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss, diagnostics, opt_state, grads


    # jit optim_step only once
    if not hasattr(optimize, "optim_step_jitted"):
        logging.info("Started jitting optim_step")

        # jitted function
        optimize.optim_step_jitted = jit(optim_step)
        first_input, first_target = next(iter(generator))
        x, *_ = optimize.optim_step_jitted(
            params=params,
            state=state,
            opt_state=opt_state,
            input_batch=first_input,
            target_batch=first_target,
        )
        block_until_ready(x)

        # Unclear if it's safe to assume whether we'll have the drop_last attr or not
        if not generator.drop_last:
            # this is necessary to let the JIT compiler see the last batch,
            # which may be a different size
            # I'm not sure if this pulls everything into memory though ...
            *_, (last_input, last_target) = iter(generator)
            y, *_ = optimize.optim_step_jitted(
                params=params,
                state=state,
                opt_state=opt_state,
                input_batch=last_input,
                target_batch=last_target,
            )
            block_until_ready(y)
        logging.info("Finished jitting optim_step")

    else:
        logging.info("optim_step already jitted")

    optim_steps = []
    loss_values = []
    loss_by_channel = []

    n_steps = len(generator)

    progress_bar = tqdm(total=n_steps, ncols=140, desc="Processing")
    for k, (input_batches, target_batches) in enumerate(generator):

        # call optimize
        params, loss, diagnostics, _, grads = optimize.optim_step_jitted(
            params=params,
            state=state,
            opt_state=opt_state,
            input_batch=input_batches,
            target_batch=target_batches,
        )

        # update progress bar from rank 0
        optim_steps.append(k)
        loss_values.append(loss)
        loss_by_channel.append(diagnostics)

        mean_grad = np.mean(
            tree_util.tree_flatten(
                tree_util.tree_map(lambda x: np.abs(x).mean(), grads)
            )[0]
        )
        progress_bar.set_description(
            f"[{emulator.mpi_rank}] loss = {loss:.5f}, mean(|grad|) = {mean_grad:.8f}"
        )
        progress_bar.update(num_gpus)

    progress_bar.close()

    # save losses for each batch
    loss_ds = xr.Dataset()
    loss_fname = os.path.join(emulator.local_store_path, "loss.nc")
    previous_optim_steps = 0
    if os.path.exists(loss_fname):
        stored_loss_ds = xr.open_dataset(loss_fname)
        previous_optim_steps = len(stored_loss_ds.optim_step)

    loss_by_channel = np.vstack(loss_by_channel)
    loss_ds["optim_step"] = [x + previous_optim_steps for x in optim_steps]
    loss_ds.attrs["batch_size"] = emulator.batch_size
    loss_ds["channels"] = xr.DataArray(
        np.arange(loss_by_channel.shape[-1]),
        coords={"channels": np.arange(loss_by_channel.shape[-1])},
        dims=("channels",),
    )
    loss_ds["loss"] = xr.DataArray(
        loss_values,
        coords={"optim_step": loss_ds["optim_step"]},
        dims=("optim_step",),
        attrs={"long_name": "loss function value"},
    )
    loss_ds["loss_by_channel"] = xr.DataArray(
        loss_by_channel,
        dims=("optim_step", "channels"),
    )

    # concatenate losses and store
    if os.path.exists(loss_fname):
        stored_loss_ds = xr.concat([stored_loss_ds, loss_ds], dim="optim_step")
    else:
        stored_loss_ds = loss_ds
    stored_loss_ds.to_netcdf(loss_fname)

    return params, loss_ds
