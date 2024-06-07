"""
Same as training.py, but for StackedGraphCast
"""

import os
import logging
import warnings
from functools import partial
import numpy as np
import xarray as xr
import jax
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
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils
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


def init_model(emulator, inputs, last_input_channel_mapping):
    """Initialize model with random weights.
    """

    @hk.transform_with_state
    def run_forward(inputs):
        predictor = construct_wrapped_graphcast(emulator, last_input_channel_mapping)
        return predictor(inputs)

    devices = jax.devices()
    sharding = PositionalSharding(devices)
    sharding = sharding.reshape((emulator.num_gpus, 1, 1, 1))

    inputs = jax.device_put(inputs, sharding)

    init = jax.jit( run_forward.init )
    params, state = init(
        rng=PRNGKey(emulator.init_rng_seed),
        inputs=inputs,
    )
    return params, state


def optimize(
    params, state, optimizer, emulator, trainer, validator, weights, last_input_channel_mapping, opt_state=None
):
    """Optimize the model parameters by running through all optim_steps in data

    Args:
        params (dict): with the initialized model parameters
        state (dict): this is empty, but for now has to be here
        optimizer (Callable, optax.optimizer): see `here <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_
        emulator (ReplayEmulator): the emulator object
        trainer (Generator): with data needed for training
        validator (Generator): with data needed for training

    Returns:
        params (dict): optimized model parameters
        loss_ds (xarray.Dataset): with the total loss function and loss per variable for each optim_step
            this doesn't have gradient info, but we could add that
    """

    opt_state = optimizer.init(params) if opt_state is None else opt_state
    mpi_size = emulator.mpi_size
    use_jax_distributed = emulator.use_jax_distributed

    devices = jax.devices()
    sharding = PositionalSharding(devices)
    sharding = sharding.reshape((emulator.num_gpus, 1, 1, 1))
    all_gpus = sharding.replicate()

    if use_jax_distributed:
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

    params = jax.device_put(params, all_gpus)
    state = jax.device_put(state, all_gpus)
    opt_state = jax.device_put(opt_state, all_gpus)
    weights = jax.device_put(weights, all_gpus)
    last_input_channel_mapping = jax.device_put(last_input_channel_mapping, all_gpus)


    # jit optim_step only once
    if not hasattr(optimize, "optim_step_jitted"):
        logging.info("Started jitting optim_step")

        # jitted function
        first_input, first_target = trainer.get_data()
        first_input = jax.device_put(first_input, sharding)
        first_target = jax.device_put(first_target, sharding)

        optimize.optim_step_jitted = jit(optim_step)
        x, *_ = optimize.optim_step_jitted(
            params=params,
            state=state,
            opt_state=opt_state,
            input_batch=first_input,
            target_batch=first_target,
        )
        # refill the queue because we pulled this first item
        trainer.restart()
        block_until_ready(x)


        # Unclear if it's safe to assume whether we'll have the drop_last attr or not
        if not trainer.drop_last:
            raise NotImplementedError
        logging.info("Finished jitting optim_step")


    if not hasattr(optimize, "vloss_jitted"):
        logging.info("Started jitting validation loss")

        first_input, first_target = validator.get_data()

        first_input = jax.device_put(first_input, sharding)
        first_target = jax.device_put(first_target, sharding)

        optimize.vloss_jitted = jit(loss_fn.apply)
        (x, _), _ = optimize.vloss_jitted(
            params=params,
            state=state,
            inputs=first_input,
            targets=first_target,
            rng=PRNGKey(0),
        )
        # refill validation queue since the first one was popped
        validator.restart()
        block_until_ready(x)
        logging.info("Finished jitting validation loss")


    # training
    optim_steps = []
    loss_values = []
    learning_rates = []
    lr = np.nan
    loss_by_channel = []
    n_steps = len(trainer)

    params = jax.device_put(params, all_gpus)
    state = jax.device_put(state, all_gpus)
    opt_state = jax.device_put(opt_state, all_gpus)
    weights = jax.device_put(weights, all_gpus)
    last_input_channel_mapping = jax.device_put(last_input_channel_mapping, all_gpus)

    progress_bar = tqdm(total=n_steps, ncols=140, desc="Processing")
    for k, (input_batch, target_batch) in enumerate(trainer):

        input_batch = jax.device_put(input_batch, sharding)
        target_batch = jax.device_put(target_batch, sharding)

        # call optimize
        params, loss, diagnostics, _, grads = optimize.optim_step_jitted(
            params=params,
            state=state,
            opt_state=opt_state,
            input_batch=input_batch,
            target_batch=target_batch,
        )

        # update progress bar from rank 0
        optim_steps.append(k)
        loss_values.append(loss)
        loss_by_channel.append(diagnostics)
        try:
            lr = opt_state[1].hyperparams["learning_rate"]
        except:
            pass
        learning_rates.append(lr)

        progress_bar.set_description(
            f"loss = {loss:.5f}, qsize = {trainer.data_queue.qsize()}",
        )
        progress_bar.update()

    progress_bar.close()
    trainer.restart()

    # validation
    loss_valid_values = []
    n_steps_valid = len(validator)
    assert (
        n_steps_valid <= n_steps
    ), f"Number of validation steps ({n_steps_valid}) must be less than or equal to the number of training steps ({n_steps})"
    progress_bar = tqdm(total=n_steps_valid, ncols=140, desc="Processing")
    for input_batch, target_batch in validator:

        input_batch = jax.device_put(input_batch, sharding)
        target_batch = jax.device_put(target_batch, sharding)

        (loss_valid, _), _ = optimize.vloss_jitted(
            params=params,
            state=state,
            inputs=input_batch,
            targets=target_batch,
            rng=PRNGKey(0),
        )
        loss_valid_values.append(loss_valid)
        progress_bar.set_description(
            f"validation loss = {loss_valid:.5f}, qsize = {validator.data_queue.qsize()}"
        )
        progress_bar.update()
    loss_valid_avg = np.mean(loss_valid)
    progress_bar.set_description(
        f"validation loss = {loss_valid_avg:.5f}"
    )
    progress_bar.close()
    validator.restart()

    # save losses for each batch
    loss_ds = xr.Dataset()
    loss_fname = os.path.join(emulator.local_store_path, "loss.nc")
    previous_optim_steps = 0
    previous_epochs = 0
    if os.path.exists(loss_fname):
        stored_loss_ds = xr.open_dataset(loss_fname)
        previous_optim_steps = len(stored_loss_ds.optim_step)
        previous_epochs = len(stored_loss_ds["epoch"])

    loss_by_channel = np.vstack(loss_by_channel)
    loss_ds["optim_step"] = [x + previous_optim_steps for x in optim_steps]
    loss_ds["epoch"] = [1 + previous_epochs]
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
    loss_ds["loss_avg"] = xr.DataArray(
        [np.mean(loss_values)],
        coords={"epoch": loss_ds["epoch"]},
        dims=("epoch",),
        attrs={
            "long_name": "average loss function value",
            "description": "averaged over training data once per epoch",
        },
    )
    loss_ds["loss_valid"] = xr.DataArray(
        [loss_valid_avg],
        coords={"epoch": loss_ds["epoch"]},
        dims=("epoch",),
        attrs={
            "long_name": "validation loss function value",
            "description": "averaged over validation data once per epoch",
        },
    )
    loss_ds["learning_rate"] = xr.DataArray(
        learning_rates,
        dims=("optim_step",),
    )
    # this is just so we know what optim steps correspond to what epoch
    loss_ds["epoch_label"] = (1+previous_epochs)*xr.ones_like(loss_ds.optim_step)

    # concatenate losses and store
    if os.path.exists(loss_fname):
        stored_loss_ds = xr.merge([stored_loss_ds, loss_ds])
    else:
        stored_loss_ds = loss_ds
    stored_loss_ds.to_netcdf(loss_fname)

    return params, loss_ds, opt_state
