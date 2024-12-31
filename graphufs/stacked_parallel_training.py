"""
Same as training.py, but for StackedGraphCast
"""

import logging
from functools import partial
import numpy as np
import xarray as xr
import jax
from jax import (
    jit,
    value_and_grad,
    tree_util,
    block_until_ready,
)
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils
from jax.random import PRNGKey
import optax
import haiku as hk

from tqdm import tqdm

from graphufs.stacked_training import (
    construct_wrapped_graphcast,
    init_model,
    add_trees,
    store_loss,
)

def optimize(
    params,
    state,
    optimizer,
    emulator,
    trainer,
    validator,
    weights,
    last_input_channel_mapping,
    opt_state=None,
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

    devices = jax.devices()[:emulator.num_gpus]
    sharding = PositionalSharding(devices)
    sharding = sharding.reshape((emulator.num_gpus, 1, 1, 1))
    all_gpus = sharding.replicate()

    if use_jax_distributed:
        raise NotImplementedError

    @hk.transform_with_state
    def sample_loss_fn(inputs, targets):
        """Note that this is only valid for a single sample, and if a batch of samples is passed,
        a batch of losses will be returned
        """
        predictor = construct_wrapped_graphcast(emulator, last_input_channel_mapping)
        return predictor.loss(inputs, targets, weights=weights)


    @hk.transform_with_state
    def batch_loss_fn(inputs, targets):
        """Note that this is only valid for a single sample, and if a batch of samples is passed,
        a batch of losses will be returned
        """
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

            (loss, diagnostics), next_state = batch_loss_fn.apply(
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
        # Unclear if it's safe to assume whether we'll have the drop_last attr or not
        if not trainer.drop_last:
            raise NotImplementedError

        logging.info("Started jitting optim_step")

        # jitted function
        optimize.optim_step_jitted = jit(optim_step)

        input_batch, target_batch = trainer.get_data()
        input_batch = jax.device_put(input_batch, sharding)
        target_batch = jax.device_put(target_batch, sharding)
        optimize.input_batch = input_batch
        optimize.target_batch = target_batch

        x, *_ = optimize.optim_step_jitted(
            params=params,
            state=state,
            opt_state=opt_state,
            input_batch=input_batch,
            target_batch=target_batch,
        )

        block_until_ready(x)
        logging.info("Finished jitting optim_step")
        trainer.restart(cancel=True)


    if not hasattr(optimize, "vloss_jitted"):
        logging.info("Started jitting validation loss")

        first_input, first_target = validator.get_data()

        first_input = jax.device_put(first_input, sharding)
        first_target = jax.device_put(first_target, sharding)

        optimize.vloss_jitted = jit(batch_loss_fn.apply)
        (x, _), _ = optimize.vloss_jitted(
            params=params,
            state=state,
            inputs=first_input,
            targets=first_target,
            rng=PRNGKey(0),
        )
        # refill validation queue since the first one was popped
        block_until_ready(x)
        logging.info("Finished jitting validation loss")
        validator.restart(cancel=True)


    # training
    optim_steps = []
    loss_values = []
    learning_rates = []
    gradient_norms = []
    lr = np.nan
    g_norm = np.nan
    loss_by_channel = []
    n_steps = len(trainer)

    params = jax.device_put(params, all_gpus)
    state = jax.device_put(state, all_gpus)
    opt_state = jax.device_put(opt_state, all_gpus)
    weights = jax.device_put(weights, all_gpus)
    last_input_channel_mapping = jax.device_put(last_input_channel_mapping, all_gpus)

    input_batch = optimize.input_batch
    target_batch = optimize.target_batch

    progress_bar = tqdm(total=n_steps, ncols=100, desc="Processing")
    for k, (input_batch, target_batch) in enumerate(trainer):

        input_batch = jax.device_put(input_batch, sharding)
        target_batch = jax.device_put(target_batch, sharding)

        # call optimize
        params, loss, diagnostics, opt_state, grads = optimize.optim_step_jitted(
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
        msg = f"loss = {loss:.5f}"
        try:
            g_norm = opt_state[0].inner_state["g_norm"]
            msg += f", g_norm = {g_norm:1.2e}"
        except:
            pass
        gradient_norms.append(g_norm)

        try:
            lr = opt_state[1].hyperparams["learning_rate"]
            msg += f", LR = {lr:.2e}"
        except:
            pass
        learning_rates.append(lr)

        progress_bar.set_description(msg)
        progress_bar.update()

    progress_bar.close()
    trainer.restart()

    # validation
    loss_valid_values = []
    loss_by_channel_valid = []
    n_steps_valid = len(validator)
    progress_bar = tqdm(total=n_steps_valid, ncols=100, desc="Processing")
    for input_batch, target_batch in validator:

        input_batch = jax.device_put(input_batch, sharding)
        target_batch = jax.device_put(target_batch, sharding)

        (loss_valid, diagnostics_valid), _ = optimize.vloss_jitted(
            params=params,
            state=state,
            inputs=input_batch,
            targets=target_batch,
            rng=PRNGKey(0),
        )
        loss_valid_values.append(loss_valid)
        loss_by_channel_valid.append(diagnostics_valid)
        progress_bar.set_description(
            f"validation loss = {loss_valid:.5f}"
        )
        progress_bar.update()
    loss_valid_avg = np.mean(loss_valid)
    progress_bar.set_description(
        f"validation loss = {loss_valid_avg:.5f}"
    )
    progress_bar.close()
    validator.restart()

    # Compute gradient avg absolute value after each epoch
    mean_grad = np.mean(
        tree_util.tree_flatten(
            tree_util.tree_map(lambda x: np.abs(x).mean(), grads)
        )[0]
    )

    # save losses for each batch
    loss_ds = store_loss(
        emulator=emulator,
        optim_steps=optim_steps,
        loss_values=loss_values,
        loss_by_channel=loss_by_channel,
        loss_valid_avg=loss_valid_avg,
        loss_by_channel_valid=loss_by_channel_valid,
        learning_rates=learning_rates,
        gradient_norms=gradient_norms,
        mean_grad=mean_grad,
    )
    return params, loss_ds, opt_state
