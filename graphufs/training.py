"""
These functions are taken from the GraphCast demo:
    https://github.com/google-deepmind/graphcast/blob/main/graphcast_demo.ipynb

Note that in the original demo, the functions:
    - run_forward.init
    - run_forward.apply
    - loss_fn.apply
    - grads_fn
are jitted with wrappers that give and takeaway things like the task and model configs, as well as the state and params
However, since we are passing an Emulator object that contains the task and model configs, and registered it as a pytree, we don't have to worry about the config, so we just have wrappers that pass state and params.
Regarding pytree, see the last few methods and lines of simple_emulatory.py, following this guidance:
    https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree
"""

import os
from functools import partial
import numpy as np
import xarray as xr
from jax import jit, value_and_grad, tree_util
from jax.random import PRNGKey
import optax
import haiku as hk

from graphcast import graphcast
from graphcast.checkpoint import dump
from graphcast.graphcast import GraphCast
from graphcast.casting import Bfloat16Cast
from graphcast.autoregressive import Predictor
from graphcast.xarray_tree import map_structure
from graphcast.normalization import InputsAndResiduals
from graphcast.xarray_jax import unwrap_data


def construct_wrapped_graphcast(emulator):
    """Constructs and wraps the GraphCast Predictor object"""

    predictor = GraphCast(emulator.model_config, emulator.task_config)

    # handle inputs/outputs float32 <-> BFloat16
    # ... and so that this happens after applying
    # normalization to inputs & targets
    predictor = Bfloat16Cast(predictor)
    predictor = InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=emulator.norm["stddiff"],
        mean_by_level=emulator.norm["mean"],
        stddev_by_level=emulator.norm["std"],
    )

    # Wraps everything so the one-step model can produce trajectories
    predictor = Predictor(predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(emulator, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(emulator)
    return predictor(
        inputs,
        targets_template=targets_template,
        forcings=forcings
    )


@hk.transform_with_state
def loss_fn(emulator, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(emulator)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return map_structure(
        lambda x: unwrap_data(x.mean(), require_jax=True),
        (loss, diagnostics))


def grads_fn(params, state, emulator, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, PRNGKey(0), emulator, i, t, f
        )
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = value_and_grad(_aux, has_aux=True)(
        params,
        state,
        inputs,
        targets,
        forcings,
    )
    return loss, diagnostics, next_state, grads


def optimize(params, state, optimizer, emulator, input_batches, target_batches, forcing_batches, store_results=True, description="", license="", verbose=False):
    """Optimize the model parameters by running through all optim_steps in data

    Args:
        params (dict): with the initialized model parameters
        state (dict): this is empty, but for now has to be here
        optimizer (Callable, optax.optimizer): see `here <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_
        emulator (ReplayEmulator): the emulator object
        input_batches, training_batches, forcing_batches (xarray.Dataset): with data needed for training
        store_results (bool, optional): if True, store the loss values to netcdf and model checkpoint (optimized weights) using graphcast's storage routines
        description, license (str, optional): information to store with the model checkpoint when writing out optimized weights, irrelevant if store_results is False
        verbose (bool, optional): if True, print loss and mean(|gradient|) at each step

    Returns:
        params (dict): optimized model parameters
        loss_ds (xarray.Dataset): with the total loss function and loss per variable for each optim_step
            this doesn't have gradient info, but we could add that
    """

    opt_state = optimizer.init(params)

    def optim_step(params, state, opt_state, emulator, inputs, targets, forcings):
        """Note that this function has to be definied within optimize so that we do not
        pass optimizer as an argument. Otherwise we get some craazy jax errors"""

        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, PRNGKey(0), emulator, i, t, f
            )
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = value_and_grad(_aux, has_aux=True)(
            params,
            state,
            inputs,
            targets,
            forcings,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss, diagnostics, opt_state, grads

    optim_step_jitted = jit( optim_step )

    loss_values = []
    loss_by_var = {k: list() for k in target_batches.data_vars}

    for k in input_batches["optim_step"].values:

        params, loss, diagnostics, opt_state, grads = optim_step_jitted(
            opt_state=opt_state,
            emulator=emulator,
            inputs=input_batches.sel(optim_step=k),
            targets=target_batches.sel(optim_step=k),
            forcings=forcing_batches.sel(optim_step=k),
            params=params,
            state=state,
        )
        loss_values.append(loss)
        for key, val in diagnostics.items():
            loss_by_var[key].append(val)

        if verbose:
            mean_grad = np.mean(tree_util.tree_flatten(tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
            print(f"Step = {k+1}, loss = {loss}, mean(|grad|) = {mean_grad}")
            print("diagnostics: ")
            print(diagnostics)
            print()

    loss_ds = xr.Dataset()
    loss_ds["optim_step"] = input_batches["optim_step"]
    loss_ds.attrs["batch_size"] = len(input_batches["batch"])
    loss_ds["var_index"] = xr.DataArray(
        np.arange(len(loss_by_var)),
        coords={"var_index": np.arange(len(loss_by_var))},
        dims=("var_index",),
    )
    loss_ds["var_names"] = xr.DataArray(
        list(loss_by_var.keys()),
        dims=("var_index",),
    )
    loss_ds["loss"] = xr.DataArray(
        loss_values,
        coords={"optim_step": loss_ds["optim_step"]},
        dims=("optim_step",),
        attrs={"long_name": "loss function value"},
    )
    loss_ds["loss_by_var"] = xr.DataArray(
        np.vstack(list(loss_by_var.values())),
        dims=("var_index", "optim_step"),
    )

    if store_results:
        loss_fname = os.path.join(emulator.local_store_path, "loss.nc")
        print(f"Storing loss function values at: {loss_fname}")
        loss_ds.to_netcdf(os.path.join(emulator.local_store_path, "loss.nc"))

        # store parameters
        ckpt = graphcast.CheckPoint(
            params=params,
            model_config=emulator.model_config,
            task_config=emulator.task_config,
            description=description,
            license=license,
        )
        ckpt_fname = os.path.join(emulator.local_store_path, "graphufs.ckpt")
        print(f"Storing model checkpoint at: {ckpt_fname}")
        with open(ckpt_fname, "wb") as f:
            dump(f, ckpt)

    return params, loss_ds
