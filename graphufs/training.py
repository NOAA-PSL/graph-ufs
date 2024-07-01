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

from graphcast import graphcast
from graphcast.checkpoint import dump
from graphcast.graphcast import GraphCast
from graphcast.casting import Bfloat16Cast
from graphcast.autoregressive import Predictor
from graphcast.xarray_tree import map_structure
from graphcast.normalization import InputsAndResiduals
from graphcast.xarray_jax import unwrap_data
from graphcast import rollout

from tqdm import tqdm

try:
    from mpi4py import MPI
    import mpi4jax
except:
    warnings.warn("Import failed for either mpi4py or mpi4jax.")


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
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


def init_model(emulator, data: dict):
    """Initialize model with random weights.

    Args:
        gufs: emulator class
        data (str): data to be used for initialization?
    """

    @hk.transform_with_state
    def run_forward(emulator, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    init_jitted = jit(run_forward.init)

    inputs=data["inputs"].isel(optim_step=0)
    targets=data["targets"].isel(optim_step=0)
    forcings=data["forcings"].isel(optim_step=0)
    # if we are not loading chunks, load slice as it
    # seems init_jitted requires it
    if not emulator.load_chunk:
        inputs = inputs.load()
        targets = targets.load()
        forcings = forcings.load()

    params, state = init_jitted(
        rng=PRNGKey(emulator.init_rng_seed),
        emulator=emulator,
        inputs=inputs,
        targets_template=targets,
        forcings=forcings,
    )
    return params, state


# Remove the first dimension (device dimension), which is added due to pmap
def remove_first_dim(d):
    return tree_util.tree_map(lambda x: x[0], d)


def aggregate_across_nodes(d):
    def aggregate(d):
        d, _ = mpi4jax.allreduce(d, op=MPI.SUM, comm=MPI.COMM_WORLD)
        d = d / mpi_size
        return d

    return tree_util.tree_map(lambda x: aggregate(x), d)

def add_trees(tree1, tree2):
    def add_inplace(x, y):
        x += y
        return x

    tree_util.tree_map(add_inplace, tree1, tree2)
    return tree1

def optimize(
    params,
    state,
    optimizer,
    emulator,
    training_data,
    validation_data,
    opt_state=None,
    compute_mean_grad=False,
):
    """Optimize the model parameters by running through all optim_steps in data

    Args:
        params (dict): with the initialized model parameters
        state (dict): this is empty, but for now has to be here
        optimizer (Callable, optax.optimizer): see `here <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_
        emulator (ReplayEmulator): the emulator object
        training_data (dict): with data needed for training
        validation_data (dict): with data for validation

    Returns:
        params (dict): optimized model parameters
    """

    opt_state = optimizer.init(params) if opt_state is None else opt_state
    num_gpus = emulator.num_gpus
    mpi_size = emulator.mpi_size
    use_jax_distributed = emulator.use_jax_distributed
    batch_size = emulator.batch_size
    num_batch_splits = emulator.num_batch_splits

    @hk.transform_with_state
    def loss_fn(emulator, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return map_structure(
            lambda x: unwrap_data(x.mean(), require_jax=True), (loss, diagnostics)
        )

    def vloss(params, state, input_batches, target_batches, forcing_batches):
        def ploss(inputs, targets, forcings):
            loss = None
            batch_size_s = batch_size // num_batch_splits
            for i in range(num_batch_splits):

                sl = slice(i * batch_size_s, (i + 1) * batch_size_s)
                ix = inputs.isel(batch=sl)
                tx = targets.isel(batch=sl)
                fx = forcings.isel(batch=sl)

                (loss_i, _), _ = loss_fn.apply(
                    params=params,
                    state=state,
                    emulator=emulator,
                    inputs=ix,
                    targets=tx,
                    forcings=fx,
                    rng=PRNGKey(0),
                )
                if loss is None:
                    loss = (loss_i / num_batch_splits)
                else:
                    loss += (loss_i / num_batch_splits)

            if num_gpus > 1:
                loss = pmean(loss, axis_name="optim_step")
            if (not use_jax_distributed) and (mpi_size > 1):
                loss = aggregate_across_nodes(loss)
            return loss

        if num_gpus > 1:
            loss = pmap(ploss, dim="optim_step")(
                input_batches, target_batches, forcing_batches
            )
            loss = remove_first_dim(loss)
        else:
            loss = ploss(input_batches, target_batches, forcing_batches)

        return loss

    def optim_step(
        params,
        state,
        opt_state,
        emulator,
        input_batches,
        target_batches,
        forcing_batches,
    ):
        """Note that this function has to be definied within optimize so that we do not
        pass optimizer as an argument. Otherwise we get some craazy jax errors"""

        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, PRNGKey(0), emulator, i, t, f
            )
            return loss, (diagnostics, next_state)

        # process one batch per GPU
        def process_batch(inputs, targets, forcings):
            # compute loss and gradient for each split batch and then
            # sum the gradients
            loss = None
            batch_size_s = batch_size // num_batch_splits
            for i in range(num_batch_splits):

                sl = slice(i * batch_size_s, (i + 1) * batch_size_s)
                ix = inputs.isel(batch=sl)
                tx = targets.isel(batch=sl)
                fx = forcings.isel(batch=sl)

                (loss_i, (diagnostics_i, next_state_i)), grads_i = value_and_grad(
                    _aux, has_aux=True
                )(
                    params,
                    state,
                    ix,
                    tx,
                    fx,
                )
                if loss is None:
                    loss = (loss_i / num_batch_splits)
                    grads = grads_i
                    diagnostics = diagnostics_i
                    next_state = next_state_i
                else:
                    loss += (loss_i / num_batch_splits)
                    add_trees(grads, grads_i)
                    add_trees(diagnostics, diagnostics_i)
                    add_trees(next_state, next_state_i)

            # aggregate across local devices
            if num_gpus > 1:
                grads = pmean(grads, axis_name="optim_step")
                loss = pmean(loss, axis_name="optim_step")
                diagnostics = pmean(diagnostics, axis_name="optim_step")
                next_state = pmean(next_state, axis_name="optim_step")

            # manually aggregate results accross nodes. if emulator.use_jax_distributed
            # is turned on, there is no need for this code.
            if (not use_jax_distributed) and (mpi_size > 1):
                loss = aggregate_across_nodes(loss)
                grads = aggregate_across_nodes(grads)
                diagnostics = aggregate_across_nodes(diagnostics)
                next_state = aggregate_across_nodes(next_state)

            return (loss, (diagnostics, next_state)), grads

        if num_gpus > 1:
            # pmap batch processing into multiple GPUs
            (loss, (diagnostics, next_state)), grads = pmap(
                process_batch, dim="optim_step"
            )(input_batches, target_batches, forcing_batches)

            loss = remove_first_dim(loss)
            grads = remove_first_dim(grads)
            diagnostics = remove_first_dim(diagnostics)
            next_state = remove_first_dim(next_state)
        else:
            (loss, (diagnostics, next_state)), grads = process_batch(
                input_batches, target_batches, forcing_batches
            )

        # update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss, diagnostics, opt_state, grads

    # compute number of steps
    n_steps = len(training_data["inputs"]["optim_step"])
    n_steps_valid = len(validation_data["inputs"]["optim_step"])
    if n_steps_valid > n_steps:
        raise ValueError(f"Number of validation steps ({n_steps_valid}) must be less than or equal to the number of training steps ({n_steps})")
    n_steps_valid_inc = ceil(n_steps / n_steps_valid) * num_gpus

    # jit optim_step only once
    if not hasattr(optimize, "optim_step_jitted"):
        logging.info("Started jitting optim_step")

        # jitted function
        optimize.optim_step_jitted = jit(optim_step)
        optimize.vloss_jitted = jit(vloss)

        # warm up step
        n_steps_jit = min(2, n_steps) * num_gpus

        sl = slice(0, num_gpus)
        i_batches = training_data["inputs"].isel(optim_step=sl).copy(deep=True)
        t_batches = training_data["targets"].isel(optim_step=sl).copy(deep=True)
        f_batches = training_data["forcings"].isel(optim_step=sl).copy(deep=True)
        i_batches_valid = validation_data["inputs"].isel(optim_step=sl).copy(deep=True)
        t_batches_valid = validation_data["targets"].isel(optim_step=sl).copy(deep=True)
        f_batches_valid = validation_data["forcings"].isel(optim_step=sl).copy(deep=True)

        optimize.i_batches = i_batches
        optimize.t_batches = t_batches
        optimize.f_batches = f_batches
        optimize.i_batches_valid = i_batches_valid
        optimize.t_batches_valid = t_batches_valid
        optimize.f_batches_valid = f_batches_valid

        x = params
        for k in range(0, n_steps_jit, num_gpus):
            sl = slice(k, k + num_gpus)
            i1_batches = training_data["inputs"].isel(optim_step=sl)
            t1_batches = training_data["targets"].isel(optim_step=sl)
            f1_batches = training_data["forcings"].isel(optim_step=sl)
            for var_name, var in i1_batches.data_vars.items():
                i_batches[var_name] = i_batches[var_name].copy(deep=False, data=var.values)
            for var_name, var in t1_batches.data_vars.items():
                t_batches[var_name] = t_batches[var_name].copy(deep=False, data=var.values)
            for var_name, var in f1_batches.data_vars.items():
                f_batches[var_name] = f_batches[var_name].copy(deep=False, data=var.values)

            x, *_ = optimize.optim_step_jitted(
                params=x,
                state=state,
                opt_state=opt_state,
                emulator=emulator,
                input_batches=i_batches,
                target_batches=t_batches,
                forcing_batches=f_batches,
            )

            if k == 0 or n_steps_valid >= n_steps_jit:
                i1_batches_valid = validation_data["inputs"].isel(optim_step=sl)
                t1_batches_valid = validation_data["targets"].isel(optim_step=sl)
                f1_batches_valid = validation_data["forcings"].isel(optim_step=sl)
                for var_name, var in i1_batches_valid.data_vars.items():
                    i_batches_valid[var_name] = i_batches_valid[var_name].copy(
                        deep=False, data=var.values
                    )
                for var_name, var in t1_batches_valid.data_vars.items():
                    t_batches_valid[var_name] = t_batches_valid[var_name].copy(
                        deep=False, data=var.values
                    )
                for var_name, var in f1_batches_valid.data_vars.items():
                    f_batches_valid[var_name] = f_batches_valid[var_name].copy(
                        deep=False, data=var.values
                    )

                y = optimize.vloss_jitted(
                    params=x,
                    state=state,
                    input_batches=i_batches_valid,
                    target_batches=t_batches_valid,
                    forcing_batches=f_batches_valid,
                )

                block_until_ready(y)

            block_until_ready(x)

        logging.info("Finished jitting optim_step")

    logging.info("Starting iterations")

    optim_steps = []
    loss_values = []
    loss_valid_values = []
    learning_rates = []
    loss_by_var = {k: list() for k in training_data["targets"].data_vars}

    # use the pre-allocated space
    i_batches = optimize.i_batches
    t_batches = optimize.t_batches
    f_batches = optimize.f_batches
    i_batches_valid = optimize.i_batches_valid
    t_batches_valid = optimize.t_batches_valid
    f_batches_valid = optimize.f_batches_valid

    loss_avg = 0
    loss_valid_avg = 0
    mean_grad_avg = 0
    lr = np.nan

    if emulator.mpi_rank == 0:
        progress_bar = tqdm(total=n_steps, ncols=160, desc="Processing")

    for k in range(0, n_steps, num_gpus):
        # When the number of batches is not evenly divisible by num_gpus
        # the last set of batches may not be enough for all gpus. We skip
        # training because aggregation can corrupt final result.
        if k + num_gpus > n_steps:
            break

        # The purpose of the following code is best described as confusing
        # the jix.jat cache system. We start from a deepcopy of slice 0 where the jitting
        # is carried out, and sneakly update its values. If you use xarray update/copy etc
        # the cache system somehow notices, and either becomes slow or messes up the result
        # Copying variable values individually avoids both, fast and produces same results as before

        # training batches
        sl = slice(k, k + num_gpus)

        i1_batches = training_data["inputs"].isel(optim_step=sl)
        t1_batches = training_data["targets"].isel(optim_step=sl)
        f1_batches = training_data["forcings"].isel(optim_step=sl)
        for var_name, var in i1_batches.data_vars.items():
            i_batches[var_name] = i_batches[var_name].copy(deep=False, data=var.values)
        for var_name, var in t1_batches.data_vars.items():
            t_batches[var_name] = t_batches[var_name].copy(deep=False, data=var.values)
        for var_name, var in f1_batches.data_vars.items():
            f_batches[var_name] = f_batches[var_name].copy(deep=False, data=var.values)

        # validation batches
        if (k % n_steps_valid_inc) == 0:
            kv = k // n_steps_valid_inc
            sl = slice(kv, kv + num_gpus)

            i1_batches_valid = validation_data["inputs"].isel(optim_step=sl)
            t1_batches_valid = validation_data["targets"].isel(optim_step=sl)
            f1_batches_valid = validation_data["forcings"].isel(optim_step=sl)

            for var_name, var in i1_batches_valid.data_vars.items():
                i_batches_valid[var_name] = i_batches_valid[var_name].copy(
                    deep=False, data=var.values
                )
            for var_name, var in t1_batches_valid.data_vars.items():
                t_batches_valid[var_name] = t_batches_valid[var_name].copy(
                    deep=False, data=var.values
                )
            for var_name, var in f1_batches_valid.data_vars.items():
                f_batches_valid[var_name] = f_batches_valid[var_name].copy(
                    deep=False, data=var.values
                )
        else:
            prev_loss_valid = loss_valid

        # call optimize
        params, loss, diagnostics, opt_state, grads = optimize.optim_step_jitted(
            params=params,
            state=state,
            opt_state=opt_state,
            emulator=emulator,
            input_batches=i_batches,
            target_batches=t_batches,
            forcing_batches=f_batches,
        )

        # get learning rate
        try:
            lr = opt_state[1].hyperparams["learning_rate"]
        except:
            try:
                lr = opt_state[2].hyperparams["learning_rate"]
            except:
                pass
        learning_rates.append(lr)

        # call validation loss
        if (k % n_steps_valid_inc) == 0:
            loss_valid = optimize.vloss_jitted(
                params=params,
                state=state,
                input_batches=i_batches_valid,
                target_batches=t_batches_valid,
                forcing_batches=f_batches_valid,
            )

        # when we don't compute validation, set to previous value
        else:
            loss_valid = prev_loss_valid

        # average loss per chunk
        loss_avg += loss
        loss_valid_avg += loss_valid

        # update progress bar from rank 0
        if emulator.mpi_rank == 0:
            optim_steps.append(k // num_gpus)
            loss_values.append(loss)
            loss_valid_values.append(loss_valid)
            for key, val in diagnostics.items():
                loss_by_var[key].append(val)

            if compute_mean_grad:
                mean_grad = np.mean(
                    tree_util.tree_flatten(
                        tree_util.tree_map(lambda x: np.abs(x).mean(), grads)
                    )[0]
                )
            else:
                mean_grad = 0
            mean_grad_avg += mean_grad
            description = f"loss = {loss:.5f}, val_loss = {loss_valid:.5f}, lr = {lr:.5e}"
            if compute_mean_grad: description += f"mean(|grad|) = {mean_grad:.8f}"
            progress_bar.set_description(description)
            progress_bar.update(num_gpus)

    # update progress bar one last time with average loss/grad values per chunk
    if emulator.mpi_rank == 0:
        N = len(loss_values)
        loss_avg /= N
        loss_valid_avg /= N
        mean_grad_avg /= N
        lr = learning_rates[-1]
        description = f"loss = {loss_avg:.5f}, val_loss = {loss_valid_avg:.5f}, lr = {lr:0.5e}"
        if compute_mean_grad: description += f"mean(|grad|) = {mean_grad_avg:.8f}"
        progress_bar.set_description(description)
        progress_bar.close()

    # save losses for each batch
    if emulator.mpi_rank == 0 and emulator.store_loss:
        loss_fname = os.path.join(emulator.local_store_path, "loss.nc")
        previous_optim_steps = 0
        if os.path.exists(loss_fname):
            stored_loss_ds = xr.open_dataset(loss_fname)
            previous_optim_steps = len(stored_loss_ds.optim_step)

        loss_ds = xr.Dataset()
        loss_ds["optim_step"] = [x + previous_optim_steps for x in optim_steps]
        loss_ds.attrs["batch_size"] = batch_size
        loss_ds["var_index"] = xr.DataArray(
            np.arange(len(loss_by_var)),
            coords={"var_index": np.arange(len(loss_by_var))},
            dims=("var_index",),
        )
        loss_ds["var_names"] = list(loss_by_var.keys())
        loss_ds["loss"] = xr.DataArray(
            loss_values,
            coords={"optim_step": loss_ds["optim_step"]},
            dims=("optim_step",),
            attrs={"long_name": "loss function value"},
        )
        loss_ds["loss_valid"] = xr.DataArray(
            loss_valid_values,
            coords={"optim_step": loss_ds["optim_step"]},
            dims=("optim_step",),
            attrs={"long_name": "validation loss function value"},
        )
        loss_ds["loss_by_var"] = xr.DataArray(
            np.vstack(list(loss_by_var.values())),
            dims=("var_index", "optim_step"),
        )
        loss_ds["learning_rate"] = xr.DataArray(
            learning_rates,
            coords={"optim_step": loss_ds["optim_step"]},
            dims=("optim_step",),
        )

        # concatenate losses and store
        if os.path.exists(loss_fname):
            stored_loss_ds = xr.concat([stored_loss_ds, loss_ds], dim="optim_step")
        else:
            stored_loss_ds = loss_ds
        stored_loss_ds.to_netcdf(loss_fname)
        logging.info("Updated loss file.")

    return params, opt_state


def predict(
    params,
    state,
    emulator,
    testing_data,
) -> xr.Dataset:
    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_params(fn):
        return partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    apply_jitted = drop_state(with_params(jit(run_forward.apply)))

    # process steps one by one
    all_predictions = []

    n_steps = testing_data["inputs"]["optim_step"].size
    progress_bar = tqdm(total=n_steps, ncols=160, desc="Processing")

    for k in range(0, n_steps):

        i_batches = testing_data["inputs"].isel(optim_step=k).compute()
        t_batches = testing_data["targets"].isel(optim_step=k).compute()
        f_batches = testing_data["forcings"].isel(optim_step=k).compute()
        predictions = rollout.chunked_prediction(
            apply_jitted,
            rng=PRNGKey(0),
            inputs=i_batches,
            targets_template=t_batches,
            forcings=f_batches,
        )

        all_predictions.append(predictions)

        progress_bar.update(1)

    progress_bar.close()

    # combine along "optim_step" dimension
    predictions = xr.concat(all_predictions, dim="optim_step")

    return predictions


def init_devices(emulator):

    # initialize distributed training
    try:
        comm = MPI.COMM_WORLD
        emulator.mpi_rank = comm.Get_rank()
        emulator.mpi_size = comm.Get_size()
    except:
        emulator.mpi_rank = 0
        emulator.mpi_size = 1

    # custom logging handler that filters messages based on mpi rank
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank

        def filter(self, record):
            return self.rank == 0 or not emulator.log_only_rank0

    # Custom formatter to include MPI rank in log messages
    class RankFormatter(logging.Formatter):
        def __init__(self, rank, fmt):
            super().__init__(fmt)
            self.rank = rank

        def format(self, record):
            record.rank = self.rank
            record.relativeCreated = record.relativeCreated // 1000
            return super().format(record)

    # create logger
    rank = emulator.mpi_rank
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addFilter(RankFilter(rank))

    formatter = RankFormatter(
        rank, fmt="[%(relativeCreated)d s] [Rank %(rank)d] [%(levelname)s] %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # turn off absl warnings
    logging.getLogger("absl").setLevel(logging.CRITICAL)

    # Set XLA flags before any JAX library calls for them
    # to take effect.

    # this one is needed for multiple logical devices
    os.environ[
        "XLA_FLAGS"
    ] = f"--xla_force_host_platform_device_count={emulator.num_gpus} "

    if emulator.use_xla_flags:

        # recommened optimization flags
        os.environ["XLA_FLAGS"] += (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=True "
            "--xla_gpu_enable_async_collectives=true "
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
        )

        # nccl flags
        os.environ.update(
            {
                "NCCL_LL128_BUFFSIZE": "-2",
                "NCCL_LL_BUFFSIZE": "-2",
                "NCCL_PROTO": "SIMPLE,LL,LL128",
            }
        )

    # distributed
    if emulator.use_jax_distributed:
        distributed.initialize()

    # are there gpus?
    try:
        N = local_device_count(backend="gpu")
    except:
        N = 0

    # set environment flags
    if N > 0:
        if N > emulator.num_gpus:
            logging.info(
                f"Using fewer gpus than available: {emulator.num_gpus} out of {N}."
            )
            gpu_devices_str = ",".join(str(i) for i in range(emulator.num_gpus))
            os.environ["XLA_FLAGS"] += f" --xla_gpu_devices={gpu_devices_str}"
        else:
            emulator.num_gpus = N
            logging.info(f"Using {N} GPUs.")
    else:
        logging.info(
            f"Using {emulator.num_gpus} logical CPUs. You may want to set OMP_NUM_THREADS to an appropriate value."
        )

    if emulator.mpi_rank == 0:
        logging.info("\n" + print_environment_info(return_string=True))

    logging.info(f"Local devices: {local_device_count()} {local_devices()}")
    logging.info(f"Global devices: {device_count()} {devices()}")
