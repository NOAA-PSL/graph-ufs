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
    logging.warning("Import failed for either mpi4py or mpi4jax.")


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
    params, state = init_jitted(
        rng=PRNGKey(emulator.init_rng_seed),
        emulator=emulator,
        inputs=data["inputs"].sel(optim_step=0),
        targets_template=data["targets"].sel(optim_step=0),
        forcings=data["forcings"].sel(optim_step=0),
    )
    return params, state


def optimize(
    params, state, optimizer, emulator, input_batches, target_batches, forcing_batches
):
    """Optimize the model parameters by running through all optim_steps in data

    Args:
        params (dict): with the initialized model parameters
        state (dict): this is empty, but for now has to be here
        optimizer (Callable, optax.optimizer): see `here <https://optax.readthedocs.io/en/latest/api/optimizers.html>`_
        emulator (ReplayEmulator): the emulator object
        input_batches, training_batches, forcing_batches (xarray.Dataset): with data needed for training

    Returns:
        params (dict): optimized model parameters
        loss_ds (xarray.Dataset): with the total loss function and loss per variable for each optim_step
            this doesn't have gradient info, but we could add that
    """

    opt_state = optimizer.init(params)
    num_gpus = emulator.num_gpus
    mpi_size = emulator.mpi_size
    use_jax_distributed = emulator.use_jax_distributed

    @hk.transform_with_state
    def loss_fn(emulator, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(emulator)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return map_structure(
            lambda x: unwrap_data(x.mean(), require_jax=True), (loss, diagnostics)
        )

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
            (loss, (diagnostics, next_state)), grads = value_and_grad(
                _aux, has_aux=True
            )(
                params,
                state,
                inputs,
                targets,
                forcings,
            )

            # aggregate across local devices
            grads = pmean(grads, axis_name="optim_step")
            loss = pmean(loss, axis_name="optim_step")
            diagnostics = pmean(diagnostics, axis_name="optim_step")
            next_state = pmean(next_state, axis_name="optim_step")

            # manually aggregate results accross nodes. if emulator.use_jax_distributed
            # is turned on, there is no need for this code.
            if (not use_jax_distributed) and (mpi_size > 1):
                # use a helpfer function for grads, which is a dict of dicts,
                # "layer_name" & "weights/bias" being the two keys
                def aggregate_across_nodes(d):
                    if isinstance(d, dict):
                        return {k: aggregate_across_nodes(v) for k, v in d.items()}
                    elif isinstance(d, jnp.ndarray):
                        d, _ = mpi4jax.allreduce(d, op=MPI.SUM, comm=MPI.COMM_WORLD)
                        d = d / mpi_size
                        return d
                    else:
                        return d

                loss = aggregate_across_nodes(loss)
                grads = aggregate_across_nodes(grads)
                diagnostics = aggregate_across_nodes(diagnostics)
                next_state = aggregate_across_nodes(next_state)

            return (loss, (diagnostics, next_state)), grads

        # pmap batch processing into multiple GPUs
        (loss, (diagnostics, next_state)), grads = pmap(
            process_batch, dim="optim_step"
        )(input_batches, target_batches, forcing_batches)

        # Remove the first dimension (device dimension), which is added due to pmap
        def remove_first_dim(d):
            if isinstance(d, dict):
                return {k: remove_first_dim(v) for k, v in d.items()}
            elif isinstance(d, jnp.ndarray):
                return d[0]
            else:
                return d

        loss = remove_first_dim(loss)
        grads = remove_first_dim(grads)
        diagnostics = remove_first_dim(diagnostics)
        next_state = remove_first_dim(next_state)

        # update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, loss, diagnostics, opt_state, grads

    optim_step_jitted = jit(optim_step)

    loss_values = []
    loss_by_var = {k: list() for k in target_batches.data_vars}

    n_steps = input_batches["optim_step"].size

    if emulator.mpi_rank == 0:
        progress_bar = tqdm(total=n_steps, ncols=140, desc="Processing")

    for k in range(0, n_steps, num_gpus):
        # When the number of batches is not evenly divisible by num_gpus
        # the last set of batches may not be enough for all gpus. We skip
        # training because aggregation can corrupt final result. Passing
        # a shorter list of devices may solve it, but the jitted optim
        # function don't like that.
        if k + num_gpus > n_steps:
            # repeat losses, see below for why?
            n_repeat = n_steps - k
            loss_values = loss_values + [loss_values[-1]] * n_repeat
            for k, v in loss_by_var.items():
                loss_by_var[k] = v + [v[-1]] * n_repeat
            break

        # the slice should provide for all gpus
        sl = slice(k, k + num_gpus)

        params, loss, diagnostics, opt_state, grads = optim_step_jitted(
            opt_state=opt_state,
            emulator=emulator,
            input_batches=input_batches.isel(optim_step=sl),
            target_batches=target_batches.isel(optim_step=sl),
            forcing_batches=forcing_batches.isel(optim_step=sl),
            params=params,
            state=state,
        )

        # Since we are doing num_gpus steps per iteration, repeat the losses.
        # Note that pmean() has averaged the losses from different gpus.
        # This is necessary to obtain loss datasets of n_steps size.
        for i in range(num_gpus):
            loss_values.append(loss)
        for key, val in diagnostics.items():
            for i in range(num_gpus):
                loss_by_var[key].append(val)

        if emulator.mpi_rank == 0:
            mean_grad = np.mean(
                tree_util.tree_flatten(
                    tree_util.tree_map(lambda x: np.abs(x).mean(), grads)
                )[0]
            )
            progress_bar.set_description(
                f"[{emulator.mpi_rank}] loss = {loss:.5f}, mean(|grad|) = {mean_grad:.8f}"
            )
            progress_bar.update(num_gpus)

    if emulator.mpi_rank == 0:
        progress_bar.close()

    # save losses for each batch
    loss_ds = xr.Dataset()
    if emulator.mpi_rank == 0:
        loss_fname = os.path.join(emulator.local_store_path, "loss.nc")
        previous_optim_steps = 0
        if os.path.exists(loss_fname):
            stored_loss_ds = xr.open_dataset(loss_fname)
            previous_optim_steps = len(stored_loss_ds.optim_step)

        loss_ds["optim_step"] = input_batches["optim_step"] + previous_optim_steps
        loss_ds.attrs["batch_size"] = len(input_batches["batch"])
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
        loss_ds["loss_by_var"] = xr.DataArray(
            np.vstack(list(loss_by_var.values())),
            dims=("var_index", "optim_step"),
        )

        # concatenate losses and store
        if os.path.exists(loss_fname):
            stored_loss_ds = xr.concat([stored_loss_ds, loss_ds], dim="optim_step")
        else:
            stored_loss_ds = loss_ds
        stored_loss_ds.to_netcdf(loss_fname)

    return params, loss_ds


def predict(
    params,
    state,
    emulator,
    input_batches,
    target_batches,
    forcing_batches,
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

    n_steps = input_batches["optim_step"].size
    progress_bar = tqdm(total=n_steps, ncols=140, desc="Processing")

    for k in range(0, n_steps):

        predictions = rollout.chunked_prediction(
            apply_jitted,
            rng=PRNGKey(0),
            inputs=input_batches.isel(optim_step=k),
            targets_template=target_batches.isel(optim_step=k),
            forcings=forcing_batches.isel(optim_step=k),
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
            return super().format(record)

    # create logger
    rank = emulator.mpi_rank
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addFilter(RankFilter(rank))

    formatter = RankFormatter(
        rank, fmt="[%(relativeCreated)d ms] [Rank %(rank)d] [%(levelname)s] %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # turn off absl warnings
    logging.getLogger("absl").setLevel(logging.CRITICAL)

    # Set XLA flags before any JAX library calls for them
    # to take effect.

    # this one is needed for multiple logical devices
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={emulator.num_gpus} "

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
