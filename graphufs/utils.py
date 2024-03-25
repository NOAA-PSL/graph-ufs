from graphcast import checkpoint, graphcast
from jax import jit
from jax.random import PRNGKey
import threading
from graphufs import run_forward


def get_chunk_data(gufs, data: dict, n_batches: int = 4, random_sample: bool = True):
    """Get multiple training batches.

    Args:
        gufs: emulator class
        data (List[3]): A list containing the [inputs, targets, forcings]
        n_batches (int): Number of batches we want to read
    """
    print("Preparing Batches from Replay on GCS")

    inputs, targets, forcings, inittimes = gufs.get_training_batches(
        n_optim_steps=n_batches,
        random_sample=random_sample,
    )

    # load into ram
    inputs.load()
    targets.load()
    forcings.load()
    inittimes.load()

    data.update(
        {
            "inputs": inputs,
            "targets": targets,
            "forcings": forcings,
            "inittimes": inittimes,
        }
    )

    print("Finished preparing batches")


def get_chunk_in_parallel(
    gufs, data: dict, data_0: dict, input_thread, it: int, args: dict
) -> threading.Thread:
    """Get a chunk of data in parallel with optimization/prediction. This keeps
    two big chunks (data and data_0) in RAM.

    Args:
        gufs: emulator class
        data (dict): the data being used by optimization/prediction process
        data_0 (dict): the data currently being fetched/processed
        input_thread: the input thread
        it: chunk number, it < 0 indicates first chunk
        args: CLI arguments
    """
    # make sure input thread finishes before copying data_0 to data
    if it >= 0:
        input_thread.join()
        for k, v in data_0.items():
            data[k] = v
    # don't prefetch a chunk on the last iteration
    if it < args.chunks_per_epoch - 1:
        input_thread = threading.Thread(
            target=get_chunk_data,
            args=(gufs, data_0, args.batches_per_chunk, False) # not args.test), # training needs to be done with unshuffled dataset as well?
        )
        input_thread.start()
    # for first chunk, wait until input thread finishes
    if it < 0:
        input_thread.join()
    return input_thread


def init_model(gufs, data: dict):
    """Initialize model with random weights.

    Args:
        gufs: emulator class
        data (str): data to be used for initialization?
    """
    init_jitted = jit(run_forward.init)
    params, state = init_jitted(
        rng=PRNGKey(gufs.init_rng_seed),
        emulator=gufs,
        inputs=data["inputs"].sel(batch=[0]),
        targets_template=data["targets"].sel(batch=[0]),
        forcings=data["forcings"].sel(batch=[0]),
    )
    return params, state


def load_checkpoint(ckpt_path: str):
    """Load checkpoint.

    Args:
        ckpt_path (str): path to model
    """
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("Model description:\n", ckpt.description, "\n")
    print("Model license:\n", ckpt.license, "\n")
    return params, state


def save_checkpoint(gufs, params, ckpt_path: str) -> None:
    """Load checkpoint.

    Args:
        gufs: emulator class
        params: the parameters (weights) of the model
        ckpt_path (str): path to model
    """
    with open(ckpt_path, "wb") as f:
        ckpt = graphcast.CheckPoint(
            params=params,
            model_config=gufs.model_config,
            task_config=gufs.task_config,
            description="GraphCast model trained on UFS data",
            license="Public domain",
        )
        checkpoint.dump(f, ckpt)
