from graphcast import checkpoint, graphcast
from jax import jit
from jax.random import PRNGKey
import threading
from graphufs import run_forward
from ufs2arco.timer import Timer


def get_chunk_data(generator, data: dict):
    """Get multiple training batches.

    Args:
        generator: chunk generator object
        data (List[3]): A list containing the [inputs, targets, forcings]
    """
    localtime = Timer()

    # get batches from replay on GCS
    localtime.start("Preparing Batches from Replay on GCS")

    try:
        inputs, targets, forcings, inittimes = next(generator)
    except StopIteration:
        return
        

    localtime.stop()

    # load into ram
    localtime.start("Loading batches into RAM")

    inputs.load()
    targets.load()
    forcings.load()
    inittimes.load()

    localtime.stop()

    # update dictionary
    data.update(
        {
            "inputs": inputs,
            "targets": targets,
            "forcings": forcings,
            "inittimes": inittimes,
        }
    )


def get_chunk_in_parallel(
    generator, data: dict, data_0: dict, input_thread, chunk_id: int
) -> threading.Thread:
    """Get a chunk of data in parallel with optimization/prediction. This keeps
    two big chunks (data and data_0) in RAM.

    Args:
        generator: chunk generator object
        data (dict): the data being used by optimization/prediction process
        data_0 (dict): the data currently being fetched/processed
        input_thread: the input thread
        chunk_id: chunk number, chunk_id < 0 indicates first chunk
    """
    # make sure input thread finishes before copying data_0 to data
    if chunk_id >= 0:
        input_thread.join()
        for k, v in data_0.items():
            data[k] = v
    # get data
    input_thread = threading.Thread(
        target=get_chunk_data,
        args=(generator, data_0),
    )
    input_thread.start()
    # for first chunk, wait until input thread finishes
    if chunk_id < 0:
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
