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

    # get batches from replay on GCS
    try:
        inputs, targets, forcings, inittimes = next(generator)
    except StopIteration:
        return

    # load into ram
    inputs.load()
    targets.load()
    forcings.load()
    inittimes.load()

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
    generator, data: dict, data_0: dict, input_thread, first_chunk: bool
) -> threading.Thread:
    """Get a chunk of data in parallel with optimization/prediction. This keeps
    two big chunks (data and data_0) in RAM.

    Args:
        generator: chunk generator object
        data (dict): the data being used by optimization/prediction process
        data_0 (dict): the data currently being fetched/processed
        input_thread: the input thread
        first_chunk: is this the first chunk?
    """
    # make sure input thread finishes before copying data_0 to data
    if not first_chunk:
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
    if first_chunk:
        input_thread.join()
    return input_thread


class DataGenerator:
    """Data generator class"""

    def __init__(self, emulator, mode: str, download_data: bool, n_optim_steps: int = None):
        self.data = {}
        self.data_0 = {}
        self.input_thread = None

        self.gen = emulator.get_batches(
            n_optim_steps=n_optim_steps,
            mode=mode,
            download_data=download_data,
        )
        self.first_chunk = True
        self.generate()

    def generate(self):
        self.input_thread = get_chunk_in_parallel(
            self.gen, self.data, self.data_0, self.input_thread, self.first_chunk
        )
        self.first_chunk = False

    def get_data(self):
        if self.data:
            return self.data;
        else:
            return self.data_0;


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
        inputs=data["inputs"].sel(optim_step=0),
        targets_template=data["targets"].sel(optim_step=0),
        forcings=data["forcings"].sel(optim_step=0),
    )
    return params, state


def load_checkpoint(ckpt_path: str, verbose: bool = False):
    """Load checkpoint.

    Args:
        ckpt_path (str): path to model
        verbose (bool, optional): print metadata about the model
    """
    with open(ckpt_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    if verbose:
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
