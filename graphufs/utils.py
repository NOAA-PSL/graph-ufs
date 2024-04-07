import logging
import threading
from graphcast import checkpoint, graphcast


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

    def __init__(self, emulator, mode: str, n_optim_steps: int = None):
        self.data = {}
        self.data_0 = {}
        self.input_thread = None

        self.gen = emulator.get_batches(
            n_optim_steps=n_optim_steps,
            mode=mode,
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
        logging.info("Model description:\n", ckpt.description, "\n")
        logging.info("Model license:\n", ckpt.license, "\n")
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

def add_emulator_arguments(emulator, parser) -> None:
    """Add settings in Emulator class into CLI argument parser

    Args:
        emulator: emulator class
        parser (argparse.ArgumentParser): argument parser
    """
    for k, v in vars(emulator).items():
        if not k.startswith("__"):
            name = "--" + k.replace("_", "-")
            if v is None:
                parser.add_argument(
                    name,
                    dest=k,
                    required=False,
                    type=int,
                    help=f"{k}: default {v}",
                )
            elif isinstance(v, (tuple, list)) and len(v):
                tp = type(v[0])
                parser.add_argument(
                    name,
                    dest=k,
                    required=False,
                    nargs="+",
                    type=tp,
                    help=f"{k}: default {v}",
                )
            elif isinstance(v,dict) and len(v):
                parser.add_argument(
                    name,
                    dest=k,
                    required=False,
                    nargs="+",
                    help=f"{k}: default {v}",
                )
            else:
                parser.add_argument(
                    name,
                    dest=k,
                    required=False,
                    type=type(v),
                    default=v,
                    help=f"{k}: default {v}",
                )

def set_emulator_options(emulator, args) -> None:
    """Set settings in Emulator class from CLI arguments

    Args:
        emulator: emulator class
        args: dictionary of CLI arguments
    """
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            arg_name = arg.replace("-", "_")
            if hasattr(emulator, arg_name):
                stored = getattr(emulator, arg_name)
                if isinstance(stored,dict):
                    value = {s.split(':')[0]: s.split(':')[1] for s in value}
                elif stored is not None:
                    attr_type = type(stored)
                    value = attr_type(value)
                setattr(emulator, arg_name, value)

