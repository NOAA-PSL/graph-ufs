import itertools
import logging
import threading
from graphcast import checkpoint, graphcast
import xarray as xr

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

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def get_channel_index(xds, preserved_dims=("batch", "lat", "lon")):
    """For StackedGraphCast, we need to add prediction to last timestamp from the initial conditions.
    To do this, we need a mapping from channel indices to the variables contained in that channel
    with all the collapsed dimensions

    Example:
        >>> inputs, targets, forcings = ...
        >>> get_channel_index(inputs)
        {0: {'varname': 'day_progress_cos', 'time': 0},
         1: {'varname': 'day_progress_cos', 'time': 1},
         2: {'varname': 'day_progress_sin', 'time': 0},
         3: {'varname': 'day_progress_sin', 'time': 1},
         4: {'varname': 'pressfc', 'time': 0},
         5: {'varname': 'pressfc', 'time': 1},
         6: {'varname': 'tmp', 'time': 0, 'level': 0},
         7: {'varname': 'tmp', 'time': 0, 'level': 1},
         8: {'varname': 'tmp', 'time': 0, 'level': 2},
         9: {'varname': 'tmp', 'time': 1, 'level': 0},
         10: {'varname': 'tmp', 'time': 1, 'level': 1},
         11: {'varname': 'tmp', 'time': 1, 'level': 2},
         12: {'varname': 'ugrd10m', 'time': 0},
         13: {'varname': 'ugrd10m', 'time': 1},
         14: {'varname': 'vgrd10m', 'time': 0},
         15: {'varname': 'vgrd10m', 'time': 1},
         16: {'varname': 'year_progress_cos', 'time': 0},
         17: {'varname': 'year_progress_cos', 'time': 1},
         18: {'varname': 'year_progress_sin', 'time': 0},
         19: {'varname': 'year_progress_sin', 'time': 1}}

    Inputs:
        xds (xarray.Dataset): e.g. inputs, targets
        preserved_dims (tuple, optional): same as in graphcast.model_utils.dataset_to_stacked

    Returns:
        mapping (dict): with keys = logical indices 0 -> n_channels-1, and values = a dict
            with "varname" and each dimension, where value corresponds to logical position of that dimension
    """

    mapping = {}
    channel = 0
    for varname in sorted(xds.data_vars):
        stacked_dims = list(set(xds[varname].dims) - set(preserved_dims))
        stacked_dims = sorted(
            stacked_dims,
            key=lambda x: list(xds[varname].dims).index(x),
        )
        stacked_dim_dict = {
            k: list(range(len(xds[k])))
            for k in stacked_dims
        }
        for i, selection in enumerate(product_dict(**stacked_dim_dict), start=channel):
            mapping[i] = {"varname": varname, **selection}
        channel = i+1
    return mapping

def get_last_input_mapping(gds):
    """Use a graphufs.torch.Dataset object to pull some sample data, use that tofigure out mapping between
    expanded variable space and stacked channel space.
    After we get the index mappings from get_channel_index, we need to loop through the
    targets and figure out the channel correpsonding to the last time step for each variable,
    also handling other variables like vertical level

    Inputs:
        gds (graphufs.torch.Dataset): view of the data

    Returns:
        mapper (dict): keys = targets logical index (0 -> n_target_channels-1) and
            values are the logical indices corresponding to input channels
    """

    # get a sample of the data to work with
    xinputs, xtargets, _ = gds.get_xarrays(0)
    inputs_index = get_channel_index(xinputs)
    targets_index = get_channel_index(xtargets)

    # figure out n_time
    n_time = 0
    for ival in inputs_index.values():
        n_time = max(n_time, ival["time"]+1)

    assert n_time > 0, "Could not find time > 0 in inputs_index"

    mapper = {}
    for ti, tval in targets_index.items():
        for ii, ival in inputs_index.items():
            is_match = ival["time"] == n_time - 1

            for k, v in tval.items():
                if k != "time":
                    is_match = is_match and v == ival[k]

            if is_match:
                mapper[ti] = ii
    return mapper

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
            elif isinstance(v,bool):
                parser.add_argument(
                    name,
                    dest=k,
                    action="store_true",
                    required=False,
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

