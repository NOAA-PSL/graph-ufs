import itertools
import argparse
import logging
import threading
import queue
import xarray as xr
import numpy as np
import concurrent.futures


def get_chunk_data(generator, gen_lock, data: dict, load_chunk: bool, shuffle: bool):
    """Get multiple training batches.

    Args:
        generator: chunk generator object
        gen_lock: generator lock - because generators are not thread-safe
        data (dict): A dict containing the [inputs, targets, forcings]
        load_chunk: load chunk into RAM
        shuffle: shuffle dataset
    """

    # get batches from replay on GCS
    try:
        with gen_lock:
            inputs, targets, forcings, inittimes = next(generator)
    except Exception as e:
        logging.info(e)
        return

    # shuffle here
    if shuffle:
        # shuffle optim_step coord
        permuted_indices = np.random.permutation(inputs["optim_step"].size)
        # shuffle each of inputs/targets/forcings/inittimes
        def shuffle_ds(ds):
            return ds.isel({"optim_step": permuted_indices})

        inputs = shuffle_ds(inputs)
        targets = shuffle_ds(targets)
        forcings = shuffle_ds(forcings)
        if inittimes is not None:
            inittimes = shuffle_ds(inittimes)

    # load into ram unless specified otherwise
    if load_chunk:
        inputs_ = inputs.compute()
        targets_ = targets.compute()
        forcings_ = forcings.compute()
        if inittimes is not None:
            inittimes_ = inittimes.compute()
        else:
            inittimes_ = None
    else:
        inputs_, targets_, forcings_, inittimes_ = (
            inputs, targets, forcings, inittimes
        )

    # update dictionary
    data.update(
        {
            "inputs": inputs_,
            "targets": targets_,
            "forcings": forcings_,
            "inittimes": inittimes_,
        }
    )


class DataGenerator:
    """Data generator class"""

    def __init__(
        self,
        emulator,
        mode: str,
        n_optim_steps: int = None,
        num_workers: int = 1,
        max_queue_size: int = 1,
    ):
        # params for data queue
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.gen_lock = threading.Lock()

        # initialize batch generator
        self.load_chunk = emulator.load_chunk
        self.shuffle = (mode != "testing") and emulator.use_preprocessed
        self.gen = emulator.get_batches(
            n_optim_steps=n_optim_steps,
            mode=mode,
        )

        # create a thread pool of workers for generating data
        if self.num_workers > 0:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            )
            self.futures = [
                self.executor.submit(self.generate) for i in range(self.num_workers)
            ]

    def generate(self):
        """ Data generator function called by workers """
        while not self.stop_event.is_set():
            chunk_data = {}
            get_chunk_data(self.gen, self.gen_lock, chunk_data, self.load_chunk, self.shuffle)
            self.data_queue.put(chunk_data)

    def get_data(self):
        """ Get data from queue """
        if self.num_workers > 0:
            return self.data_queue.get()
        else:
            chunk_data = {}
            get_chunk_data(self.gen, self.gen_lock, chunk_data, self.load_chunk, self.shuffle)
            return chunk_data

    def stop(self):
        """ Stop generator at the end of training"""
        self.stop_event.set()
        while not self.data_queue.empty():
            self.data_queue.get()
            self.data_queue.task_done()


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
                    type=str2bool,
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


def str2bool(v):
    """Convert string to boolean type"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_approximate_memory_usage(data, max_queue_size, num_workers, load_chunk):
    """Gets approximate memory usage of a given configuration.
    Each data generator's memory usage depends on the chunk size, bigger chunks requiring more RAM.
    Since we keep two chunks in RAM by default, the requirement doubles.
    We add 6 Gb to other program memory requirements.

    Args:
        data (list(dict)): a list of training and validation data
        max_queue_size (int): maximum queue size
        num_workers (int): number of worker threads
        load_chunk: load chunk into RAM
    Returns:
        memory usage in GBs
    """
    total = 6
    if load_chunk:
        for d in data:
            chunk_ram = 0
            for k, v in d.items():
                if v is not None:
                    chunk_ram += v.nbytes
            chunk_ram /= 1024 * 1024 * 1024
            total += (max_queue_size + num_workers) * chunk_ram
    return total
