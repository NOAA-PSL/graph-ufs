import os
import logging
import argparse
import math
import yaml
import warnings
import random
import itertools
import dataclasses
import numpy as np
import pandas as pd
import xarray as xr
from jax import tree_util

from ufs2arco.regrid.ufsregridder import UFSRegridder
from graphcast import checkpoint
from graphcast.graphcast import ModelConfig, TaskConfig, CheckPoint
from graphcast.data_utils import extract_inputs_targets_forcings
from graphcast.model_utils import dataset_to_stacked
from graphcast.losses import normalized_level_weights, normalized_latitude_weights

from .utils import (
    get_channel_index, get_last_input_mapping,
    add_emulator_arguments, set_emulator_options
)


class ReplayEmulator:
    """An emulator based on UFS Replay data. This manages all model configuration settings and normalization fields. Currently it is designed to be inherited for a specific use-case, and this could easily be generalized to read in settings via a configuration file (yaml, json, etc). Be sure to register any inherited class as a pytree for it to work with JAX.

    Example:
        see graph-ufs/prototypes/p0/simple_emulator.py for example usage
    """

    data_url = ""
    norm_urls = {
        "mean": "",
        "std": "",
        "stddiff": "",
    }
    norm = dict()
    stacked_norm = dict()
    wb2_obs_url = ""
    local_store_path = None     # directory where zarr file, model weights etc are stored
    no_cache_data = None        # don't cache or use zarr dataset downloaded from GCS on disk

    # these could be moved to a yaml file later
    # task config options
    input_variables = tuple()
    target_variables = tuple()
    forcing_variables = tuple()
    all_variables = tuple()     # this is created in __init__
    pressure_levels = tuple()
    levels = list()             # created in __init__, has exact pfull level values
    latitude = tuple()
    longitude = tuple()

    # time related
    delta_t = None              # the model time step
    input_duration = None       # time covered by initial condition(s)
    target_lead_time = None     # how long the forecast is, i.e., when we compare to data
    training_dates = tuple()    # bounds of training data (inclusive)
    testing_dates = tuple()     # bounds of testing data (inclusive)
    validation_dates = tuple()  # bounds of validation data (inclusive)

    # training protocol
    batch_size = None               # number of forecasts averaged over in loss per optim_step
    num_epochs = None               # number of epochs
    chunks_per_epoch = None         # number of chunks per epoch
    steps_per_chunk = None          # number of steps to train for in each chunk
    checkpoint_chunks = None        # save model after this many chunks are processed
    max_queue_size = None           # number of chunks in queue of data generators
    num_workers = None              # number of worker threads for data generators
    no_load_chunk = None            # don't load chunk into RAM, has the lowest memory usage if true
    store_loss = None               # store loss in a netcdf file

    # others
    num_gpus = None                 # number of GPUs to use for training
    log_only_rank0 = None           # log only messages from rank 0
    use_jax_distributed = None      # Use jax's distributed mechanism, no need for manula mpi4jax calls
    use_xla_flags = None            # Use recommended flags for XLA and NCCL https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
    dask_threads = None             # number of threads to use for dask

    # model config options
    resolution = None               # nominal spatial resolution
    mesh_size = None                # how many refinements to do on the multi-mesh
    latent_size = None              # how many latent features to include in MLPs
    gnn_msg_steps = None            # how many graph network message passing steps to do
    hidden_layers = None            # number of hidden layers for each MLP
    radius_query_fraction_edge_length = None    # Scalar that will be multiplied by the length of the longest edge of
                                                # the finest mesh to define the radius of connectivity to use in the
                                                # Grid2Mesh graph. Reasonable values are between 0.6 and 1. 0.6 reduces
                                                # the number of grid points feeding into multiple mesh nodes and therefore
                                                # reduces edge count and memory use, but gives better predictions.
    mesh2grid_edge_normalization_factor = 0.6180338738074472 # Allows explicitly controlling edge normalization for mesh2grid edges.
                                                             # If None, defaults to max edge length.This supports using pre-trained
                                                             # model weights with a different graph structure to what it was trained on.
    mesh2grid_edge_normalization_factor = None

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_latitude = True
    weight_loss_per_level = True
    loss_weights_per_variable = {
        "tmp2m"         : 1.0,
        "ugrd10m"       : 0.1,
        "vgrd10m"       : 0.1,
        "pressfc"       : 0.1,
        "prateb_ave"    : 0.1,
    }

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = None
    init_rng_seed = None
    training_batch_rng_seed = None # used to randomize the training batches

    # data chunking options


    # for stacked graphcast
    last_input_channel_mapping = None

    def __init__(self, mpi_rank=None, mpi_size=None):

        if self.local_store_path is None:
            warnings.warn("ReplayEmulator.__init__: no local_store_path set, data will always be accessed remotely. Proceed with patience.")

        if any(x not in self.input_variables for x in self.target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict target variables that are not also inputs")

        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size

        pfull = self._get_replay_vertical_levels()
        latitude, longitude = self._get_replay_grid(self.resolution)
        self.latitude = tuple(float(x) for x in latitude)
        self.longitude = tuple(float(x) for x in longitude)
        self.levels = list(
            pfull.sel(
                pfull=list(self.pressure_levels),
                method="nearest",
            ).values
        )
        self.model_config = ModelConfig(
            resolution=self.resolution,
            mesh_size=self.mesh_size,
            latent_size=self.latent_size,
            gnn_msg_steps=self.gnn_msg_steps,
            hidden_layers=self.hidden_layers,
            radius_query_fraction_edge_length=self.radius_query_fraction_edge_length,
            mesh2grid_edge_normalization_factor=self.mesh2grid_edge_normalization_factor,
        )
        # try/except logic to support original graphcast.graphcast.TaskConfig
        # since I couldn't get inspect.getfullargspec to work
        try:
            self.task_config = TaskConfig(
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                forcing_variables=self.forcing_variables,
                pressure_levels=tuple(self.levels),
                input_duration=self.input_duration,
                longitude=self.longitude,
                latitude=self.latitude,
            )
        except ValueError:
            self.task_config = TaskConfig(
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                forcing_variables=self.forcing_variables,
                pressure_levels=tuple(self.levels),
                input_duration=self.input_duration,
            )


        self.all_variables = tuple(set(
            self.input_variables + self.target_variables + self.forcing_variables
        ))

        # convert some types
        self.delta_t = pd.Timedelta(self.delta_t)
        self.input_duration = pd.Timedelta(self.input_duration)

        # set normalization here so that we can jit compile with this class
        self.set_normalization()
        self.set_stacked_normalization()


    @property
    def time_per_forecast(self):
        return self.target_lead_time + self.input_duration

    @property
    def n_input(self):
        """Number of steps that initial condition(s) cover"""
        return self.input_duration // self.delta_t

    @property
    def n_forecast(self):
        """Number of steps covered by a single forecast, including initial condition(s)"""
        return self.time_per_forecast // self.delta_t

    @property
    def n_target(self):
        """Number of steps in the target, doesn't include initial condition(s)"""
        return self.target_lead_time // self.delta_t

    @property
    def extract_kwargs(self):
        kw = {k: v for k, v in dataclasses.asdict(self.task_config).items() if k not in ("latitude", "longitude")}
        kw["target_lead_times"] = self.target_lead_time
        return kw

    @property
    def local_data_path(self):
        return os.path.join(self.local_store_path, "data.zarr")

    @property
    def checkpoint_dir(self):
        return os.path.join(self.local_store_path, "models")


    def open_dataset(self, **kwargs):
        xds = xr.open_zarr(self.data_url, storage_options={"token": "anon"}, **kwargs)
        return xds


    def get_time(self, mode):
        # choose dates based on mode
        if mode == "training":
            start = self.training_dates[ 0]
            end   = self.training_dates[-1]
        elif mode == "testing":
            start = self.testing_dates[ 0]
            end   = self.testing_dates[-1]
        elif mode == "validation":
            start = self.validation_dates[ 0]
            end   = self.validation_dates[-1]
        else:
            raise ValueError("Unknown mode: make sure it is either training/testing/validation")

        # build time vector based on the model, not the data
        start = pd.Timestamp(start)
        end   = pd.Timestamp(end)
        time = pd.date_range(
            start=start,
            end=end,
            freq=self.delta_t,
            inclusive="both",
        )
        return time


    def subsample_dataset(self, xds, new_time=None):
        """Get the subset of the data that we want in terms of time, vertical levels, and variables

        Args:
            xds (xarray.Dataset): with replay data
            new_time (pandas.Daterange or similar, optional): time vector to select from the dataset

        Returns:
            newds (xarray.Dataset): subsampled/subset that we care about
        """

        # select our vertical levels
        xds = xds.sel(pfull=self.levels)

        # only grab variables we care about
        myvars = list(x for x in self.all_variables if x in xds)
        xds = xds[myvars]

        if new_time is not None:
            xds = xds.sel(time=new_time)

        return xds


    def preprocess(self, xds, batch_index=None):
        """Prepare a single batch for GraphCast

        Args:
            xds (xarray.Dataset): with replay data
            batch_index (int, optional): the index of this batch
        Returns:
            bds (xarray.Dataset): this batch of data
        """
        bds = xds
        bds["time"] = (bds.datetime - bds.datetime[0])
        bds = bds.swap_dims({"datetime": "time"}).reset_coords()
        if batch_index is not None:
            bds = bds.expand_dims({
                "batch": [batch_index],
            })
        bds = bds.set_coords(["datetime"])
        return bds

    def get_the_data(
        self,
        all_new_time=None,
        mode="training",
    ):
        """Handle the local storage, caching, missing dates stuff here, pass to get_batches to batch it up"""

        all_new_time = all_new_time if all_new_time is not None else self.get_time(mode=mode)
        # download only missing dates and write them to disk
        if self.no_cache_data or not os.path.exists(self.local_data_path):
            logging.info(f"Downloading missing {mode} data for {len(all_new_time)} time stamps.")
            xds = xr.open_zarr(self.data_url, storage_options={"token": "anon"})
            all_xds = self.subsample_dataset(xds, new_time=all_new_time)
            if not self.no_cache_data:
                all_xds.to_zarr(self.local_data_path)
                all_xds.close()
                all_xds = xr.open_zarr(self.local_data_path)
        else:
            # figure out missing dates
            xds_on_disk = xr.open_zarr(self.local_data_path)
            missing_dates = set(all_new_time.values) - set(xds_on_disk.time.values)
            if len(missing_dates) > 0:
                xds_on_disk.close()
                logging.info(f"Downloading missing {mode} data for {len(missing_dates)} time stamps.")
                # download and write missing dates to disk

                missing_xds = self.open_dataset()
                missing_xds = self.subsample_dataset(missing_xds, new_time=list(missing_dates))
                missing_xds.to_zarr(self.local_data_path, append_dim="time")
                # now that the data on disk is complete, reopen the dataset from disk
                all_xds = xr.open_zarr(self.local_data_path)
            else:
                all_xds = xds_on_disk

        return all_xds

    def get_batches(
        self,
        n_optim_steps=None,
        drop_cftime=True,
        mode="training",
        allow_overlapped_chunks=None,
    ):
        """Get a dataset with all the batches of data necessary for training

        Note:
            Here we're using target_lead_time as a single value, see graphcast.data_utils.extract ... where it could be multi valued. However, since we are using it to compute the total forecast time per batch soit seems more straightforward as a scalar.

        Args:
            n_optim_steps (int, optional): number of training batches to grab ... number of times we will update the parameters during optimization. If not specified, use as many as are available based on the available training data.
            drop_cftime (bool, optional): may be useful for debugging
            mode (str, optional): can be either "training", "validation" or "testing"
            allow_overlapped_chunks (bool, optional): overlapp chunks
        Returns:
            inputs, targets, forcings (xarray.Dataset): with new dimension "batch"
                and appropriate fields for each dataset, based on the variables in :attr:`task_config`
        """
        all_new_time = self.get_time(mode=mode)

        # split the dataset across nodes
        # make sure work is _exactly_ equally distirubuted to prevent hangs
        # when the number of time stamps is not evenly divisible by the number of ranks,
        # we discard whatever data is left over. Not a problem because parallelization is not done for testing.
        if self.mpi_size > 1:
            mpi_chunk_size = len(all_new_time) // self.mpi_size
            start = self.mpi_rank * mpi_chunk_size
            end = (self.mpi_rank + 1) * mpi_chunk_size
            all_new_time = all_new_time[start:end]
            logging.info(f"Data for {mode} MPI rank {self.mpi_rank}: {all_new_time[0]} to {all_new_time[-1]} : {len(all_new_time)} time stamps.")


        all_xds = self.get_the_data(all_new_time=all_new_time, mode=mode)
        # split dataset into chunks
        n_chunks = self.chunks_per_epoch
        chunk_size = len(all_new_time) // n_chunks
        all_new_time_chunks = []

        # overlap chunks by lead time + input duration
        overlap_step = (self.target_lead_time + self.input_duration) // delta_t if allow_overlapped_chunks else 0
        for i in range(n_chunks):
            if i == n_chunks - 1:
                all_new_time_chunks.append(all_new_time[i * chunk_size:len(all_new_time)])
            else:
                all_new_time_chunks.append(all_new_time[i * chunk_size:(i + 1) * chunk_size + overlap_step])

        # print chunk boundaries
        message = f"Chunks for {mode}: {len(all_new_time_chunks)}"
        for chunk_id, new_time in enumerate(all_new_time_chunks):
            message += f"\nChunk {chunk_id+1}: {new_time[0]} to {new_time[-1]} : {len(new_time)} time stamps"
        logging.info(message)

        # warnings before we get started
        if pd.Timedelta(self.target_lead_time) > self.delta_t:
            warnings.warn("ReplayEmulator.get_training_batches: need to rework this to pull targets for all steps at delta_t intervals between initial conditions and target_lead times, at least in part because we need the forcings at each delta_t time step, and the data extraction code only pulls this at each specified target_lead_time")

        # loop forever
        while True:

            # shuffle chunks
            if mode != "testing":
                random.shuffle(all_new_time_chunks)

            # iterate over all chunks
            for chunk_id, new_time in enumerate(all_new_time_chunks):

                # chunk start and end times
                start = new_time[0]
                end = new_time[-1]

                # figure out duration of IC(s), forecast, all of training
                data_duration = end - start

                n_max_forecasts = (data_duration - self.input_duration) // self.delta_t
                if n_max_forecasts <= 0:
                    raise ValueError(f"n_max_forecasts for {mode} is {n_max_forecasts}")

                n_max_optim_steps = math.ceil(n_max_forecasts / self.batch_size)
                n_optim_steps = n_max_optim_steps if n_optim_steps is None else n_optim_steps
                n_forecasts = n_optim_steps * self.batch_size
                n_forecasts = min(n_forecasts, n_max_forecasts)

                # note that this max can be violated if we sample with replacement ...
                # but I'd rather just work with epochs and use all the data
                if n_optim_steps > n_max_optim_steps:
                    n_optim_steps = n_max_optim_steps
                    warnings.warn(f"There's less data than the number of batches requested, reducing n_optim_steps to {n_optim_steps}")

                if self.steps_per_chunk is None and mode != "validation":
                    self.steps_per_chunk = n_optim_steps

                # create a new time vector with desired delta_t
                # this has to end such that we can pull an entire forecast from the training data
                all_initial_times = pd.date_range(
                    start=start,
                    end=end - self.time_per_forecast,
                    freq=self.delta_t,
                    inclusive="both",
                )

                # randomly sample without replacement
                # note that GraphCast samples with replacement
                if mode != "testing":
                    rstate = np.random.RandomState(seed=self.training_batch_rng_seed)
                    forecast_initial_times = rstate.choice(
                        all_initial_times,
                        size=(n_forecasts,),
                        replace=False
                    )
                else:
                    forecast_initial_times = all_initial_times[:n_forecasts]

                # subsample in time, grab variables and vertical levels we want
                xds = self.subsample_dataset(all_xds, new_time=new_time)
                xds = xds.rename({
                    "pfull": "level",
                    "grid_xt": "lon",
                    "grid_yt": "lat",
                    "time": "datetime",
                    })
                xds = xds.drop(["cftime", "ftime"])
                xds.load()

                # iterate through batches
                inputs = []
                targets = []
                forcings = []
                inittimes = []
                for i, (k, b) in enumerate(
                    itertools.product(range(n_optim_steps), range(self.batch_size))
                ):

                    if i >= n_max_forecasts:
                        # If the last batch won't be full, take values from the previous batch and complete it.
                        # Do this only for training, because in testing mode this process will mess up the output.
                        # For testing, we will cleanup after prediction using dropna()
                        def copy_values(ds_list):
                            s = len(ds_list)
                            if s >= self.batch_size:
                                mds = ds_list[-self.batch_size].copy()
                            else:
                                bs = random.randint(0,s-1)
                                mds = ds_list[bs].copy()
                                mds["batch"] = [b]
                            mds["optim_step"] = [k]
                            ds_list.append(mds)
                        if mode != "testing":
                            copy_values(inputs)
                            copy_values(targets)
                            copy_values(forcings)
                        continue

                    timestamps_in_this_forecast = pd.date_range(
                        start=forecast_initial_times[i],
                        end=forecast_initial_times[i]+self.time_per_forecast,
                        freq=self.delta_t,
                        inclusive="both",
                    )
                    batch = self.preprocess(
                        xds.sel(datetime=timestamps_in_this_forecast),
                        batch_index=b,
                    )

                    this_input, this_target, this_forcing = extract_inputs_targets_forcings(
                        batch,
                        **self.extract_kwargs,
                    )

                    # note that the optim_step dim has to be added after the extract_inputs_targets_forcings call
                    inputs.append(this_input.expand_dims({"optim_step": [k]}))
                    targets.append(this_target.expand_dims({"optim_step": [k]}))
                    forcings.append(this_forcing.expand_dims({"optim_step": [k]}))

                    if mode == "testing":
                        # fix this later for batch_size != 1
                        this_inittimes = batch.datetime.isel(time=0)
                        this_inittimes = this_inittimes.to_dataset(name="inittimes")
                        inittimes.append(this_inittimes.expand_dims({"optim_step": [k]}))

                del xds
                inputs = xr.combine_by_coords(inputs)
                targets = xr.combine_by_coords(targets)
                forcings = xr.combine_by_coords(forcings)
                if mode == "testing":
                    inittimes = xr.combine_by_coords(inittimes)
                else:
                    inittimes = None
                yield inputs, targets, forcings, inittimes


    def set_normalization(self, **kwargs):
        """Load the normalization fields into memory

        Returns:
            mean_by_level, stddev_by_level, diffs_stddev_by_level (xarray.Dataset): with normalization fields
        """

        def open_normalization(component, **kwargs):

            # try to read locally first
            local_path = os.path.join(
                self.local_store_path,
                "normalization",
                os.path.basename(self.norm_urls[component]),
            )

            if os.path.isdir(local_path):
                xds = xr.open_zarr(local_path)
                xds = xds.load()
                foundit = True

            else:
                xds = xr.open_zarr(self.norm_urls[component], **kwargs)
                myvars = list(x for x in self.all_variables if x in xds)
                xds = xds[myvars]
                xds = xds.sel(pfull=self.levels)
                xds = xds.load()
                xds = xds.rename({"pfull": "level"})
                xds.to_zarr(local_path)
            return xds

        for key in ["mean", "std", "stddiff"]:
            self.norm[key] = open_normalization(key)

    def set_stacked_normalization(self):

        assert len(self.norm["mean"]) > 0, "normalization not set, call Emulator.set_normalization()"

        def open_normalization(component):

            # try to read locally first
            inputs_path = os.path.join(
                self.local_store_path,
                "stacked-normalization",
                "inputs",
                os.path.basename(self.norm_urls[component]),
            )
            targets_path = os.path.join(
                self.local_store_path,
                "stacked-normalization",
                "targets",
                os.path.basename(self.norm_urls[component]),
            )

            if os.path.isdir(inputs_path) and os.path.isdir(targets_path):
                inputs = xr.open_zarr(inputs_path)
                inputs = inputs["inputs"].load()
                targets = xr.open_zarr(targets_path)
                targets = targets["targets"].load()

            else:
                inputs, targets = self.normalization_to_stacked(self.norm[component], preserved_dims=tuple())
                ds = xr.Dataset()
                inputs = inputs.load()
                targets = targets.load()
                inputs.to_dataset(name="inputs").to_zarr(inputs_path)
                targets.to_dataset(name="targets").to_zarr(targets_path)
            return inputs.data, targets.data

        for key in self.norm.keys():
            self.stacked_norm[key] = dict()
            input_norms, target_norms = open_normalization(key)
            self.stacked_norm[key] = {"inputs": input_norms, "targets": target_norms}


    def normalization_to_stacked(self, xds, **kwargs):
        """
        kwargs passed to graphcast.model_utils.dataset_to_stacked
        """

        def stackit(xds, varnames, n_time, **kwargs):
            norms = xds[[x for x in varnames if x in xds]]
            # do this to replicate across time dimension
            norms = xr.concat(
                [norms.copy() for _ in range(n_time)],
                dim="time",
            )
            dimorder = ("batch", "time", "level", "lat", "lon")
            dimorder = tuple(x for x in dimorder if x in norms.dims)
            norms = norms.transpose(*dimorder)
            return dataset_to_stacked(norms, **kwargs)

        input_norms = stackit(xds, self.input_variables, n_time=self.n_input, **kwargs)
        forcing_norms = stackit(xds, self.forcing_variables, n_time=self.n_target, **kwargs)
        target_norms = stackit(xds, self.target_variables, n_time=self.n_target, **kwargs)
        input_norms = xr.concat(
            [
                input_norms,
                forcing_norms,
            ],
            dim="channels",
        )
        return input_norms, target_norms


    def calc_loss_weights(self, gds):

        _, xtargets, _ = gds.get_xarrays(0)
        _, targets = gds[0]

        if targets.ndim == 3:
            weights = np.ones_like(targets)
        else:
            weights = np.ones_like(targets[0])
            weights = weights[None]

        # 1. compute latitude weighting
        if self.weight_loss_per_latitude:
            lat_weights = normalized_latitude_weights(xtargets)
            lat_weights = lat_weights.data[...,None][...,None]

            weights *= lat_weights

        # 2. compute per variable weighting
        target_idx = get_channel_index(xtargets)
        for ichannel in range(targets.shape[-1]):
            varname = target_idx[ichannel]["varname"]
            if varname in self.loss_weights_per_variable:
                weights[..., ichannel] *= self.loss_weights_per_variable[varname]

        # 3. compute per level weighting
        if self.weight_loss_per_level:
            level_weights = normalized_level_weights(xtargets)
            for ichannel in range(targets.shape[-1]):
                if "level" in target_idx[ichannel].keys():
                    ilevel = target_idx[ichannel]["level"]
                    weights[..., ichannel] *= level_weights.isel(level=ilevel).data

        # do we need to put this on the device(s)?
        return weights



    @staticmethod
    def _get_replay_vertical_levels():
        pfull_path = os.path.join(os.path.dirname(__file__), "replay_vertical_levels.yaml")
        with open(pfull_path, "r") as f:
            pfull = yaml.safe_load(f)["pfull"]
        return xr.DataArray(pfull, coords={"pfull": pfull}, dims="pfull")

    def _get_replay_grid(self, resolution: int | float):
        if int(resolution) == 1:
            if "0.25-degree-subsampled" in self.data_url:
                lats, lons = UFSRegridder.compute_gaussian_grid(768, 1536)
                lats = lats[::4]
                lons = lons[::4]
            else:
                lats, lons = UFSRegridder.compute_gaussian_grid(192, 384)

        elif int(resolution*100) == 25:
            lats, lons = UFSRegridder.compute_gaussian_grid(768, 1536)
        else:
            raise NotImplementedError("Resolution not available in Replay data")
        lats = lats[::-1]
        return lats, lons


    @classmethod
    def from_parser(cls):
        """Parse CLI arguments."""

        # parse arguments
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(
            "--test",
            dest="test",
            action="store_true",
            required=False,
            help="Test model specified with --id. Otherwise train model.",
        )
        parser.add_argument(
            "--id",
            "-i",
            dest="id",
            required=False,
            type=int,
            default=-1,
            help="ID of neural networks to resume training/testing from.",
        )

        # add arguments from emulator
        add_emulator_arguments(cls, parser)

        # parse CLI args
        args = parser.parse_args()

        # override options in emulator class by those from CLI
        set_emulator_options(cls, args)
        emulator = cls()

        return emulator, args

    def save_checkpoint(self, params, id) -> None:
        """Store checkpoint.

        Args:
            params: the parameters (weights) of the model
            ckpt_path (str): path to model
            id (int): the stored iteration ID
        """

        ckpt_path = os.path.join(self.checkpoint_dir, f"model_{id}.npz")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        with open(ckpt_path, "wb") as f:
            ckpt = CheckPoint(
                params=params,
                model_config=self.model_config,
                task_config=self.task_config,
                description=f"GraphCast model trained on UFS Replay data, ID = {id}",
                license="Public domain",
            )
            checkpoint.dump(f, ckpt)

    def checkpoint_exists(self, id):
        ckpt_path = os.path.join(self.checkpoint_dir, f"model_{id}.npz")
        return os.path.exists(ckpt_path)

    def load_checkpoint(self, id, verbose: bool = False):
        """Load checkpoint.

        Args:
            id (int): integer ID num to load
            verbose (bool, optional): print metadata about the model
        """
        ckpt_path = os.path.join(self.checkpoint_dir, f"model_{id}.npz")

        with open(ckpt_path, "rb") as f:
            ckpt = checkpoint.load(f, CheckPoint)
        params = ckpt.params
        state = {}
        model_config = ckpt.model_config
        task_config = ckpt.task_config
        if verbose:
            logging.info("Model description:\n", ckpt.description, "\n")
            logging.info("Model license:\n", ckpt.license, "\n")
        return params, state


    def _tree_flatten(self):
        """Pack up everything needed to remake this object.
        Since this class is static, we don't really need anything now, but that will change if we
        set the class attributes with a yaml file.
        In that case the yaml filename will needto be added to the aux_data bit

        See `here <https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree>`_
        for reference.
        """
        children = tuple()
        aux_data = {"mpi_rank": self.mpi_rank, "mpi_size": self.mpi_size}
        return (children, aux_data)

    def set_last_input_mapping(self, gds):
        self.last_input_channel_mapping = get_last_input_mapping(gds)


    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    ReplayEmulator,
    ReplayEmulator._tree_flatten,
    ReplayEmulator._tree_unflatten,
)
