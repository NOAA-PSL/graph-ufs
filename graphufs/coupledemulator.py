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
from graphcast.data_utils import extract_inputs_targets_forcings_coupled
from graphcast import data_utils
from graphcast.model_utils import dataset_to_stacked
from graphcast.losses import normalized_level_weights, normalized_latitude_weights

from .utils import (
    get_channel_index, get_last_input_mapping,
    add_emulator_arguments, set_emulator_options,
    get_num_params,
)


class ReplayCoupledEmulator:
    """A coupled emulator based on UFS Replay data. This manages all model configuration settings and normalization fields. Currently it is designed to be inherited for a specific use-case, and this could easily be generalized to read in settings via a configuration file (yaml, json, etc). Be sure to register any inherited class as a pytree for it to work with JAX.

    Example:
        see graph-ufs/prototypes/cp0_mini/mini_coupled_emulator.py for example usage
    """

    data_url = {"atm":""}
    norm_urls = {"atm":{
        "mean": "",
        "std": "",
        "stddiff": "",
    }}
    norm = dict()
    stacked_norm = dict()
    wb2_obs_url = ""
    local_store_path = None     # directory where zarr file, model weights etc are stored
    cache_data = None           # cache or use zarr dataset downloaded from GCS on disk

    # these could be moved to a yaml file later
    # task config options
    atm_input_variables = tuple()
    atm_target_variables = tuple()
    atm_forcing_variables = tuple()
    ocn_input_variables = tuple()
    ocn_target_variables = tuple()
    ocn_forcing_variables = tuple()
    ice_input_variables = tuple()
    ice_target_variables = tuple()
    ice_forcing_variables = tuple()
    land_input_variables = tuple()
    land_target_variables = tuple()
    land_forcing_variables = tuple()
    all_variables = tuple()     # this is created in __init__
    atm_pressure_levels = tuple()
    ocn_vert_levels = tuple()
    levels = list()             # created in __init__, has exact pfull level values
    latitude = tuple()
    longitude = tuple()
    tisr_integration_period = None  # TOA Incident Solar Radiation, integration period used in the function:
                                    # graphcast.solar_radiation.get_toa_incident_solar_radiation_for_xarray
                                    # default = self.delta_t, i.e. the ML model time step
                                    # Note: the value provided here has no effect unless "toa_incident_solar_radiation" is listed in "forcing_variables",
                                    # which indicates to graphcast that TISR needs to be computed.

    # time related
    delta_t = None              # the model time step, same for both ocean and atmosphere
    input_duration = None       # time covered by initial condition(s)
    target_lead_time = None     # when we compare to data, e.g. singular "3h", or many ["3h", "12h", "24h"]
    forecast_duration = None    # Created in __init__, total forecast time
    training_dates = tuple()    # bounds of training data (inclusive)
    testing_dates = tuple()     # bounds of testing data (inclusive)
    validation_dates = tuple()  # bounds of validation data (inclusive)

    # training protocol
    batch_size = None               # number of forecasts averaged over in loss per optim_step
    grad_clip_value = 32.
    num_batch_splits = None         # number of batch splits
    num_epochs = None               # number of epochs
    chunks_per_epoch = None         # number of chunks per epoch
    steps_per_chunk = None          # number of steps to train for in each chunk
    checkpoint_chunks = None        # save model after this many chunks are processed
    max_queue_size = None           # number of chunks in queue of data generators
    num_workers = None              # number of worker threads for data generators
    load_chunk = None               # load chunk into RAM, has the lowest memory usage if false
    store_loss = None               # store loss in a netcdf file
    use_preprocessed = None         # use pre-processed dataset

    # evaluation
    sample_stride = 1               # skip over initial conditions during evaluation by this stride
    evaluation_checkpoint_id = None # checkpoint used in evaluation scripts

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
    mesh2grid_edge_normalization_factor = None

    # loss weighting, defaults to GraphCast implementation
    weight_loss_per_latitude = True
    weight_loss_per_level = True
    weight_loss_per_channel = False
    atm_loss_weights_per_variable = {
        "tmp2m"         : 1.0,
        "ugrd10m"       : 0.1,
        "vgrd10m"       : 0.1,
        "pressfc"       : 0.1,
        "prateb_ave"    : 0.1,
    }

    ocn_loss_weights_per_variable = {
        "SSH"           : 1.0,
        "so"            : 0.1,
        "temp"          : 0.1,
    }
    ice_loss_weights_per_variable = {
        "icec"          : 1.0,
        "icetk"         : 0.1,
    }
    land_loss_weights_per_variable = {
        "soilm"         : 0.1,
    }
    loss_weights_per_variable = {}
    loss_weights_per_variable.update(atm_loss_weights_per_variable) 
    loss_weights_per_variable.update(ocn_loss_weights_per_variable) 
    loss_weights_per_variable.update(ice_loss_weights_per_variable)
    loss_weights_per_variable.update(land_loss_weights_per_variable)
    input_transforms = None
    output_transforms = None

    # this is used for initializing the state in the gradient computation
    grad_rng_seed = None
    init_rng_seed = None
    training_batch_rng_seed = None # used to randomize the training batches

    # data chunking options

    # for stacked graphcast
    last_input_channel_mapping = None

    def __init__(self, mpi_rank=None, mpi_size=None):

        if self.local_store_path is None:
            warnings.warn("ReplayCoupledEmulator.__init__: no local_store_path set, data will always be accessed remotely. Proceed with patience.")
        
        # make sure target_variables is a subset of input_variables
        if any(x not in self.atm_input_variables for x in self.atm_target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict atm target variables that are not also inputs")
        
        if any(x not in self.ocn_input_variables for x in self.ocn_target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict ocn target variables that are not also inputs")
       
        if any(x not in self.ice_input_variables for x in self.ice_target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict ice target variables that are not also inputs")

        if any(x not in self.land_input_variables for x in self.land_target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict land target variables that are not also inputs")

        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size

        # get vertical levels
        pfull = self._get_replay_vertical_levels(es_comp="atm")
        z_l = self._get_replay_vertical_levels(es_comp="ocn")

        latitude, longitude = self._get_replay_grid(self.resolution)
        self.latitude = tuple(float(x) for x in latitude)
        self.longitude = tuple(float(x) for x in longitude)
        self.atm_levels = list(
            pfull.sel(
                pfull=list(self.atm_pressure_levels),
                method="nearest",
            ).values
        )
        self.ocn_levels = list(
            z_l.sel(
                z_l=list(self.ocn_vert_levels),
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

        # Combine input and target variables from all components
        self.input_variables = tuple(set(self.atm_input_variables+self.ocn_input_variables+
            self.ice_input_variables+self.land_input_variables))
        self.target_variables = tuple(set(self.atm_target_variables+self.ocn_target_variables+
            self.ice_target_variables+self.land_target_variables))
        self.forcing_variables = tuple(set(self.atm_forcing_variables+self.ocn_forcing_variables+
            self.ice_forcing_variables+self.land_forcing_variables))

        # try/except logic to support original graphcast.graphcast.TaskConfig
        # since I couldn't get inspect.getfullargspec to work
        try:
            self.task_config = TaskConfig(
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                forcing_variables=self.forcing_variables,
                pressure_levels=tuple(set(self.atm_levels)),
                ocn_vert_levels=sorted(tuple(set(self.ocn_levels))),
                input_duration=self.input_duration,
                longitude=self.longitude,
                latitude=self.latitude,
            )
        except ValueError:
            self.task_config = TaskConfig(
                input_variables=self.input_variables,
                target_variables=self.target_variables,
                forcing_variables=self.forcing_variables,
                pressure_levels=tuple(set(self.atm_levels)),
                ocn_vert_levels=sorted(tuple(set(self.ocn_levels))),
                input_duration=self.input_duration,
            )

        self.all_variables = tuple(
                set(self.input_variables +
                    self.target_variables +
                    self.forcing_variables)
            )
        
        # convert some types
        self.delta_t = pd.Timedelta(self.delta_t)
        self.input_duration = pd.Timedelta(self.input_duration)
        lead_times, duration = data_utils._process_target_lead_times_and_get_duration(self.target_lead_time)
        self.forecast_duration = duration

        logging.debug(f"target_lead_time: {self.target_lead_time}")
        logging.debug(f"lead_times: {lead_times}")
        logging.debug(f"self.forecast_duration: {self.forecast_duration}")
        logging.debug(f"self.time_per_forecast: {self.time_per_forecast}")
        logging.debug(f"self.n_input: {self.n_input}")
        logging.debug(f"self.n_forecast: {self.n_forecast}")
        logging.debug(f"self.n_target: {self.n_target}")

        # set normalization here so that we can jit compile with this class
        # a bit annoying, have to copy datatypes here to avoid the Ghost Bus problem
        self.norm_urls = self.norm_urls.copy()
        self.norm = dict()
        self.stacked_norm = dict()
        self.set_normalization()
        self.set_stacked_normalization()

        # TOA Incident Solar Radiation integration period
        if self.tisr_integration_period is None:
            self.tisr_integration_period = self.delta_t

    @property
    def name(self):
        return str(type(self).__name__)

    @property
    def time_per_forecast(self):
        return self.forecast_duration + self.input_duration

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
        return self.forecast_duration // self.delta_t

    @property
    def extract_kwargs(self):
        kw = {k: v for k, v in dataclasses.asdict(self.task_config).items() if k not in ("latitude", "longitude")}
        kw["target_lead_times"] = self.target_lead_time
        kw["integration_period"] = self.tisr_integration_period
        return kw

    @property
    def local_data_path(self):
        return os.path.join(self.local_store_path, "data.zarr")

    @property
    def checkpoint_dir(self):
        return os.path.join(self.local_store_path, "models")


    def open_atm_dataset(self, **kwargs):
        xds = xr.open_zarr(self.data_url["atm"], storage_options={"token": "anon"}, **kwargs)
        return xds
    
    def open_ocn_dataset(self, **kwargs):
        xds = xr.open_zarr(self.data_url["ocn"], storage_options={"token": "anon"}, **kwargs)
        return xds
    
    def open_ice_dataset(self, **kwargs):
        xds = xr.open_zarr(self.data_url["ice"], storage_options={"token": "anon"}, **kwargs)
        return xds

    def open_land_dataset(self, **kwargs):
        xds = xr.open_zarr(self.data_url["land"], storage_options={"token": "anon"}, **kwargs)
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


    def subsample_dataset(self, xds, es_comp="atm", new_time=None):
        """Get the subset of the data that we want in terms of time, vertical levels, and variables

        Args:
            xds (xarray.Dataset): with replay data
            es_comp (string): earth system component, "atm/ocn/coupled/ice/land"
            new_time (pandas.Daterange or similar, optional): time vector to select from the dataset

        Returns:
            newds (xarray.Dataset): subsampled/subset that we care about
        """

        # select our vertical levels
        if es_comp.lower() == "atm".lower():
            xds = xds.sel(pfull=self.atm_levels)
            myvars = list(x for x in set(self.atm_input_variables+self.atm_target_variables+self.atm_forcing_variables) if x in xds)
        elif es_comp.lower() == "ocn".lower():
            xds = xds.sel(z_l=self.ocn_levels)
            myvars = list(x for x in set(self.ocn_input_variables+self.ocn_target_variables+self.ocn_forcing_variables) if x in xds)
        elif es_comp.lower() == "ice".lower():
            myvars = list(x for x in set(self.ice_input_variables+self.ice_target_variables+self.ice_forcing_variables) if x in xds)
        elif es_comp.lower() == "land".lower():
            myvars = list(x for x in set(self.land_input_variables+self.land_target_variables+self.land_forcing_variables) if x in xds)
        elif es_comp.lower() == "coupled".lower():
            xds = xds.sel(pfull=self.atm_levels, z_l=self.ocn_levels)
            myvars = list(x for x in self.all_variables if x in xds)
        else:
            raise ValueError("Unknown earth system component: only atm, ocn, ice, and land are supported")

        # only grab variables we care about
        xds = xds[myvars]

        if new_time is not None:
            xds = xds.sel(time=new_time)
        
        # mask nans in ocean target variables
        if es_comp.lower() == "ocn".lower() or es_comp.lower() == "ice" or es_comp.lower() == "land":
            xds = xds.fillna(0)

        # if we have any transforms to apply, do it here
        xds = self.transform_variables(xds)

        return xds


    def transform_variables(self, xds):
        """e.g. transform spfh -> log(spfh), but keep the name the same for ease with GraphCast code"""
        if self.input_transforms is not None:
            for key, mapping in self.input_transforms.items():
                logging.info(f"{type(self).__name__}: transforming {key} -> {mapping.__name__}({key})")
                with xr.set_options(keep_attrs=True):
                    xds[key] = mapping(xds[key])
                xds[key].attrs["transformation"] = f"this variable shows {mapping.__name__}({key})"
        return xds


    def check_for_ints(self, xds):
        """Turn data variable integers into floats, because otherwise the normalization in GraphCast goes haywire
        """

        for key in xds.data_vars:
            if "int" in str(xds[key].dtype):
                logging.debug(f"Converting {key} from {xds[key].dtype} to float32")
                xds[key] = xds[key].astype(np.float32)
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
            logging.info(f"Downloading {mode} data for {len(all_new_time)} time stamps.")
            # atm
            xds_atm = xr.open_zarr(self.data_url["atm"], storage_options={"token": "anon"})
            all_xds_atm = self.subsample_dataset(xds_atm, new_time=all_new_time, es_comp="atm")
            # ocn
            xds_ocn = xr.open_zarr(self.data_url["ocn"], storage_options={"token": "anon"})
            all_xds_ocn = self.subsample_dataset(xds_ocn, new_time=all_new_time, es_comp="ocn")
            all_xds_ocn = all_xds_ocn.rename({"lat":"grid_yt", "lon":"grid_xt"})
            # ice
            xds_ice = xr.open_zarr(self.data_url["ice"], storage_options={"token": "anon"})
            all_xds_ice = self.subsample_dataset(xds_ice, new_time=all_new_time, es_comp="ice")
            # land
            xds_land = xr.open_zarr(self.data_url["land"], storage_options={"token": "anon"})
            all_xds_land = self.subsample_dataset(xds_land, new_time=all_new_time, es_comp="land")

            all_xds = xr.merge([all_xds_atm, all_xds_ocn, all_xds_ice, all_xds_land])
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
                # atm
                missing_xds_atm = self.open_atm_dataset()
                missing_xds_atm = self.subsample_dataset(missing_xds_atm, new_time=list(missing_dates), es_comp="atm")
                # ocn
                missing_xds_ocn = self.open_ocn_dataset()
                missing_xds_ocn = self.subsample_dataset(missing_xds_ocn, new_time=list(missing_dates), es_comp="ocn")
                missing_xds_ocn = missing_xds_ocn.rename({"lat":"grid_yt", "lon":"grid_xt"})
                # ice
                missing_xds_ice = self.open_ice_dataset()
                missing_xds_ice = self.subsample_dataset(missing_xds_ice, new_time=list(missing_dates), es_comp="ice")
                # land
                missing_xds_land = self.open_land_dataset()
                missing_xds_land = self.subsample_dataset(missing_xds_land, new_time=list(missing_dates), es_comp="land")
                
                missing_xds = xr.merge([missing_xds_atm, missing_xds_ocn, missing_xds_ice, missing_xds_land])
                missing_xds.to_zarr(self.local_data_path, append_dim="time")
                # now that the data on disk is complete, reopen the dataset from disk
                all_xds = xr.open_zarr(self.local_data_path)
            else:
                all_xds = xds_on_disk

        all_xds = self.check_for_ints(all_xds)
        return all_xds

    @staticmethod
    def divide_into_slices(N, K):
        """
        Divide N items into K groups and return an array of slice objects.

        Args:
            N (int): Total number of items.
            K (int): Number of groups.
        Returns:
           list of slice: A list containing slice objects for each group.
        """
        base_size = N // K
        extra_items = N % K

        slices = []
        start = 0
        for i in range(K):
            if i < extra_items:
                end = start + base_size + 1
                slices.append(slice(start, end))
            else:
                end = start + base_size
                slices.append(slice(start - (1 if extra_items else 0), end))
            start = end

        return slices

    @staticmethod
    def rechunk(xds):
        chunksize = {
            "optim_step": 1,
            "batch": -1,
            "time": -1,
            "level": -1,
            "lat": -1,
            "lon": -1,
        }
        chunksize = {k:v for k,v in chunksize.items() if k in xds}
        xds = xds.chunk(chunksize)
        return xds

    def get_batches(
        self,
        n_optim_steps=None,
        drop_cftime=True,
        mode="training",
    ):
        """Get a dataset with all the batches of data necessary for training

        Note:
            Here we're using target_lead_time as a single value, see graphcast.data_utils.extract ... where it could be multi valued. However, since we are using it to compute the total forecast time per batch soit seems more straightforward as a scalar.

        Args:
            n_optim_steps (int, optional): number of training batches to grab ... number of times we will update the parameters during optimization. If not specified, use as many as are available based on the available training data.
            drop_cftime (bool, optional): may be useful for debugging
            mode (str, optional): can be either "training", "validation" or "testing"
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
        has_preprocessed = False
        
        if self.use_preprocessed:
            # chunks zarr datasets
            xds_chunks = {
                "inputs": [None] * n_chunks,
                "targets": [None] * n_chunks,
                "forcings": [None] * n_chunks,
                "inittimes": [None] * n_chunks,
            }

            # open chunk files if they exist
            try:
                message = f"Chunks for {mode}: {n_chunks}"
                for chunk_id in range(n_chunks):
                    base_name = f"{self.local_store_path}/extracted/{mode}-chunk-{chunk_id:04d}-of-{n_chunks:04d}-rank-{self.mpi_rank:03d}-of-{self.mpi_size:03d}-bs-{self.batch_size}-"
                    xds_chunks["inputs"][chunk_id] = xr.open_zarr(f"{base_name}inputs.zarr")
                    xds_chunks["targets"][chunk_id] = xr.open_zarr(f"{base_name}targets.zarr")
                    xds_chunks["forcings"][chunk_id] = xr.open_zarr(f"{base_name}forcings.zarr")
                    if mode == "testing":
                        xds_chunks["inittimes"][chunk_id] = xr.open_zarr(f"{base_name}inittimes.zarr")
                n_steps = len(xds_chunks["inputs"][0]["optim_step"])
                message += f" each with {n_steps} steps"
                logging.info(message)
                has_preprocessed = True
            except:
                has_preprocessed = False

        # raw dataset
        if not has_preprocessed:
            all_new_time = self.get_time(mode=mode)

            # split the dataset _equally_ across nodes to prevent hangs
            # Note: The possible overlaps maybe a problem if/when testing is parallelized
            if self.mpi_size > 1:
                slices = self.divide_into_slices(len(all_new_time), self.mpi_size)
                all_new_time = all_new_time[slices[self.mpi_rank]]
                logging.info(f"Data for {mode} MPI rank {self.mpi_rank}: {all_new_time[0]} to {all_new_time[-1]} : {len(all_new_time)} time stamps.")

            # download the data
            all_xds = self.get_the_data(all_new_time=all_new_time, mode=mode)
            # split dataset into chunks
            slices = self.divide_into_slices(len(all_new_time), n_chunks)
            all_new_time_chunks = []
            for sl in slices:
                all_new_time_chunks.append(all_new_time[sl])

            # print chunk boundaries
            message = f"Chunks for {mode}: {len(all_new_time_chunks)} each with {len(all_new_time_chunks[0])} time stamps"
            logging.info(message)

        # list of chunk ids
        chunk_ids = [i for i in range(n_chunks)]
        n_optim_steps_arg = n_optim_steps

        # loop forever
        while True:

            # shuffle chunks
            if mode != "testing":
                random.shuffle(chunk_ids)

            # iterate over all chunks
            for chunk_id in chunk_ids:
                # check for pre-processed inputs
                if self.use_preprocessed:
                    if xds_chunks["inputs"][chunk_id] is not None:
                        logging.debug(f"\nReusing {mode} chunk {chunk_id}.")
                        inputs = xds_chunks["inputs"][chunk_id]
                        targets = xds_chunks["targets"][chunk_id]
                        forcings = xds_chunks["forcings"][chunk_id]
                        inittimes = None
                        if mode == "testing":
                            inittimes = xds_chunks["inittimes"][chunk_id]
                        yield inputs, targets, forcings, inittimes
                        continue
                    else:
                        logging.debug(f"\nOpening {mode} chunk {chunk_id} from scratch.")
                
                # chunk start and end times
                new_time = all_new_time_chunks[chunk_id]
                start = new_time[0]
                end = new_time[-1]

                # figure out duration of IC(s), forecast, all of training
                data_duration = end - start

                n_max_forecasts = (data_duration - self.time_per_forecast) // self.delta_t + 1
                if n_max_forecasts <= 0:
                    raise ValueError(f"n_max_forecasts for {mode} is {n_max_forecasts}")

                n_max_optim_steps = math.ceil(n_max_forecasts / self.batch_size)
                n_optim_steps = n_max_optim_steps if n_optim_steps_arg is None else n_optim_steps_arg
                n_forecasts = n_optim_steps * self.batch_size
                n_forecasts = min(n_forecasts, n_max_forecasts)

                # note that this max can be violated if we sample with replacement ...
                # but I'd rather just work with epochs and use all the data
                if n_optim_steps > n_max_optim_steps:
                    n_optim_steps = n_max_optim_steps
                    warnings.warn(f"There's less data than the number of batches requested, reducing n_optim_steps to {n_optim_steps}")

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
                xds = self.subsample_dataset(all_xds, es_comp="coupled", new_time=new_time)
                xds = xds.rename({
                    "pfull": "level",
                    "grid_xt": "lon",
                    "grid_yt": "lat",
                    "time": "datetime",
                    })
                xds = xds.drop(["cftime", "ftime"])

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
                    this_input, this_target, this_forcing = extract_inputs_targets_forcings_coupled(
                        batch,
                        **self.extract_kwargs,
                    )

                    # note that the optim_step dim has to be added after the extract_inputs_targets_forcings call
                    inputs.append(this_input.expand_dims({"optim_step": [k]}))
                    targets.append(this_target.expand_dims({"optim_step": [k]}))
                    forcings.append(this_forcing.expand_dims({"optim_step": [k]}))

                    if mode == "testing":
                        # fix this later for batch_size != 1
                        this_inittimes = batch.datetime.isel(time=self.n_input-1)
                        this_inittimes = this_inittimes.to_dataset(name="inittimes")
                        inittimes.append(this_inittimes.expand_dims({"optim_step": [k]}))

                # write chunks to disk
                base_name = f"{self.local_store_path}/extracted/{mode}-chunk-{chunk_id:04d}-of-{n_chunks:04d}-rank-{self.mpi_rank:03d}-of-{self.mpi_size:03d}-bs-{self.batch_size}-"
                def combine_chunk_save(xds, name):
                    xds = xr.combine_by_coords(xds)
                    if name != "inittimes":
                        xds = self.rechunk(xds)
                    if self.use_preprocessed:
                        file_name = f"{base_name}{name}.zarr"
                        xds.to_zarr(file_name)
                        xds.close()
                        xds = xr.open_zarr(file_name)
                        xds_chunks[name][chunk_id] = xds
                    return xds

                inputs = combine_chunk_save(inputs, "inputs")
                targets = combine_chunk_save(targets, "targets")
                forcings = combine_chunk_save(forcings, "forcings")
                if mode == "testing":
                    inittimes = combine_chunk_save(inittimes, "inittimes")
                else:
                    inittimes = None

                yield inputs, targets, forcings, inittimes



    def set_normalization(self, **kwargs):
        """Load the normalization fields into memory

        Returns:
            mean_by_level, stddev_by_level, diffs_stddev_by_level (xarray.Dataset): with normalization fields
        """

        def open_normalization(moment):

            # try to read locally first
            local_path = os.path.join(
                self.local_store_path,
                "normalization",
                "coupled",
                os.path.basename(self.norm_urls["atm"][moment]),
            )



            if os.path.isdir(local_path):
                xds = xr.open_zarr(local_path)
                myvars = list(x for x in self.all_variables if x in xds)
                xds = xds[myvars]
                xds = xds.load()
                foundit = True

            else:
                kwargs = {"storage_options": {"token": "anon"}} if any(x in self.norm_urls[comp][moment] for x in ["gs://", "gcs://"] for comp in ["atm", "ocn", "ice", "land"]) else {}
                xds_atm = xr.open_zarr(self.norm_urls["atm"][moment], **kwargs)
                xds_ocn = xr.open_zarr(self.norm_urls["ocn"][moment], **kwargs)
                xds_ice = xr.open_zarr(self.norm_urls["ice"][moment], **kwargs)
                xds_land = xr.open_zarr(self.norm_urls["land"][moment], **kwargs)
                vars_atm = list(x for x in self.all_variables if x in xds_atm)
                vars_ocn = list(x for x in self.all_variables if x in xds_ocn)
                vars_ice = list(x for x in self.all_variables if x in xds_ice)
                vars_land = list(x for x in self.all_variables if x in xds_land)
                
                # keep attributes in order to distinguish static from time varying components
                with xr.set_options(keep_attrs=True):

                    if self.input_transforms is not None:
                        for key, transform_function in self.input_transforms.items():

                            # make sure e.g. log_spfh is in the dataset
                            transformed_key = f"{transform_function.__name__}_{key}" # e.g. log_spfh
                            assert transformed_key in xds, \
                                f"Emulator.set_normalization: couldn't find {transformed_key} in {moment} normalization dataset"
                            # there's a chance the original, e.g. spfh, is not in the dataset
                            # if it is, replace it with e.g. log_spfh
                            if key in myvars:
                                idx = myvars.index(key)
                                myvars[idx] = transformed_key
                    xds_atm = xds_atm[vars_atm]
                    if self.input_transforms is not None:
                        for key, transform_function in self.input_transforms.items():
                            transformed_key = f"{transform_function.__name__}_{key}" # e.g. log_spfh
                            idx = myvars.index(transformed_key)
                            myvars[idx] = key

                            # necessary for graphcast.dataset to stacked operations
                            xds = xds.rename({transformed_key: key})
                    xds_atm = xds_atm.sel(pfull=self.atm_levels)
                    xds_ocn = xds_ocn[vars_ocn]
                    xds_ocn = xds_ocn.sel(z_l=self.ocn_levels)
                    xds_ice = xds_ice[vars_ice]
                    xds_land = xds_land[vars_land]
                    xds = xr.merge([xds_atm, xds_ocn, xds_ice, xds_land])
                    xds = xds.load()
                    xds = xds.rename({"pfull": "level"})
                xds.to_zarr(local_path)
            return xds

        for moment in ["mean", "std", "stddiff"]:
            self.norm[moment] = open_normalization(moment)

    def set_stacked_normalization(self):

        assert len(self.norm["mean"]) > 0, "normalization not set, call Emulator.set_normalization()"

        def open_normalization(moment):

            # try to read locally first
            inputs_path = os.path.join(
                self.local_store_path,
                "stacked-normalization",
                "inputs",
                os.path.basename(self.norm_urls["atm"][moment]),
            )
            targets_path = os.path.join(
                self.local_store_path,
                "stacked-normalization",
                "targets",
                os.path.basename(self.norm_urls["atm"][moment]),
            )

            if os.path.isdir(inputs_path) and os.path.isdir(targets_path):
                inputs = xr.open_zarr(inputs_path)
                inputs = inputs["inputs"].load()
                targets = xr.open_zarr(targets_path)
                targets = targets["targets"].load()

            else:
                inputs, targets = self.normalization_to_stacked(self.norm[moment], preserved_dims=tuple())
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
            # replicate time varying variables
            for key in norms.data_vars:
                if "description" in xds[key].attrs and "time" in xds[key].attrs["description"]:
                    norms[key] = xr.concat(
                        [norms[key].copy() for _ in range(n_time)],
                        dim="time",
                    )
            dimorder = ("batch", "time", "level", "z_l", "lat", "lon")
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
            # for the subsampled case, we want to compute weights
            # on the parent 0.25 degree grid, and subsample it
            # because the pole points in the subsampled version
            # are not equidistant between their neighbor and the pole
            if "0.25-degree-subsampled" in self.data_url:
                lat = xr.open_zarr(
                    self.data_url.replace("-subsampled",""),
                    storage_options={"token":"anon"}
                )["grid_yt"].rename({"grid_yt": "lat"})
                lat_weights = normalized_latitude_weights(lat)
                lat_weights = lat_weights.isel(lat=slice(None, None, 4))
            else:
                lat_weights = normalized_latitude_weights(xtargets)

            lat_weights = lat_weights.data[...,None][...,None]

            weights *= lat_weights
            weights /= (len(xtargets["lon"]) * len(xtargets["lat"]))


        # 2. compute per variable weighting
        # Either do this per channel, or per variable as in GraphCast
        n_channels = targets.shape[-1]
        if self.weight_loss_per_channel:
            for ichannel in range(n_channels):
                weights[..., ichannel] /= n_channels

        else:
            #   a. incorporate user-specified variable weights
            target_idx = get_channel_index(xtargets)
            var_count = {k: 0 for k in self.target_variables}
            for ichannel in range(targets.shape[-1]):
                varname = target_idx[ichannel]["varname"]
                var_count[varname] += 1
                if varname in self.loss_weights_per_variable:
                    weights[..., ichannel] *= self.loss_weights_per_variable[varname]

            #   b. take average within variable, so if we have 3 levels of 1 var, divide by 3*n_latitude*n_longitude
            for ichannel in range(targets.shape[-1]):
                varname = target_idx[ichannel]["varname"]
                weights[..., ichannel] /= var_count[varname]


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
    def _get_replay_vertical_levels(es_comp="atm"):
        if es_comp.lower() == "atm".lower():
            pfull_path = os.path.join(os.path.dirname(__file__), "replay_vertical_levels.yaml")
            with open(pfull_path, "r") as f:
                pfull = yaml.safe_load(f)["pfull"]
            return xr.DataArray(pfull, coords={"pfull": pfull}, dims="pfull")

        elif es_comp.lower() == "ocn".lower():
            z_l_path = os.path.join(os.path.dirname(__file__), "replay_vertical_levels.yaml")
            with open(z_l_path, "r") as f:
                z_l = yaml.safe_load(f)["z_l"]
            return xr.DataArray(z_l, coords={"z_l": z_l}, dims="z_l")
        
        else:
            raise ValueError("only atm and ocn are supported in 3D")

    def _get_replay_grid(self, resolution: int | float):
        if int(resolution) == 1:
            if "0.25-degree-subsampled" in self.data_url["atm"]:
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
        
        # Check the total number of trainable parameters in this checkpoint
        print("Total number of trainable parameters:", get_num_params(ckpt))

        logging.info(f"Stored checkpoint: {ckpt_path}")

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
    ReplayCoupledEmulator,
    ReplayCoupledEmulator._tree_flatten,
    ReplayCoupledEmulator._tree_unflatten,
)
