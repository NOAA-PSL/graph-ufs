import logging
import warnings
import numpy as np
import xarray as xr
import pandas as pd
try:
    import flox
    _has_flox = True
except ImportError:
    _has_flox = False

from ufs2arco import Layers2Pressure
from graphcast.graphcast import ModelConfig, TaskConfig
from graphcast import data_utils

from .emulator import ReplayEmulator

class FVEmulator(ReplayEmulator):
    interfaces = None # Note the these values can be approximate, we'll grab nearest neighbors to Replay dataset


    def __init__(self, mpi_rank=None, mpi_size=None):

        if not _has_flox:
            warnings.warn("Could not import flox, install with 'conda install -c conda-forge flox' for faster volume averaging (i.e. groupby operations)")

        if self.local_store_path is None:
            warnings.warn("FVEmulator.__init__: no local_store_path set, data will always be accessed remotely. Proceed with patience.")

        if any(x not in self.input_variables for x in self.target_variables):
            raise NotImplementedError(f"GraphUFS cannot predict target variables that are not also inputs")

        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size

        latitude, longitude = self._get_replay_grid(self.resolution)
        self.latitude = tuple(float(x) for x in latitude)
        self.longitude = tuple(float(x) for x in longitude)

        # TODO Here
        nds = get_new_vertical_grid(list(self.interfaces))
        self.levels = list(nds["pfull"].values)
        self.pressure_levels = tuple(nds["pfull"].values)
        self.ak = nds["ak"]
        self.bk = nds["bk"]

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


    def subsample_dataset(self, xds, new_time=None):

        # make sure that we have 'delz' for the vertical averaging
        allvars = list(self.all_variables)
        if "delz" not in self.all_variables:
            allvars.append("delz")
        myvars = list(x for x in allvars if x in xds)
        xds = xds[myvars]

        if new_time is not None:
            xds = xds.sel(time=new_time)

        xds = fv_vertical_regrid(
            xds,
            interfaces=list(self.interfaces),
        )

        # if we didn't want delz and just kept it for regridding, remove it here
        xds = xds[[x for x in self.all_variables if x in xds]]

        # perform transform after vertical averaging, less subsceptible to noisy results
        xds = self.transform_variables(xds)
        return xds

def get_new_vertical_grid(interfaces):


    # Create the parent vertical grid via layers2pressure object
    replay_layers = Layers2Pressure()
    phalf = replay_layers.phalf.sel(phalf=interfaces, method="nearest")

    # Make a new Layers2Pressure object, which has the subsampled vertical grid
    # note that pfull gets defined internally
    child_layers = Layers2Pressure(
        ak=replay_layers.xds["ak"].sel(phalf=phalf),
        bk=replay_layers.xds["bk"].sel(phalf=phalf),
    )
    nds = child_layers.xds.copy(deep=True)
    return nds


def fv_vertical_regrid(xds, interfaces, keep_delz=False):
    """Vertically regrid a dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in interfaces

    Args:
        xds (xr.Dataset)
        interfaces (array_like)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """
    # create a new dataset with the new vertical grid
    nds = get_new_vertical_grid(interfaces)


    # if the dataset has somehow already renamed pfull -> level, rename to pfull for Layers2Pressure computations
    has_level_not_pfull = False
    if "level" in xds.dims and "pfull" not in xds.dims:
        with xr.set_options(keep_attrs=True):
            xds = xds.rename({"pfull": "level"})

    # Regrid vertical distance, and get weighting
    delz = xds["delz"].groupby_bins(
        "pfull",
        bins=nds["phalf"],
    ).sum()
    new_delz_inverse = 1/delz

    # Do the regridding
    vars2d = [x for x in xds.data_vars if "pfull" not in xds[x].dims]
    vars3d = [x for x in xds.data_vars if "pfull" in xds[x].dims and x != "delz"]
    for key in vars3d:
        with xr.set_options(keep_attrs=True):
            nds[key] = new_delz_inverse * (
                (
                    xds[key]*xds["delz"]
                ).groupby_bins(
                    "pfull",
                    bins=nds["phalf"],
                ).sum()
            )
        nds[key].attrs = xds[key].attrs.copy()

    nds = nds.set_coords("pfull")
    nds["pfull_bins"] = nds["pfull_bins"].swap_dims({"pfull_bins": "pfull"})
    for key in vars3d:
        with xr.set_options(keep_attrs=True):
            nds[key] = nds[key].swap_dims({"pfull_bins": "pfull"})
        nds[key].attrs["regridding"] = "delz weighted average in vertical, new coordinate bounds represented by 'phalf'"
    for v in vars2d:
        nds[v] = xds[v]

    if keep_delz:
        delz = delz.swap_dims({"pfull_bins": "pfull"})
        nds["delz"] = delz

    # unfortunately, cannot store the pfull_bins due to this issue: https://github.com/pydata/xarray/issues/2847
    nds = nds.drop_vars("pfull_bins")
    return nds
