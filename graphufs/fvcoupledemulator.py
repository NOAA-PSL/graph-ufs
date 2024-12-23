import logging
import warnings
import os
import numpy as np
import xarray as xr
import pandas as pd
import yaml
try:
    import flox
    _has_flox = True
except ImportError:
    _has_flox = False
from typing import Optional
from ufs2arco import Layers2Pressure
from graphcast.graphcast import ModelConfig, TaskConfig
from graphcast import data_utils

from .coupledemulator import ReplayCoupledEmulator
from .fvemulator import (
    get_new_vertical_grid as get_new_atm_vertical_grid,
    fv_vertical_regrid as atm_fv_vertical_regrid,
)

class FVCoupledEmulator(ReplayCoupledEmulator):
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
        vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_vertical_levels.yaml")
        with open(vcoord_path, "r") as f:
            vcoords = yaml.safe_load(f)
        latitude, longitude = self._get_replay_grid(self.resolution)
        self.latitude = tuple(float(x) for x in latitude)
        self.longitude = tuple(float(x) for x in longitude)

        # TODO Here
        nds = get_new_vertical_grid(list(self.interfaces))
        self.levels = list(nds["pfull"].values)
        self.pressure_levels = tuple(nds["pfull"].values)

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

def get_new_vertical_grid(interfaces, comp):

    # Create the parent vertical grid via layers2pressure object
    if comp.lower()=="atm".lower():
        nds = get_new_atm_vertical_grid(interfaces)
    
    elif comp.lower()=="ocn".lower():
        vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_vertical_levels.yaml")
        with open(vcoord_path, "r") as f:
            vcoords = yaml.safe_load(f)
        replay_mom6_interface = xr.DataArray(np.array(vcoords["z_i"]),
                coords={"z_i":np.array(vcoords["z_i"])},
                dims=["z_i"],
                )
        #replay_mom6_layerinterface = _get_ocn_xds(np.array(vcoords["z_i"]))
        nz_i = replay_mom6_interface.sel(z_i=interfaces, method='nearest').values
        nz_l = (nz_i[1:] + nz_i[:-1])/2
        nds = _get_ocn_xds(nz_i, nz_l)
        
    return nds

def _get_ocn_xds(
        z_i: np.ndarray,
        z_l: Optional[np.ndarray] = None,
    ) -> xr.Dataset:
    
        if z_i is not None:
            z_i = xr.DataArray(
                z_i,
                coords={"z_i":z_i},
                dims=["z_i"],
            )
            if z_l is not None:
                z_l = xr.DataArray(
                    z_l,
                    coords={"z_l":z_l},
                    dims=["z_l"],
                )

                xds = xr.Dataset({
                    "z_i": z_i,
                    "z_l": z_l,
                })
                xds = xds.set_coords(["z_i", "z_l"])
            
            else:
                xds = xr.Dataset({
                    "z_i":z_i,
                })
                xds = xds.set_coords(["z_i"])
            
            return xds
        
        else:
            raise ValueError("Both z_i and z_l are empty")

def fv_vertical_regrid(xds, interfaces, keep_delz=False):
    """Vertically regrid a dataset based on approximately located interfaces
    by "approximately" we mean to grab the nearest neighbor to the values in interfaces

    Args:
        xds (xr.Dataset)
        interfaces (array_like)

    Returns:
        nds (xr.Dataset): with vertical averaging
    """
    # 2D or 3D: no vertical regridding required for 2D
    vars2d = [x for x in xds.data_vars if not ("pfull" in xds[x].dims or "z_l" in xds[x].dims)]
    vars3d = {}
    vars3d["atm"] = [x for x in xds.data_vars if "pfull" in xds[x].dims]
    vars3d["ocn"] = [x for x in xds.data_vars if "z_l" in xds[x].dims]

    if vars3d:
        if vars3d["atm"]:
            logging.info(f"3D atmospheric variable detected: {vars3d} ")
            # create a new dataset with the new vertical grid 
            nds = atm_fv_vertical_regrid(xds, interfaces, keep_delz=keep_delz)

        if vars3d["ocn"]:
            logging.info(f"3D ocean variable detected:{vars3d} ")
            # create a new dataset with the new vertical grid
            nds = get_new_vertical_grid(interfaces, "ocn")

            # Regrid static layer thickness and get weighting
            vcoord_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"replay_vertical_levels.yaml")
            with open(vcoord_path, "r") as f:
                vcoords = yaml.safe_load(f)
            xa_interface = xr.DataArray(np.array(vcoords["z_i"]),
                    coords={"z_i":np.array(vcoords["z_i"])},
                    dims=["z_i"],
                    )

            xa_dz = xa_interface.diff(dim="z_i")
            xa_dz = xa_dz.assign_coords({"z_i":xds.coords["z_l"].values})
            xa_dz = xa_dz.rename({"z_i":"z_l"})
            nds["dz"] = xa_dz.groupby_bins(
                    "z_l",
                    bins=nds["z_i"],
            ).sum()
            dz_inverse = 1/nds["dz"]
            # do the regridding for all variables now.
            for key in vars3d["ocn"]:
                with xr.set_options(keep_attrs=True):
                    weighted_mult = xds[key]*xa_dz
                    nds[key] = dz_inverse*(
                        weighted_mult.groupby_bins(
                            "z_l",
                            bins=nds["z_i"],
                        ).sum()
                    )
                nds[key].attrs = xds[key].attrs.copy()
                # set the coordinates
                nds = nds.set_coords("z_l")
                nds["z_l_bins"] = nds["z_l_bins"].swap_dims({"z_l_bins": "z_l"})
                with xr.set_options(keep_attrs=True):
                    nds[key] = nds[key].swap_dims({"z_l_bins": "z_l"})
                nds[key].attrs["regridding"] = "layer thickness weighted average in vertical, new coordinate bounds represented by 'z_l_bins'"

                nds = nds.drop_vars("z_l_bins")
            nds["dz"] = nds["dz"].swap_dims({"z_l_bins": "z_l"})

    if vars2d:
        logging.info("2D atmospheric variable detected: no regridding applied")
        nds = xr.Dataset(
                data_vars={}, 
                coords=xds.coords,
                attrs=xds.attrs,
        )
        for v in vars2d:
            nds[v] = xds[v]
    
    return nds
